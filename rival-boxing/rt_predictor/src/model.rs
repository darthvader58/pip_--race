use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::{fs, path::Path};
use tch::{kind::Kind, CModule, Device, IndexOp, Tensor};

#[derive(Deserialize)]
struct MetaJson {
    feat_list: Vec<String>,
    in_dim: Option<usize>,
}

pub struct Model {
    model: CModule,
    device: Device,
    pub n_actions: i64, // expected 2: [no_pit, pit]
    pub n_quant: i64,   // number of quantiles per action
}

impl Model {
    pub fn new(model_path: &str, meta_path: &str) -> Result<(Self, usize, Vec<String>)> {
        let device = Device::Cpu;

        // Load meta.json to get feature ordering and input dim
        let meta_txt = fs::read_to_string(Path::new(meta_path))
            .with_context(|| format!("failed to read meta at {}", meta_path))?;
        let meta: MetaJson =
            serde_json::from_str(&meta_txt).with_context(|| "failed to parse meta.json")?;

        let feat_list = meta.feat_list;
        let in_dim = meta.in_dim.unwrap_or(feat_list.len());

        // Load TorchScript model
        let model = CModule::load_on_device(model_path, device)
            .with_context(|| format!("failed to load TorchScript {}", model_path))?;

        // Probe output shape with a dummy forward — expect [B=1, A, Q]
        let dummy = Tensor::zeros([1, in_dim as i64], (Kind::Float, device));
        let t = model.forward_ts(&[dummy])?;
        let sz = t.size();
        if sz.len() != 3 || sz[0] != 1 {
            bail!("unexpected model output size: {:?}", sz);
        }
        let n_actions = sz[1];
        let n_quant = sz[2];

        Ok((
            Self {
                model,
                device,
                n_actions,
                n_quant,
            },
            in_dim,
            feat_list,
        ))
    }

    /// Returns (p2, p3): probability to box within 2 and 3 laps.
    /// Heuristic: mean over quantiles to get Q for each action, then
    /// gap = Q(pit) - Q(no_pit); map gap → probability via sigmoid.
    pub fn predict_probs(&self, x: &[f32], in_dim_expected: usize) -> Result<(f32, f32)> {
        if x.len() != in_dim_expected {
            bail!(
                "feature length mismatch: got {}, expected {}",
                x.len(),
                in_dim_expected
            );
        }

        let input = Tensor::from_slice(x)
            .reshape([1, in_dim_expected as i64])
            .to_device(self.device);

        // Forward: [1, A, Q]
        let t = self.model.forward_ts(&[input])?;

        // Mean over quantiles (dim=2) -> [1, A]
        let q_per_action = t.mean_dim(&[2i64][..], false, Kind::Float);
        let sz = q_per_action.size();
        if sz.len() != 2 || sz[0] != 1 || sz[1] < 2 {
            bail!("unexpected q_per_action shape: {:?}", sz);
        }

        // gap = Q(pit) - Q(no_pit)
        let q_no = q_per_action.i((0, 0));
        let q_yes = q_per_action.i((0, 1));
        let gap = (&q_yes - &q_no).to_kind(Kind::Float); // scalar tensor

        // Tensor-native sigmoid; slightly steeper for 3-lap horizon
        let p2_t = gap.sigmoid();
        let p3_t = (gap * 1.25).sigmoid();

        let p2 = p2_t.double_value(&[]) as f32;
        let p3 = p3_t.double_value(&[]) as f32;

        Ok((p2, p3))
    }
}
