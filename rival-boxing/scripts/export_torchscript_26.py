#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import torch
import importlib.util
from typing import Any, Dict, Tuple

# -- Utilities --------------------------------------------------

def load_mlp_class():
    # Load MLPQRDQN class from scripts/export_torchscript.py
    p = Path("scripts/export_torchscript.py").resolve()
    spec = importlib.util.spec_from_file_location("export_torchscript_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.MLPQRDQN

def load_feat_list(meta_path: Path):
    meta = json.loads(meta_path.read_text())
    feat_list = meta["feat_list"]
    assert isinstance(feat_list, list) and len(feat_list) == 26, \
        f"feat_list must exist and be length 26, got {len(feat_list)}"
    return feat_list, meta

def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Return a state_dict from:
       - full nn.Module (obj.state_dict())
       - dict with 'state_dict' or 'model_state_dict' or nested
       - plain state_dict
    """
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        return obj.state_dict()
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "net", "model"):
            if k in obj and hasattr(obj[k], "state_dict"):
                return obj[k].state_dict()
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    raise RuntimeError(f"Unrecognized checkpoint type: {type(obj)}")

def infer_in_dim_from_sd(sd: Dict[str, torch.Tensor]) -> int:
    # assumes first layer is backbone.0.weight with shape [hidden, in_dim]
    for k in ("backbone.0.weight", "backbone.0.linear.weight", "backbone.0.lin.weight"):
        if k in sd:
            return sd[k].shape[1]
    raise RuntimeError("Could not infer input dim from state_dict; first layer weight not found.")

# -- Wrapper that pads 26 -> orig_in_dim -----------------------

class PadWrap(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, in_dim_runtime: int, in_dim_orig: int):
        super().__init__()
        self.base = base
        self.in_dim_runtime = in_dim_runtime
        self.in_dim_orig = in_dim_orig
        assert in_dim_orig >= in_dim_runtime, "orig_in_dim must be >= runtime in_dim"
        self.pad = in_dim_orig - in_dim_runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 26]  -> cat zeros -> [B, 38]
        if self.pad > 0:
            z = torch.zeros((x.shape[0], self.pad), dtype=x.dtype, device=x.device)
            x = torch.cat([x, z], dim=1)
        return self.base(x)

# -- Main ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/rl/qrdqn.pt",
                    help="Checkpoint (module pickle OR dict OR state_dict)")
    ap.add_argument("--meta", default="artifacts/rl/meta.json",
                    help="Meta file containing feat_list (len=26)")
    ap.add_argument("--out_ts", default="artifacts/rl/qrdqn_torchscript.pt",
                    help="Output TorchScript path to write (accepts 26-dim input)")
    ap.add_argument("--update_meta", action="store_true",
                    help="Write back in_dim=26 and n_quantiles if present")
    ap.add_argument("--n_actions", type=int, default=2)
    ap.add_argument("--n_quantiles", type=int, default=101)  # from your train defaults
    ap.add_argument("--hidden", type=int, default=256)       # from your train defaults
    args = ap.parse_args()

    meta_path = Path(args.meta)
    out_ts = Path(args.out_ts)
    ckpt_path = Path(args.ckpt)

    feat_list, meta = load_feat_list(meta_path)
    in_dim_runtime = len(feat_list)  # 26

    MLPQRDQN = load_mlp_class()

    # Load checkpoint and infer original input dim (likely 38)
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(obj)
    in_dim_orig = infer_in_dim_from_sd(sd)
    print(f"Inferred original input dim from checkpoint: {in_dim_orig}")

    # Build base model with ORIGINAL in_dim so weights load cleanly
    base = MLPQRDQN(
        in_dim=in_dim_orig,
        hidden=args.hidden,
        n_actions=args.n_actions,
        n_quant=args.n_quantiles
    )
    missing, unexpected = base.load_state_dict(sd, strict=False)
    print("load_state_dict -> missing:", missing, "unexpected:", unexpected)

    # Wrap with PadWrap so runtime expects 26 and pads internally to in_dim_orig
    wrapped = PadWrap(base, in_dim_runtime=in_dim_runtime, in_dim_orig=in_dim_orig)
    wrapped.eval()

    with torch.inference_mode():
        dummy = torch.zeros(1, in_dim_runtime, dtype=torch.float32)
        ts = torch.jit.trace(wrapped, dummy)
        ts.save(str(out_ts))
        print(f"Saved TorchScript (runtime in=26, internal in={in_dim_orig}) -> {out_ts}")

    if args.update_meta:
        # Keep runtime contract explicit
        meta["in_dim"] = in_dim_runtime
        meta["n_quantiles"] = args.n_quantiles
        meta.setdefault("n_actions", args.n_actions)
        meta["orig_in_dim"] = in_dim_orig
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Updated {meta_path} with in_dim={in_dim_runtime}, orig_in_dim={in_dim_orig}, n_quantiles={args.n_quantiles}")

if __name__ == "__main__":
    main()
