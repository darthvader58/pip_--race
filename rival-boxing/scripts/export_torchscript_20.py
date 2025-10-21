#!/usr/bin/env python3
"""
Export QR-DQN checkpoint to TorchScript for real-time inference.
SIMPLIFIED: No padding needed - model and features both use 20 dimensions.
"""
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# ----------------------------- Model Definition -----------------------------

class QRDQN(nn.Module):
    """Same architecture as training"""
    def __init__(self, in_dim: int, n_actions: int = 2, n_quantiles: int = 51, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions * n_quantiles)

    def forward(self, x):
        z = self.backbone(x)
        q = self.head(z)
        return q.view(-1, self.n_actions, self.n_quantiles)

# ----------------------------- Helper Functions -----------------------------

def load_checkpoint(ckpt_path: Path):
    """Load checkpoint and extract state_dict"""
    obj = torch.load(ckpt_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"], obj
        elif "model_state_dict" in obj:
            return obj["model_state_dict"], obj
        else:
            # Assume it's already a state_dict
            return obj, {}
    else:
        # It's a model object
        return obj.state_dict(), {}

def infer_dimensions(state_dict: dict) -> tuple:
    """Infer model dimensions from state_dict"""
    # First layer: backbone.0.weight shape is [hidden, in_dim]
    first_layer_key = None
    for k in ["backbone.0.weight", "backbone.0.linear.weight", "backbone.0.lin.weight"]:
        if k in state_dict:
            first_layer_key = k
            break
    
    if first_layer_key is None:
        raise RuntimeError("Could not find first layer in state_dict")
    
    hidden, in_dim = state_dict[first_layer_key].shape
    
    # Last layer: head.weight shape is [n_actions * n_quantiles, hidden]
    head_key = None
    for k in ["head.weight", "head.linear.weight"]:
        if k in state_dict:
            head_key = k
            break
    
    if head_key is None:
        raise RuntimeError("Could not find head layer in state_dict")
    
    out_features = state_dict[head_key].shape[0]
    
    return int(in_dim), int(hidden), int(out_features)

# ----------------------------- Main Export Function -----------------------------

def export_torchscript(
    ckpt_path: str,
    meta_path: str,
    out_path: str,
    n_actions: int = 2,
    n_quantiles: int = 101,
    update_meta: bool = True
):
    """Export model to TorchScript"""
    
    ckpt_path = Path(ckpt_path)
    meta_path = Path(meta_path)
    out_path = Path(out_path)
    
    # Load meta to get feature list
    meta = json.loads(meta_path.read_text())
    feat_list = meta.get("feat_list", [])
    
    if not feat_list:
        raise ValueError("feat_list not found in meta.json")
    
    in_dim_expected = len(feat_list)
    print(f"Feature list has {in_dim_expected} features")
    
    # Load checkpoint
    state_dict, ckpt_meta = load_checkpoint(ckpt_path)
    
    # Infer dimensions from checkpoint
    in_dim_ckpt, hidden, out_features = infer_dimensions(state_dict)
    print(f"Checkpoint expects in_dim={in_dim_ckpt}, hidden={hidden}, out_features={out_features}")
    
    # Verify dimensions match
    if in_dim_ckpt != in_dim_expected:
        raise ValueError(
            f"Dimension mismatch! Checkpoint expects {in_dim_ckpt} features, "
            f"but meta.json has {in_dim_expected}. Did you retrain the model?"
        )
    
    # Infer n_quantiles from output
    if out_features % n_actions != 0:
        raise ValueError(f"out_features {out_features} not divisible by n_actions {n_actions}")
    
    n_quantiles_inferred = out_features // n_actions
    print(f"Inferred n_quantiles={n_quantiles_inferred} from checkpoint")
    
    # Build model
    model = QRDQN(
        in_dim=in_dim_ckpt,
        n_actions=n_actions,
        n_quantiles=n_quantiles_inferred,
        hidden=hidden
    )
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
    
    model.eval()
    
    # Export to TorchScript
    with torch.inference_mode():
        dummy_input = torch.zeros(1, in_dim_ckpt, dtype=torch.float32)
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Verify output shape
        test_output = traced_model(dummy_input)
        expected_shape = (1, n_actions, n_quantiles_inferred)
        if tuple(test_output.shape) != expected_shape:
            raise RuntimeError(
                f"Output shape mismatch! Got {tuple(test_output.shape)}, "
                f"expected {expected_shape}"
            )
        
        # Save
        traced_model.save(str(out_path))
        print(f"✓ Saved TorchScript model to {out_path}")
        print(f"  Input: [batch, {in_dim_ckpt}]")
        print(f"  Output: [batch, {n_actions}, {n_quantiles_inferred}]")
    
    # Update meta.json if requested
    if update_meta:
        meta["in_dim"] = in_dim_ckpt
        meta["n_quantiles"] = n_quantiles_inferred
        meta["n_actions"] = n_actions
        meta["hidden"] = hidden
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"✓ Updated {meta_path}")

# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export QR-DQN to TorchScript (no padding needed)"
    )
    parser.add_argument(
        "--ckpt",
        default="artifacts/rl/qrdqn.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--meta",
        default="artifacts/rl/meta.json",
        help="Path to meta.json with feat_list"
    )
    parser.add_argument(
        "--out",
        default="artifacts/rl/qrdqn_torchscript.pt",
        help="Output TorchScript path"
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=2,
        help="Number of actions (default: 2)"
    )
    parser.add_argument(
        "--n_quantiles",
        type=int,
        default=101,
        help="Number of quantiles (default: 101)"
    )
    parser.add_argument(
        "--update_meta",
        action="store_true",
        help="Update meta.json with inferred dimensions"
    )
    
    args = parser.parse_args()
    
    export_torchscript(
        ckpt_path=args.ckpt,
        meta_path=args.meta,
        out_path=args.out,
        n_actions=args.n_actions,
        n_quantiles=args.n_quantiles,
        update_meta=args.update_meta
    )

if __name__ == "__main__":
    main()