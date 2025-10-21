#!/usr/bin/env python3
"""
Verification script to ensure training, export, and inference dimensions align.
Run this after training and before deploying to catch dimension mismatches.
"""
import json
import argparse
from pathlib import Path
import torch
import sys

def check_meta(meta_path: Path):
    """Check meta.json structure"""
    print(f"\n📋 Checking {meta_path}...")
    
    if not meta_path.exists():
        print(f"  ❌ File not found!")
        return None
    
    meta = json.loads(meta_path.read_text())
    
    feat_list = meta.get("feat_list", [])
    in_dim = meta.get("in_dim")
    n_quantiles = meta.get("n_quantiles")
    n_actions = meta.get("n_actions", 2)
    
    print(f"  ✓ feat_list length: {len(feat_list)}")
    print(f"  ✓ in_dim: {in_dim}")
    print(f"  ✓ n_quantiles: {n_quantiles}")
    print(f"  ✓ n_actions: {n_actions}")
    
    # Check for duplicates
    if len(feat_list) != len(set(feat_list)):
        print(f"  ❌ WARNING: Duplicate features detected!")
        dupes = [f for f in feat_list if feat_list.count(f) > 1]
        print(f"      Duplicates: {set(dupes)}")
        return None
    else:
        print(f"  ✓ No duplicate features")
    
    # Check alignment
    if in_dim and in_dim != len(feat_list):
        print(f"  ❌ WARNING: in_dim ({in_dim}) != feat_list length ({len(feat_list)})")
        return None
    
    # Show first few features
    print(f"  Features (first 5): {feat_list[:5]}")
    
    return meta

def check_checkpoint(ckpt_path: Path):
    """Check checkpoint structure"""
    print(f"\n💾 Checking {ckpt_path}...")
    
    if not ckpt_path.exists():
        print(f"  ❌ File not found!")
        return None
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        metadata = ckpt
    else:
        state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else {}
        metadata = {}
    
    # Find first layer to infer input dim
    first_layer_key = None
    for k in ["backbone.0.weight", "backbone.0.linear.weight", "backbone.0.lin.weight"]:
        if k in state_dict:
            first_layer_key = k
            break
    
    if first_layer_key:
        hidden, in_dim = state_dict[first_layer_key].shape
        print(f"  ✓ Inferred from weights: in_dim={in_dim}, hidden={hidden}")
    else:
        print(f"  ⚠️  Could not find first layer in state_dict")
        in_dim = metadata.get("in_dim")
        hidden = metadata.get("hidden")
    
    # Check metadata
    meta_in_dim = metadata.get("in_dim")
    meta_feat_list = metadata.get("feat_list", [])
    
    print(f"  ✓ Metadata in_dim: {meta_in_dim}")
    print(f"  ✓ Metadata feat_list length: {len(meta_feat_list)}")
    
    if meta_in_dim and in_dim and meta_in_dim != in_dim:
        print(f"  ❌ WARNING: Metadata in_dim ({meta_in_dim}) != weight in_dim ({in_dim})")
        return None
    
    return {"in_dim": in_dim or meta_in_dim, "feat_list": meta_feat_list}

def check_torchscript(ts_path: Path, expected_in_dim: int):
    """Check TorchScript model"""
    print(f"\n🔥 Checking {ts_path}...")
    
    if not ts_path.exists():
        print(f"  ❌ File not found!")
        return None
    
    try:
        model = torch.jit.load(str(ts_path), map_location="cpu")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None
    
    # Test with dummy input
    try:
        dummy = torch.zeros(1, expected_in_dim, dtype=torch.float32)
        output = model(dummy)
        
        print(f"  ✓ Input shape: {tuple(dummy.shape)}")
        print(f"  ✓ Output shape: {tuple(output.shape)}")
        
        if len(output.shape) == 3:
            batch, n_actions, n_quantiles = output.shape
            print(f"  ✓ Parsed: batch={batch}, n_actions={n_actions}, n_quantiles={n_quantiles}")
        else:
            print(f"  ⚠️  Unexpected output shape: {output.shape}")
            return None
        
        # Test with non-zero input
        dummy_nonzero = torch.randn(1, expected_in_dim, dtype=torch.float32) * 0.5
        output_nonzero = model(dummy_nonzero)
        
        # Check if outputs are different (model is working)
        diff = torch.abs(output - output_nonzero).max().item()
        print(f"  ✓ Model responds to input (max diff: {diff:.6f})")
        
        if diff < 1e-6:
            print(f"  ⚠️  WARNING: Model output seems constant!")
            return None
        
        return True
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Verify model pipeline dimensions")
    parser.add_argument("--meta", default="artifacts/rl/meta.json", help="Path to meta.json")
    parser.add_argument("--ckpt", default="artifacts/rl/qrdqn.pt", help="Path to checkpoint")
    parser.add_argument("--ts", default="artifacts/rl/qrdqn_torchscript.pt", help="Path to TorchScript")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 PIPELINE DIMENSION VERIFICATION")
    print("=" * 70)
    
    meta_path = Path(args.meta)
    ckpt_path = Path(args.ckpt)
    ts_path = Path(args.ts)
    
    # Check meta.json
    meta_info = check_meta(meta_path)
    if not meta_info:
        print("\n❌ meta.json check failed!")
        sys.exit(1)
    
    expected_in_dim = len(meta_info["feat_list"])
    
    # Check checkpoint
    ckpt_info = check_checkpoint(ckpt_path)
    if not ckpt_info:
        print("\n❌ checkpoint check failed!")
        sys.exit(1)
    
    # Verify checkpoint matches meta
    if ckpt_info["in_dim"] != expected_in_dim:
        print(f"\n❌ DIMENSION MISMATCH!")
        print(f"   meta.json expects: {expected_in_dim}")
        print(f"   checkpoint has: {ckpt_info['in_dim']}")
        sys.exit(1)
    
    # Check TorchScript
    ts_ok = check_torchscript(ts_path, expected_in_dim)
    if not ts_ok:
        print("\n❌ TorchScript check failed!")
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 70)
    print(f"✓ Consistent input dimension: {expected_in_dim}")
    print(f"✓ Feature list length: {expected_in_dim}")
    print(f"✓ No duplicate features")
    print(f"✓ Checkpoint weights match")
    print(f"✓ TorchScript model works")
    print("\n🎉 Pipeline is ready for inference!")
    
    # Show feature list
    print(f"\n📋 Feature List ({expected_in_dim} features):")
    for i, feat in enumerate(meta_info["feat_list"]):
        print(f"   [{i:2d}] {feat}")

if __name__ == "__main__":
    main()