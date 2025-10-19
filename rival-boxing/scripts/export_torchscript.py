#!/usr/bin/env python3
"""
Export trained QR-DQN to TorchScript for Rust (tch) runtime.

Fixes:
- Properly loads ckpt["state_dict"] instead of the entire dict.
- Derives in_dim/n_actions/n_quantiles/hidden from ckpt (with safe fallbacks).
- Ensures meta.json exists/updated with feat_list & shapes for Rust.

Run:
  python scripts/export_torchscript.py
Outputs:
  artifacts/rl/qrdqn_torchscript.pt
  artifacts/rl/meta.json   (updated/created if needed)
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

ART = Path("artifacts/rl")
CKPT = ART / "qrdqn.pt"
OUT_TS = ART / "qrdqn_torchscript.pt"
META_JSON = ART / "meta.json"

class MLPQRDQN(nn.Module):
    def __init__(self, in_dim:int, hidden:int=128, n_actions:int=2, n_quant:int=51):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.head = nn.Linear(hidden, n_actions * n_quant)
        self.n_actions = n_actions
        self.n_quant = n_quant

    def forward(self, x):
        z = self.backbone(x)
        qz = self.head(z).view(-1, self.n_actions, self.n_quant)  # (B,A,Q)
        return qz

def main():
    if not CKPT.exists():
        print(f"ERROR: checkpoint not found: {CKPT}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(CKPT, map_location="cpu")

    # pull state_dict (some trainers save raw SD; others wrap it)
    state_dict = ckpt.get("state_dict", ckpt)

    # shapes & hyperparams (prefer ckpt)
    n_actions = int(ckpt.get("n_actions", 2))
    n_quant   = int(ckpt.get("n_quantiles", ckpt.get("n_quant", 51)))
    hidden    = int(ckpt.get("hidden", 128))

    # feature list & in_dim
    feat_list = ckpt.get("feat_list", None)
    if feat_list is None and META_JSON.exists():
        try:
            meta_disk = json.loads(META_JSON.read_text())
            feat_list = meta_disk.get("feat_list", None)
            if feat_list is None and "features" in meta_disk:
                feat_list = meta_disk["features"]
        except Exception:
            pass
    if feat_list is None:
        print("ERROR: feat_list not found in ckpt or meta.json; cannot determine input dim.", file=sys.stderr)
        sys.exit(2)
    in_dim = int(ckpt.get("in_dim", len(feat_list)))

    # build & load model
    net = MLPQRDQN(in_dim=in_dim, hidden=hidden, n_actions=n_actions, n_quant=n_quant)
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"WARNING: unexpected keys when loading state_dict: {unexpected}")
    net.eval()

    # trace
    example = torch.zeros(1, in_dim)
    with torch.inference_mode():
        ts = torch.jit.trace(net, example)
    OUT_TS.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(OUT_TS))
    print(f"Saved TorchScript: {OUT_TS}")

    # (Re)write meta.json to keep Rust runtime aligned
    meta_out = {
        "feat_list": feat_list,
        "in_dim": in_dim,
        "n_actions": n_actions,
        "n_quant": n_quant,
        "hidden": hidden
    }
    META_JSON.write_text(json.dumps(meta_out, indent=2))
    print(f"Wrote/updated meta: {META_JSON}")

if __name__ == "__main__":
    main()
