from __future__ import annotations


def build_model(input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 2):
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Install pip-race[ml] to build or train the PyTorch model.") from exc

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def export_onnx(output_path: str, input_dim: int = 16) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install pip-race[ml] to export ONNX models.") from exc

    model = build_model(input_dim=input_dim)
    model.eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
