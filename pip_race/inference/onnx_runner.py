from __future__ import annotations

from pathlib import Path
from typing import Sequence


class OnnxRunner:
    """Thin ONNX Runtime wrapper with a deterministic fallback for dev/test."""

    def __init__(self, model_path: str | Path | None = None, providers: list[str] | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.session = None
        self.input_name = "features"
        if self.model_path:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install pip-race[ml] to run ONNX inference.") from exc
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers or ["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: Sequence[Sequence[float]]) -> tuple[float, float, float]:
        if self.session is not None:
            import numpy as np

            features_array = np.asarray(features, dtype=np.float32)
            logits = self.session.run(None, {self.input_name: features_array})[0][0]
            probs = _sigmoid(np.asarray(logits, dtype=np.float32))
            pit_risk = float(probs[0])
            tire_degradation = float(probs[1]) if probs.shape[0] > 1 else pit_risk
            confidence = float(max(pit_risk, 1.0 - pit_risk))
            return pit_risk, tire_degradation, confidence

        row = features[0]
        score = float(row[3] * 0.08 + row[11] * 0.65 + row[14] * 0.75)
        pit_risk = 1.0 / (1.0 + pow(2.718281828459045, -score))
        tire_degradation = float(min(1.0, row[14]))
        confidence = float(max(pit_risk, 1.0 - pit_risk))
        return pit_risk, tire_degradation, confidence


def _sigmoid(x):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))
