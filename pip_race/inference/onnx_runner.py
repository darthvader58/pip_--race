from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pip_race.inference.native import NativeScorer


ACCELERATOR_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "metal": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


class OnnxRunner:
    """Thin ONNX Runtime wrapper with a deterministic fallback for dev/test."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        accelerator: str = "cpu",
        native_library_path: str | Path | None = None,
        use_native_fallback: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.session = None
        self.native = NativeScorer.try_load(native_library_path) if use_native_fallback and not self.model_path else None
        self.input_name = "features"
        if self.model_path:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("Install pip-race[ml] to run ONNX inference.") from exc
            selected_providers = select_execution_providers(
                available_providers=ort.get_available_providers(),
                accelerator=accelerator,
                providers=providers,
            )
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=selected_providers,
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

        if self.native is not None:
            return self.native.predict(features)

        row = features[0]
        tire_age_norm = min(1.0, max(0.0, row[3] / 45.0))
        degradation_index = min(1.0, max(0.0, row[14]))
        pit_entry_window = 1.0 - min(1.0, max(0.0, row[15]) / 0.35)
        cheap_stop = row[11]
        score = float(
            -2.0
            + tire_age_norm * 1.4
            + cheap_stop * 0.8
            + degradation_index * 1.0
            + pit_entry_window * 1.2
            + row[2] * 0.35
            - row[1] * 0.2
        )
        pit_risk = 1.0 / (1.0 + pow(2.718281828459045, -score))
        tire_degradation = degradation_index
        confidence = float(max(pit_risk, 1.0 - pit_risk))
        return pit_risk, tire_degradation, confidence


def _sigmoid(x):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))


def select_execution_providers(
    available_providers: Sequence[str],
    accelerator: str = "cpu",
    providers: Sequence[str] | None = None,
) -> list[str]:
    """Resolve ONNX Runtime providers for CPU, CUDA, TensorRT, or Apple CoreML/Metal-class execution."""

    available = list(available_providers)
    if providers is not None:
        selected = list(providers)
    elif accelerator == "auto":
        selected = _auto_providers(available)
    else:
        try:
            selected = ACCELERATOR_PROVIDERS[accelerator]
        except KeyError as exc:
            choices = ", ".join(sorted([*ACCELERATOR_PROVIDERS, "auto"]))
            raise ValueError(f"Unknown accelerator {accelerator!r}. Expected one of: {choices}.") from exc

    missing = [provider for provider in selected if provider not in available]
    if missing:
        raise RuntimeError(
            "Requested ONNX Runtime provider(s) are unavailable: "
            f"{', '.join(missing)}. Available providers: {', '.join(available) or 'none'}."
        )
    return selected


def _auto_providers(available_providers: Sequence[str]) -> list[str]:
    available = list(available_providers)
    priority = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [provider for provider in priority if provider in available]
    if not selected:
        raise RuntimeError("No supported ONNX Runtime execution providers are available.")
    if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
        selected.append("CPUExecutionProvider")
    return selected
