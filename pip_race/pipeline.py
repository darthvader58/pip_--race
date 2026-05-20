from __future__ import annotations

from time import perf_counter_ns
from collections.abc import Iterable

from pip_race.contracts import DashboardFrame, HpcTelemetryPacket, InferenceResult
from pip_race.features import FeatureExtractor
from pip_race.inference.onnx_runner import OnnxRunner


class PitWallPipeline:
    """Feature extraction plus ONNX inference for single-frame latency."""

    def __init__(
        self,
        runner: OnnxRunner | None = None,
        extractor: FeatureExtractor | None = None,
        red_threshold: float = 0.75,
        amber_threshold: float = 0.45,
    ):
        self.runner = runner or OnnxRunner()
        self.extractor = extractor or FeatureExtractor()
        self.red_threshold = red_threshold
        self.amber_threshold = amber_threshold

    def process(self, packet: HpcTelemetryPacket) -> DashboardFrame:
        features = self.extractor.transform_one(packet)
        start = perf_counter_ns()
        pit_risk, tire_degradation, confidence = self.runner.predict(features)
        latency_ns = perf_counter_ns() - start
        result = InferenceResult(
            car_id=packet.car_id,
            lap=packet.lap,
            pit_risk=pit_risk,
            tire_degradation=tire_degradation,
            confidence=confidence,
            model_latency_ns=latency_ns,
            ts_ns=packet.ts_ns,
        )
        return DashboardFrame(telemetry=packet, inference=result, status=self._status(pit_risk))

    def process_many(self, packets: Iterable[HpcTelemetryPacket]) -> list[DashboardFrame]:
        """Process an iterable of telemetry packets and return dashboard-ready frames."""

        return [self.process(packet) for packet in packets]

    def _status(self, pit_risk: float) -> str:
        if pit_risk >= self.red_threshold:
            return "RED"
        if pit_risk >= self.amber_threshold:
            return "AMBER"
        return "GREEN"
