"""Low-latency telemetry inference primitives for pit-wall dashboards."""

from pip_race.contracts import DashboardFrame, HpcTelemetryPacket, InferenceResult
from pip_race.pipeline import PitWallPipeline

__all__ = [
    "DashboardFrame",
    "HpcTelemetryPacket",
    "InferenceResult",
    "PitWallPipeline",
]
