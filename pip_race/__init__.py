"""Low-latency telemetry inference primitives for pit-wall dashboards."""

from pip_race.contracts import DashboardFrame, HpcTelemetryPacket, InferenceResult
from pip_race.data import frames_to_rows, summarize_frames
from pip_race.pipeline import PitWallPipeline

__all__ = [
    "DashboardFrame",
    "HpcTelemetryPacket",
    "InferenceResult",
    "PitWallPipeline",
    "frames_to_rows",
    "summarize_frames",
]
