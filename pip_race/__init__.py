"""Low-latency telemetry inference primitives for pitwit dashboards."""

from pip_race.benchmark import BenchmarkResult, run_pitwit_benchmark
from pip_race.contracts import DashboardFrame, HpcTelemetryPacket, InferenceResult
from pip_race.data import frames_to_rows, summarize_frames
from pip_race.pitwit import PitWit

__all__ = [
    "DashboardFrame",
    "BenchmarkResult",
    "HpcTelemetryPacket",
    "InferenceResult",
    "PitWit",
    "frames_to_rows",
    "run_pitwit_benchmark",
    "summarize_frames",
]
