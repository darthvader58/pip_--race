from __future__ import annotations

import json
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from time import perf_counter_ns

from pip_race.contracts import HpcTelemetryPacket
from pip_race.pipeline import PitWallPipeline


@dataclass(slots=True)
class BenchmarkResult:
    iterations: int
    warmup: int
    min_ns: int
    max_ns: int
    mean_ns: float
    p50_ns: int
    p95_ns: int
    p99_ns: int
    python_version: str
    platform: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def run_pipeline_benchmark(
    pipeline: PitWallPipeline | None = None,
    packet: HpcTelemetryPacket | None = None,
    iterations: int = 10_000,
    warmup: int = 1_000,
) -> BenchmarkResult:
    """Measure end-to-end single-packet pipeline latency in nanoseconds."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    pipeline = pipeline or PitWallPipeline()
    packet = packet or HpcTelemetryPacket(
        car_id="BENCH",
        lap=1,
        lap_distance_m=2410.0,
        speed_kph=178.0,
        tire_age_laps=29.0,
        compound="MEDIUM",
        track_status="VSC",
    )

    for _ in range(warmup):
        pipeline.process(packet)

    samples: list[int] = []
    for _ in range(iterations):
        start = perf_counter_ns()
        pipeline.process(packet)
        samples.append(perf_counter_ns() - start)

    ordered = sorted(samples)
    return BenchmarkResult(
        iterations=iterations,
        warmup=warmup,
        min_ns=ordered[0],
        max_ns=ordered[-1],
        mean_ns=statistics.fmean(ordered),
        p50_ns=_percentile(ordered, 50),
        p95_ns=_percentile(ordered, 95),
        p99_ns=_percentile(ordered, 99),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def _percentile(ordered_samples: list[int], percentile: int) -> int:
    if not ordered_samples:
        raise ValueError("ordered_samples must not be empty")
    idx = round((percentile / 100) * (len(ordered_samples) - 1))
    return ordered_samples[idx]
