from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time_ns
from typing import Any


def now_ns() -> int:
    return time_ns()


@dataclass(slots=True)
class HpcTelemetryPacket:
    """Normalized high-rate telemetry frame from an HPC/trackside stream."""

    car_id: str
    lap: int
    lap_distance_m: float
    speed_kph: float
    throttle: float = 0.0
    brake: float = 0.0
    tire_age_laps: float = 0.0
    track_temp_c: float = 0.0
    air_temp_c: float = 0.0
    compound: str = "UNKNOWN"
    track_status: str = "GREEN"
    ts_ns: int = field(default_factory=now_ns)
    extras: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "HpcTelemetryPacket":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known and k != "extras"}
        extras = data.get("extras") or {
            k: float(v) for k, v in data.items() if k not in known and isinstance(v, int | float)
        }
        return cls(**kwargs, extras=extras)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceResult:
    car_id: str
    lap: int
    pit_risk: float
    tire_degradation: float
    confidence: float
    model_latency_ns: int
    ts_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DashboardFrame:
    """Compact payload for a pit-crew UI."""

    telemetry: HpcTelemetryPacket
    inference: InferenceResult
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "telemetry": self.telemetry.to_dict(),
            "inference": self.inference.to_dict(),
            "status": self.status,
        }
