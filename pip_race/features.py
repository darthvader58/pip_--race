from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from pip_race.contracts import HpcTelemetryPacket


FEATURE_NAMES = [
    "speed_mps",
    "throttle",
    "brake",
    "tire_age_laps",
    "track_temp_c",
    "air_temp_c",
    "is_soft",
    "is_medium",
    "is_hard",
    "is_intermediate",
    "is_wet",
    "is_cheap_stop",
    "speed_delta_3",
    "speed_var_5",
    "degradation_index",
    "distance_to_pit_entry_norm",
]


@dataclass
class FeatureExtractor:
    """Stateful feature extraction tuned for tiny online batches."""

    pit_entry_m: float = 2700.0
    track_length_m: float = 3337.0
    history_size: int = 8
    _speed_history: dict[str, deque[float]] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def transform_one(self, packet: HpcTelemetryPacket) -> list[list[float]]:
        speeds = self._speed_history.setdefault(packet.car_id, deque(maxlen=self.history_size))
        speed_mps = packet.speed_kph / 3.6
        speeds.append(speed_mps)

        compound = packet.compound.upper()
        track_status = packet.track_status.upper()
        speed_delta_3 = speed_mps - speeds[-3] if len(speeds) >= 3 else 0.0
        recent = list(speeds)[-5:]
        if len(recent) >= 2:
            mean = sum(recent) / len(recent)
            speed_var_5 = sum((value - mean) ** 2 for value in recent) / len(recent)
        else:
            speed_var_5 = 0.0
        degradation_index = max(0.0, packet.tire_age_laps / 35.0) + max(0.0, -speed_delta_3 / 20.0)
        distance_to_pit = (self.pit_entry_m - packet.lap_distance_m) % self.track_length_m

        features = [
            [
                speed_mps,
                packet.throttle,
                packet.brake,
                packet.tire_age_laps,
                packet.track_temp_c,
                packet.air_temp_c,
                float(compound == "SOFT"),
                float(compound == "MEDIUM"),
                float(compound == "HARD"),
                float(compound == "INTERMEDIATE"),
                float(compound == "WET"),
                float(track_status in {"YELLOW", "VSC", "SC", "SAFETY_CAR"}),
                speed_delta_3,
                speed_var_5,
                degradation_index,
                distance_to_pit / self.track_length_m,
            ],
        ]
        return features
