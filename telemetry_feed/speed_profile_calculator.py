from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class SpeedSample:
    x_m: float
    v_mps: float

    def to_dict(self) -> dict[str, float]:
        return {"x_m": self.x_m, "v_mps": self.v_mps}


class SpeedProfileCalculator:
    """Maintains recent telemetry and emits distance-sorted speed profiles."""

    def __init__(self, window_size: int = 50, lookahead_m: float = 500.0):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size
        self.lookahead_m = lookahead_m
        self._window: deque[SpeedSample] = deque(maxlen=window_size)

    def add_sample(self, lap_distance_m: float, speed_kph: float) -> None:
        self._window.append(SpeedSample(float(lap_distance_m), float(speed_kph) / 3.6))

    def get_profile(
        self,
        current_distance_m: float,
        target_distance_m: float,
        buffer_m: float = 50.0,
    ) -> list[dict[str, float]] | None:
        if len(self._window) < 2:
            return None

        min_x = current_distance_m - buffer_m
        max_x = target_distance_m + buffer_m
        samples = [s for s in self._window if min_x <= s.x_m <= max_x]
        if len(samples) < 2:
            return None

        samples.sort(key=lambda sample: sample.x_m)
        return [sample.to_dict() for sample in samples]

    def get_lookahead_profile(self, current_distance_m: float) -> list[dict[str, float]] | None:
        return self.get_profile(current_distance_m, current_distance_m + self.lookahead_m)

    def reset(self) -> None:
        self._window.clear()

    @property
    def window_len(self) -> int:
        return len(self._window)
