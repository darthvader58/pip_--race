from __future__ import annotations

import json
from dataclasses import dataclass

from pip_race.contracts import DashboardFrame


@dataclass
class RedisDashboardPublisher:
    url: str = "redis://localhost:6379/0"
    channel: str = "pitcrew:dashboard"
    stream: str = "pitcrew:frames"
    maxlen: int = 512

    def __post_init__(self) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install redis to publish dashboard frames.") from exc
        self.client = redis.Redis.from_url(self.url, decode_responses=True)

    def publish(self, frame: DashboardFrame) -> None:
        payload = json.dumps(frame.to_dict(), separators=(",", ":"))
        self.client.publish(self.channel, payload)
        self.client.xadd(self.stream, {"frame": payload}, maxlen=self.maxlen, approximate=True)
