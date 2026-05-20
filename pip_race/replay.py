from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

from pip_race.contracts import HpcTelemetryPacket
from pip_race.pipeline import PitWallPipeline
from pip_race.streaming.redis_bus import RedisDashboardPublisher


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay telemetry JSONL through the pit-wall pipeline.")
    parser.add_argument("--redis-url", default=None)
    args = parser.parse_args()

    pipeline = PitWallPipeline()
    publisher = RedisDashboardPublisher(args.redis_url) if args.redis_url else None
    for packet in iter_jsonl(sys.stdin):
        frame = pipeline.process(packet)
        if publisher:
            publisher.publish(frame)
        print(json.dumps(frame.to_dict(), separators=(",", ":")), flush=True)


def iter_jsonl(lines: Iterable[str]) -> Iterable[HpcTelemetryPacket]:
    for line in lines:
        line = line.strip()
        if line:
            yield HpcTelemetryPacket.from_mapping(json.loads(line))


if __name__ == "__main__":
    main()
