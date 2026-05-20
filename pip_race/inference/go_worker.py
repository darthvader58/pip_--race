from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pip_race.contracts import DashboardFrame, HpcTelemetryPacket, InferenceResult


class GoPitWitWorker:
    """Wrapper for the optional Go parallel JSONL scoring worker."""

    def __init__(self, binary_path: str | Path = "pitwit-worker", workers: int | None = None):
        self.binary_path = str(binary_path)
        self.workers = workers

    def process_many(self, packets: list[HpcTelemetryPacket]) -> list[DashboardFrame]:
        args = [self.binary_path]
        if self.workers:
            args.extend(["--workers", str(self.workers)])

        payload = "\n".join(json.dumps(packet.to_dict(), separators=(",", ":")) for packet in packets)
        if payload:
            payload += "\n"

        proc = subprocess.run(args, input=payload, text=True, capture_output=True, check=True)

        frames: list[DashboardFrame] = []
        for line in proc.stdout.splitlines():
            data = json.loads(line)
            telemetry = HpcTelemetryPacket.from_mapping(data["telemetry"])
            inference_data = data["inference"]
            inference = InferenceResult(
                car_id=inference_data["car_id"],
                lap=inference_data["lap"],
                pit_risk=inference_data["pit_risk"],
                tire_degradation=inference_data["tire_degradation"],
                confidence=inference_data["confidence"],
                model_latency_ns=inference_data["model_latency_ns"],
                ts_ns=inference_data["ts_ns"],
            )
            frames.append(DashboardFrame(telemetry=telemetry, inference=inference, status=data["status"]))
        return frames
