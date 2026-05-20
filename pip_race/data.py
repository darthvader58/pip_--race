from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pip_race.contracts import DashboardFrame, HpcTelemetryPacket


def packet_from_dict(data: dict[str, Any]) -> HpcTelemetryPacket:
    """Create a normalized telemetry packet from a mapping."""

    return HpcTelemetryPacket.from_mapping(data)


def packets_from_jsonl(path: str | Path) -> list[HpcTelemetryPacket]:
    """Load telemetry packets from newline-delimited JSON."""

    packets: list[HpcTelemetryPacket] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                packets.append(packet_from_dict(json.loads(line)))
    return packets


def frames_to_rows(frames: Iterable[DashboardFrame]) -> list[dict[str, Any]]:
    """Flatten dashboard frames into table-shaped rows for CSV, notebooks, or BI tools."""

    rows: list[dict[str, Any]] = []
    for frame in frames:
        telemetry = frame.telemetry
        inference = frame.inference
        rows.append(
            {
                "car_id": telemetry.car_id,
                "lap": telemetry.lap,
                "lap_distance_m": telemetry.lap_distance_m,
                "speed_kph": telemetry.speed_kph,
                "throttle": telemetry.throttle,
                "brake": telemetry.brake,
                "tire_age_laps": telemetry.tire_age_laps,
                "compound": telemetry.compound,
                "track_status": telemetry.track_status,
                "pit_risk": inference.pit_risk,
                "tire_degradation": inference.tire_degradation,
                "confidence": inference.confidence,
                "model_latency_ns": inference.model_latency_ns,
                "status": frame.status,
                "ts_ns": inference.ts_ns,
            }
        )
    return rows


def summarize_frames(frames: Iterable[DashboardFrame]) -> dict[str, Any]:
    """Return high-signal operational summary metrics for a processed stint/race."""

    frame_list = list(frames)
    if not frame_list:
        return {
            "frames": 0,
            "cars": [],
            "max_pit_risk": 0.0,
            "mean_pit_risk": 0.0,
            "mean_model_latency_ns": 0.0,
            "status_counts": {},
        }

    risks = [frame.inference.pit_risk for frame in frame_list]
    latencies = [frame.inference.model_latency_ns for frame in frame_list]
    return {
        "frames": len(frame_list),
        "cars": sorted({frame.telemetry.car_id for frame in frame_list}),
        "max_pit_risk": max(risks),
        "mean_pit_risk": sum(risks) / len(risks),
        "mean_model_latency_ns": sum(latencies) / len(latencies),
        "status_counts": dict(Counter(frame.status for frame in frame_list)),
    }


def write_frames_csv(frames: Iterable[DashboardFrame], path: str | Path) -> None:
    """Write flattened inference frames as CSV."""

    rows = frames_to_rows(frames)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return

    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def frames_to_jsonl(frames: Iterable[DashboardFrame]) -> str:
    """Serialize dashboard frames as newline-delimited JSON."""

    return "\n".join(json.dumps(frame.to_dict(), separators=(",", ":")) for frame in frames)
