from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pip_race import HpcTelemetryPacket, PitWit
from pip_race.data import frames_to_rows, summarize_frames
from pip_race.visualization import pit_risk_svg
from pip_race.inference import OnnxRunner


app = Flask(
    __name__,
    static_folder=str(ROOT / "demo_static"),
    static_url_path="/static",
    template_folder=str(ROOT / "demo_templates"),
)


DEFAULT_PACKET: dict[str, Any] = {
    "car_id": "ALB",
    "lap": 42,
    "lap_distance_m": 2410.0,
    "speed_kph": 178.0,
    "throttle": 0.71,
    "brake": 0.0,
    "tire_age_laps": 29.0,
    "track_temp_c": 42.0,
    "air_temp_c": 27.0,
    "compound": "MEDIUM",
    "track_status": "VSC",
}


@app.get("/")
def index():
    return render_template("index.html", defaults=DEFAULT_PACKET)


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "pip-race-demo"})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        packet = _packet_from_payload(payload)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    packets = _demo_stint(packet)
    frames = _pitwit().process_many(packets)
    rows = frames_to_rows(frames)
    current = frames[-1]

    return jsonify(
        {
            "frame": current.to_dict(),
            "rows": rows,
            "summary": summarize_frames(frames),
            "svg": pit_risk_svg(frames, width=760, height=280),
            "model_source": "demo_model/pitwit_demo.onnx",
        }
    )


@lru_cache(maxsize=1)
def _pitwit() -> PitWit:
    model_path = ROOT / "demo_model" / "pitwit_demo.onnx"
    return PitWit(runner=OnnxRunner(model_path, use_native_fallback=False))


def _packet_from_payload(payload: dict[str, Any]) -> HpcTelemetryPacket:
    data = {**DEFAULT_PACKET, **payload}
    car_id = str(data["car_id"]).strip().upper()
    if not car_id:
        raise ValueError("car_id is required")

    compound = str(data["compound"]).strip().upper()
    track_status = str(data["track_status"]).strip().upper()

    return HpcTelemetryPacket(
        car_id=car_id[:8],
        lap=_int_value(data, "lap", minimum=1, maximum=200),
        lap_distance_m=_float_value(data, "lap_distance_m", minimum=0.0, maximum=8000.0),
        speed_kph=_float_value(data, "speed_kph", minimum=0.0, maximum=380.0),
        throttle=_float_value(data, "throttle", minimum=0.0, maximum=1.0),
        brake=_float_value(data, "brake", minimum=0.0, maximum=1.0),
        tire_age_laps=_float_value(data, "tire_age_laps", minimum=0.0, maximum=90.0),
        track_temp_c=_float_value(data, "track_temp_c", minimum=-10.0, maximum=80.0),
        air_temp_c=_float_value(data, "air_temp_c", minimum=-10.0, maximum=55.0),
        compound=compound or "UNKNOWN",
        track_status=track_status or "GREEN",
    )


def _demo_stint(packet: HpcTelemetryPacket) -> list[HpcTelemetryPacket]:
    samples: list[HpcTelemetryPacket] = []
    for idx, offset in enumerate((-900.0, -600.0, -300.0, 0.0)):
        distance = max(0.0, packet.lap_distance_m + offset)
        tire_age = max(0.0, packet.tire_age_laps - (3 - idx) * 0.6)
        speed = max(40.0, packet.speed_kph + (3 - idx) * 4.5)
        samples.append(
            HpcTelemetryPacket(
                car_id=packet.car_id,
                lap=packet.lap,
                lap_distance_m=distance,
                speed_kph=speed,
                throttle=packet.throttle,
                brake=packet.brake,
                tire_age_laps=tire_age,
                track_temp_c=packet.track_temp_c,
                air_temp_c=packet.air_temp_c,
                compound=packet.compound,
                track_status=packet.track_status,
            )
        )
    return samples


def _float_value(data: dict[str, Any], key: str, minimum: float, maximum: float) -> float:
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")
    return value


def _int_value(data: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = int(_float_value(data, key, float(minimum), float(maximum)))
    return value
