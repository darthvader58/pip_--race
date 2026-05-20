from pip_race.contracts import HpcTelemetryPacket
from pip_race.data import frames_to_jsonl, frames_to_rows, summarize_frames
from pip_race.pipeline import PitWallPipeline
from pip_race.visualization import pit_risk_svg, pit_risk_vega_spec, status_bar_vega_spec


def test_pipeline_emits_dashboard_frame_under_sub_ms_model_latency():
    pipeline = PitWallPipeline()
    packet = HpcTelemetryPacket(
        car_id="ALB",
        lap=42,
        lap_distance_m=2410.0,
        speed_kph=178.0,
        throttle=0.71,
        brake=0.0,
        tire_age_laps=29.0,
        track_temp_c=42.0,
        air_temp_c=27.0,
        compound="MEDIUM",
        track_status="VSC",
    )

    frame = pipeline.process(packet)

    assert frame.telemetry.car_id == "ALB"
    assert 0.0 <= frame.inference.pit_risk <= 1.0
    assert frame.inference.model_latency_ns < 1_000_000
    assert frame.status in {"GREEN", "AMBER", "RED"}


def test_library_data_and_visualization_helpers():
    pipeline = PitWallPipeline()
    packets = [
        HpcTelemetryPacket(car_id="ALB", lap=42, lap_distance_m=2300.0, speed_kph=188.0, tire_age_laps=27.0),
        HpcTelemetryPacket(car_id="ALB", lap=42, lap_distance_m=2410.0, speed_kph=178.0, tire_age_laps=29.0, track_status="VSC"),
    ]

    frames = pipeline.process_many(packets)
    rows = frames_to_rows(frames)
    summary = summarize_frames(frames)
    svg = pit_risk_svg(frames)
    line_spec = pit_risk_vega_spec(frames)
    bar_spec = status_bar_vega_spec(frames)

    assert len(rows) == 2
    assert summary["frames"] == 2
    assert summary["cars"] == ["ALB"]
    assert "pit_risk" in rows[0]
    assert frames_to_jsonl(frames).count("\n") == 1
    assert svg.startswith("<svg")
    assert "Pit risk" in svg
    assert line_spec["mark"]["type"] == "line"
    assert bar_spec["mark"] == "bar"
