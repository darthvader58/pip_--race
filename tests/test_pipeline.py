from pip_race.contracts import HpcTelemetryPacket
from pip_race.pipeline import PitWallPipeline


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
