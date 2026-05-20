from pip_race.contracts import HpcTelemetryPacket
from pip_race.data import frames_to_jsonl, frames_to_rows, summarize_frames
from pip_race.pitwit import PitWit
from pip_race.visualization import pit_risk_svg, pit_risk_vega_spec, status_bar_vega_spec
from pip_race.cli import main as cli_main
from pip_race.benchmark import run_pitwit_benchmark


def test_pitwit_emits_dashboard_frame_under_sub_ms_model_latency():
    pitwit = PitWit()
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

    frame = pitwit.process(packet)

    assert frame.telemetry.car_id == "ALB"
    assert 0.0 <= frame.inference.pit_risk <= 1.0
    assert frame.inference.model_latency_ns < 1_000_000
    assert frame.status in {"GREEN", "AMBER", "RED"}


def test_library_data_and_visualization_helpers():
    pitwit = PitWit()
    packets = [
        HpcTelemetryPacket(car_id="ALB", lap=42, lap_distance_m=2300.0, speed_kph=188.0, tire_age_laps=27.0),
        HpcTelemetryPacket(car_id="ALB", lap=42, lap_distance_m=2410.0, speed_kph=178.0, tire_age_laps=29.0, track_status="VSC"),
    ]

    frames = pitwit.process_many(packets)
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


def test_cli_writes_jsonl_summary_and_svg(tmp_path):
    input_path = tmp_path / "telemetry.jsonl"
    output_path = tmp_path / "frames.jsonl"
    summary_path = tmp_path / "summary.json"
    svg_path = tmp_path / "pit_risk.svg"
    input_path.write_text(
        '{"car_id":"ALB","lap":42,"lap_distance_m":2410,"speed_kph":178,"tire_age_laps":29,"compound":"MEDIUM","track_status":"VSC"}\n',
        encoding="utf-8",
    )

    result = cli_main(
        [
            "infer",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--svg",
            str(svg_path),
        ]
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8").count("\n") == 1
    assert '"frames": 1' in summary_path.read_text(encoding="utf-8")
    assert svg_path.read_text(encoding="utf-8").startswith("<svg")


def test_benchmark_returns_latency_percentiles():
    result = run_pitwit_benchmark(iterations=5, warmup=1)

    assert result.iterations == 5
    assert result.p50_ns > 0
    assert result.p95_ns >= result.p50_ns
    assert result.p99_ns >= result.p95_ns
