# API Reference

This document covers the public library surface intended for users. Internal modules may change faster than the APIs below.

## Contracts

### `HpcTelemetryPacket`

Normalized input packet for a telemetry frame.

```python
from pip_race import HpcTelemetryPacket

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
```

Required fields:

- `car_id`
- `lap`
- `lap_distance_m`
- `speed_kph`

Optional fields default to neutral values. Unknown numeric fields can be placed in `extras`.

### `DashboardFrame`

Output packet produced by `PitWit`. It contains:

- original normalized telemetry
- inference outputs
- alert status: `GREEN`, `AMBER`, or `RED`

## PitWit Runtime

### `PitWit`

```python
from pip_race import PitWit

pitwit = PitWit()
frame = pitwit.process(packet)
frames = pitwit.process_many([packet])
```

Constructor options:

- `runner`: custom inference runner, usually `OnnxRunner`
- `extractor`: custom `FeatureExtractor`
- `red_threshold`: pit-risk threshold for `RED`
- `amber_threshold`: pit-risk threshold for `AMBER`

## Inference

### `OnnxRunner`

```python
from pip_race.inference import OnnxRunner

runner = OnnxRunner("model.onnx")
```

If no model path is supplied, `OnnxRunner` uses a deterministic fallback model for tests and development. Production usage should pass an ONNX model.

### PyTorch Export

```python
from pip_race.inference.torch_model import export_onnx

export_onnx("model.onnx", input_dim=16)
```

## Data Helpers

```python
from pip_race.data import (
    frames_to_jsonl,
    frames_to_rows,
    packets_from_jsonl,
    summarize_frames,
    write_frames_csv,
)
```

- `packets_from_jsonl(path)`: load telemetry packets from JSONL.
- `frames_to_rows(frames)`: flatten frames into table rows.
- `summarize_frames(frames)`: produce aggregate operational metrics.
- `write_frames_csv(frames, path)`: write frame rows to CSV.
- `frames_to_jsonl(frames)`: serialize output frames as JSONL.

## Visualization

```python
from pip_race.visualization import (
    pit_risk_svg,
    pit_risk_vega_spec,
    status_bar_vega_spec,
    write_pit_risk_svg,
)
```

- `pit_risk_svg(frames)`: dependency-free SVG line chart.
- `write_pit_risk_svg(frames, path)`: write SVG to disk.
- `pit_risk_vega_spec(frames)`: Vega-Lite line chart spec.
- `status_bar_vega_spec(frames)`: Vega-Lite alert-count bar chart spec.

## Streaming

```python
from pip_race.streaming import RedisDashboardPublisher

publisher = RedisDashboardPublisher("redis://localhost:6379/0")
publisher.publish(frame)
```

Default outputs:

- Pub/sub channel: `pitcrew:dashboard`
- Redis Stream: `pitcrew:frames`

## CLI

```bash
pip-race infer --input telemetry.jsonl --output frames.jsonl
pip-race infer --input telemetry.jsonl --output frames.csv --format csv
pip-race infer --input telemetry.jsonl --output frames.jsonl --summary summary.json --svg pit_risk.svg
pip-race benchmark --iterations 10000 --warmup 1000
```

Use `--model model.onnx` for ONNX Runtime inference and `--redis-url redis://localhost:6379/0` to publish frames.

## Benchmarking

```python
from pip_race import run_pitwit_benchmark

result = run_pitwit_benchmark(iterations=10_000, warmup=1_000)
print(result.p50_ns, result.p95_ns, result.p99_ns)
```
