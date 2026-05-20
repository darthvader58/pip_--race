# pip-race

`pip-race` is an open-source Python library for building low-latency motorsport telemetry inference pipelines. It normalizes high-rate HPC/trackside telemetry, extracts model features, runs ONNX/PyTorch-derived inference, returns dashboard-ready decision frames, and generates data tables and visualizations directly from the library.

The project is intentionally a library, not a hosted web app. There is no Vercel deployment and no bundled React dashboard. Consumers can embed `pip-race` in notebooks, race-engineering tools, internal dashboards, Redis streams, Rust sidecars, or batch analysis jobs.

## Why This Exists

Modern race strategy systems look like the same problem class as F1 pit-wall analytics: live car telemetry, historical stint context, track status, and tire behavior must be converted into tactical decisions quickly enough for race engineers to act. `pip-race` provides a compact, inspectable version of that pipeline for experimentation and production-style prototyping.

See [docs/problem_fit.md](docs/problem_fit.md) for the real-world problem-solution fit.

## What The Library Does

- Normalizes HPC or simulator telemetry into typed Python objects.
- Extracts online features from speed, tire age, compound, throttle/brake, track status, and lap position.
- Runs model inference through ONNX Runtime when a model is supplied.
- Provides a deterministic fallback model for development and tests.
- Returns structured `DashboardFrame` objects with pit risk, tire degradation, confidence, latency, and alert status.
- Converts inference frames into table rows, JSONL, CSV, and summary metrics.
- Generates built-in visualizations without a web frontend:
  - Vega-Lite specs for notebooks and BI tools.
  - Dependency-free SVG pit-risk charts.
- Publishes optional Redis pub/sub and Redis Stream frames for external dashboards.
- Keeps Rust available as a low-latency timing/fan-out sidecar, not as the primary library interface.

## Pipeline

```text
Telemetry source
  FastF1 replay, simulator, HPC stream, or live adapter
        |
        v
HpcTelemetryPacket
  normalized typed contract
        |
        v
FeatureExtractor
  online rolling features + tactical state
        |
        v
OnnxRunner
  ONNX Runtime model or deterministic dev fallback
        |
        v
PitWallPipeline
  DashboardFrame: telemetry + inference + alert status
        |
        +--> frames_to_rows / summarize_frames / write_frames_csv
        +--> pit_risk_svg / Vega-Lite specs
        +--> RedisDashboardPublisher
        +--> Rust pit-timer sidecar
```

## Install

For core library usage:

```bash
pip install -e .
```

For ML export and ONNX inference:

```bash
pip install -e ".[ml]"
```

For Redis streaming:

```bash
pip install -e ".[streaming]"
```

For development:

```bash
pip install -e ".[dev,ml,streaming]"
```

## Quick Start

```python
from pip_race import HpcTelemetryPacket, PitWallPipeline
from pip_race.data import frames_to_rows, summarize_frames
from pip_race.visualization import pit_risk_svg

packets = [
    HpcTelemetryPacket(
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
]

pipeline = PitWallPipeline()
frames = pipeline.process_many(packets)

print(summarize_frames(frames))
print(frames_to_rows(frames))

svg = pit_risk_svg(frames)
```

## Core APIs

See [docs/api.md](docs/api.md) for the concise API reference.

### Telemetry Contracts

```python
from pip_race import HpcTelemetryPacket

packet = HpcTelemetryPacket(
    car_id="SAR",
    lap=35,
    lap_distance_m=1980.0,
    speed_kph=211.4,
    tire_age_laps=18,
    compound="HARD",
    track_status="GREEN",
)
```

### Inference

```python
from pip_race import PitWallPipeline

pipeline = PitWallPipeline()
frame = pipeline.process(packet)

print(frame.inference.pit_risk)
print(frame.inference.tire_degradation)
print(frame.status)
```

To use an ONNX model:

```python
from pip_race.inference import OnnxRunner
from pip_race import PitWallPipeline

pipeline = PitWallPipeline(runner=OnnxRunner("model.onnx"))
```

### Data Products

```python
from pip_race.data import frames_to_rows, summarize_frames, write_frames_csv

rows = frames_to_rows(frames)
summary = summarize_frames(frames)
write_frames_csv(frames, "race_frames.csv")
```

### Visualizations

```python
from pip_race.visualization import (
    pit_risk_svg,
    pit_risk_vega_spec,
    status_bar_vega_spec,
    write_pit_risk_svg,
)

write_pit_risk_svg(frames, "pit_risk.svg")
vega_line = pit_risk_vega_spec(frames)
vega_status = status_bar_vega_spec(frames)
```

The SVG renderer is dependency-free and works in scripts, CI, notebooks, and reports. The Vega-Lite specs are plain dictionaries that can be handed to notebook renderers, dashboards, or BI systems.

### Redis Streaming

```python
from pip_race.streaming import RedisDashboardPublisher

publisher = RedisDashboardPublisher("redis://localhost:6379/0")
publisher.publish(frame)
```

By default, frames are published to:

- Pub/sub channel: `pitcrew:dashboard`
- Redis Stream: `pitcrew:frames`

## Model Lifecycle

`pip_race.inference.torch_model` provides a compact PyTorch model factory and ONNX export helper:

```python
from pip_race.inference.torch_model import export_onnx

export_onnx("model.onnx", input_dim=16)
```

The intended lifecycle is:

1. Train or fine-tune a PyTorch model.
2. Export to ONNX.
3. Load with `OnnxRunner`.
4. Run through `PitWallPipeline`.
5. Export frames as data, visualization specs, SVG reports, or Redis messages.

Use [docs/model_card_template.md](docs/model_card_template.md) for every model you train or publish.

## Docker

Docker is provided only for infrastructure components, not for a hosted web UI.

```bash
docker compose up redis
docker compose --profile pipeline up inference
```

The Rust pit-timer sidecar can still be run when you need low-latency timing math:

```bash
docker compose up pit-timer
```

## Repository Layout

```text
pip_race/                 Python library
  contracts.py            Typed telemetry and inference contracts
  features.py             Online feature extraction
  pipeline.py             End-to-end inference orchestration
  data.py                 Rows, summaries, CSV, JSONL
  visualization.py        SVG and Vega-Lite visualization helpers
  inference/              ONNX and PyTorch model utilities
  streaming/              Redis publishers
pit_timer_backend/        Optional Rust timing sidecar
telemetry_feed/           FastF1/replay utilities and speed-profile tests
examples/                 Library usage examples
tests/                    Python tests
docs/                     Design and problem-fit documentation
```

## Development

```bash
python3 -m pytest tests/test_pipeline.py telemetry_feed/test_speed_profile.py
cd pit_timer_backend && cargo check
```

Run the example:

```bash
python examples/basic_pipeline.py
```

Run the CLI:

```bash
pip-race infer --input telemetry.jsonl --output frames.jsonl --summary summary.json --svg pit_risk.svg
pip-race infer --input telemetry.jsonl --output frames.csv --format csv
```

## Project Status

This is an early-stage library foundation. The public API is intentionally small and centered around `HpcTelemetryPacket`, `PitWallPipeline`, data exporters, visualization helpers, ONNX inference, Redis streaming, and the optional Rust sidecar.

## License

MIT. See [LICENSE](LICENSE).
