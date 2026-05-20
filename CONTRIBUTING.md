# Contributing

Thanks for improving `pip-race`. The project is a Python library first, with optional Redis and Rust sidecars.

## Local Setup

```bash
pip install -e ".[dev,ml,streaming]"
```

## Checks

```bash
python3 -m pytest tests/test_pitwit.py telemetry_feed/test_speed_profile.py
cd pit_timer_backend && cargo check
```

## Project Boundaries

- Keep reusable behavior inside `pip_race/`.
- Do not add generated folders such as `node_modules`, `target`, caches, or reports.
- Prefer library APIs, examples, docs, and tests over hosted web-app code.
- Keep visualizations exportable as data, SVG, or specs that downstream tools can render.
- Keep public runtime naming centered on `PitWit`.
