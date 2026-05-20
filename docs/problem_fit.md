# Problem-Solution Fit

## Real-World Analogue

The closest production analogue is Formula 1 Insights powered by AWS, especially the Pit Strategy Battle and pit-window prediction work. F1 describes a live race environment where hundreds of sensors per car generate more than a million telemetry data points per second, and where timing, telemetry, historical race data, streaming analytics, and machine learning are used to explain pit strategy and tactical decisions in real time.

Primary references:

- F1 and AWS official sports page: https://aws.amazon.com/sports/f1/
- F1 announcement naming AWS as cloud and ML provider: https://corp.formula1.com/formula-1-selects-aws-as-its-official-cloud-and-machine-learning-provider/
- AWS ML blog on the real-time race strategy prediction app behind Pit Strategy Battle: https://aws.amazon.com/blogs/machine-learning/accelerating-innovation-how-serverless-machine-learning-on-aws-powers-f1-insights/

## Fit for This Project

`pip-race` is the team-facing version of that pattern:

1. HPC or trackside processes emit normalized telemetry frames.
2. Python performs feature extraction and ONNX inference with sub-ms single-frame model latency.
3. PyTorch owns training and model export into ONNX.
4. Redis provides low-latency pub/sub plus a short replayable stream for downstream tools.
5. Rust services handle hard real-time fan-out and timing math when a Python-only deployment is not enough.
6. Docker Compose runs infrastructure sidecars such as Redis, the Python inference publisher, and the Rust timer service.

## Latency Budget

Target budget for a single frame on localhost or same-rack deployment:

- Feature extraction: 20-100 us
- ONNX Runtime CPU inference for the compact MLP: 50-500 us
- Redis publish: 100-400 us on local network
- Rust fan-out/timer sidecar: 50-300 us in colocated deployments

The sub-ms target should be interpreted as model-path latency or backend runtime latency under colocated services. Notebook rendering, BI visualization, browser paint, and WAN transport are separate budgets.

## Initial ML Task

The first inference target is a compact multi-output model:

- `pit_risk`: probability that the car should be escalated to pit-window attention.
- `tire_degradation`: normalized tire degradation pressure.
- `confidence`: calibration signal for dashboard display and alert gating.

This keeps the system useful before full proprietary tire-energy labels exist, while still matching the real F1 use case: convert high-rate car telemetry into fast pitwit decisions.
