# Model Card Template

Use this template for every model shipped with or evaluated through `pip-race`.

## Model Details

- Name:
- Version:
- Owner:
- Model format: ONNX / TorchScript / PyTorch checkpoint
- Input feature set:
- Output targets:
- Training code:
- Training date:

## Intended Use

Describe the operating context, for example:

- offline race replay analysis
- simulator strategy research
- real-time pitwit decision support
- dashboard alerting

## Not Intended For

List explicit non-goals and safety boundaries, such as:

- autonomous pit-call execution without human review
- safety-critical control loops
- betting, market making, or compliance-sensitive decisions

## Data

- Data source:
- Tracks:
- Seasons:
- Sessions:
- Sampling rate:
- Label definition:
- Known missing data:

## Features

List all features in exact inference order.

## Metrics

Include at least:

- classification or regression metric tied to the target
- calibration metric if outputs are probabilities
- latency percentile on target hardware
- train/validation/test split details

## Latency

- Hardware:
- Batch size:
- Runtime:
- p50:
- p95:
- p99:

## Limitations

Document known failure modes:

- unusual safety-car patterns
- wet/intermediate tire transitions
- tracks not represented in training
- sparse historical data for a driver/team

## Ethical And Operational Notes

Document human-in-the-loop expectations, monitoring requirements, and how false positives/negatives should be handled.

## Release Checklist

- [ ] Feature order frozen and documented
- [ ] ONNX export validated
- [ ] `PitWit` smoke test added
- [ ] Calibration checked
- [ ] Latency measured
- [ ] Model card completed
