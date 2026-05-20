# Low-Latency Inference Workflow

`pip-race` is designed around a multi-language deployment shape:

- **Python + PyTorch** for model development, training, feature experiments, and ONNX export.
- **ONNX Runtime** for production model execution.
- **C++** for optional native scoring kernels and future custom preprocessing/postprocessing kernels.
- **Rust** for async fan-out, timing sidecars, and low-latency service boundaries.
- **Linux** as the target production runtime for predictable scheduling, CPU affinity, and container deployment.

## Reference Runtime Path

```text
Linux host / HPC node
  |
  +-- Python process
  |     HpcTelemetryPacket
  |       -> FeatureExtractor
  |       -> OnnxRunner
  |          -> ONNX Runtime CPUExecutionProvider
  |          -> optional C++ native fallback kernel
  |       -> DashboardFrame
  |
  +-- Redis sidecar
  |     pub/sub + replayable streams
  |
  +-- Rust sidecar
        pit timing, fan-out, websocket or TCP integration
```

## C++ Native Kernel

The optional C++ scorer lives in `cpp/pip_race_native`.

Build on Linux:

```bash
cmake -S cpp/pip_race_native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
export PIP_RACE_NATIVE_LIB="$PWD/build/native/libpip_race_native.so"
```

On macOS for local smoke testing:

```bash
mkdir -p build/native
c++ -std=c++17 -O3 -fPIC -shared cpp/pip_race_native/pip_race_native.cpp \
  -o build/native/libpip_race_native.dylib
export PIP_RACE_NATIVE_LIB="$PWD/build/native/libpip_race_native.dylib"
```

When no ONNX model path is supplied, `OnnxRunner` will try to load this native scorer first. If unavailable, it uses the pure-Python deterministic fallback.

## Sub-Millisecond Practices

These are the operating assumptions for sub-ms model-path latency:

- keep batch size at 1 for live telemetry decisions
- pre-load ONNX sessions and warm them before green-flag operation
- keep feature vectors compact and fixed-width
- avoid per-frame dynamic imports and large allocations
- colocate Redis/Rust/Python services on the same host or low-latency network
- pin CPU cores for inference workers on Linux
- run with a performance CPU governor on dedicated hosts
- keep visualization/report generation off the hot path

## Recommended Linux Runtime Flags

Example process-level practices:

```bash
taskset -c 2 python -m pip_race.cli infer --input telemetry.jsonl --output frames.jsonl
```

For benchmarking, run on an otherwise quiet host and report:

- CPU model
- Python version
- model runtime: ONNX / C++ native / Python fallback
- p50 / p95 / p99 latency
- batch size
- number of frames

The library includes a benchmark command:

```bash
pip-race benchmark --iterations 10000 --warmup 1000
```

Programmatic usage:

```python
from pip_race import run_pipeline_benchmark

result = run_pipeline_benchmark(iterations=10_000, warmup=1_000)
print(result.to_json())
```

## Rust Sidecar

The Rust sidecar in `pit_timer_backend` is not required for the Python library, but it is useful when deployment needs:

- async websocket/TCP fan-out
- hard timing loops
- stable low-allocation telemetry services
- separation between model inference and race-control integration

Run:

```bash
cd pit_timer_backend
cargo check
cargo run
```
