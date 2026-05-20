# syntax=docker/dockerfile:1
#
# Rust sidecar image for PitWit timing/fan-out experiments.
# The Python library itself is packaged by Dockerfile.inference.

FROM rust:1-bookworm AS builder
WORKDIR /app

COPY pit_timer_backend/Cargo.toml ./Cargo.toml
RUN mkdir -p src && echo "fn main(){}" > src/main.rs
RUN cargo build --release || true

COPY pit_timer_backend/src ./src
COPY pit_timer_backend/src/tracks ./tracks
RUN cargo build --release

FROM debian:bookworm-slim AS runtime
WORKDIR /app
RUN useradd -m appuser

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl netcat-openbsd \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/pit_timer_backend /app/pit_timer_backend
COPY --from=builder /app/tracks /app/tracks

USER appuser
ENV RUST_LOG=info
ENV BIND_ADDR=0.0.0.0:8765
EXPOSE 8765
CMD ["/app/pit_timer_backend"]
