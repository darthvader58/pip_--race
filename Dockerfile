# syntax=docker/dockerfile:1

# -------- Builder image --------
FROM rust:1-bookworm AS builder
WORKDIR /app

# Create a minimal layer cache for dependencies
# Copy manifests first
COPY pit_timer_backend/Cargo.toml ./Cargo.toml

# Create a dummy src to prebuild deps
RUN mkdir -p src && echo "fn main(){}" > src/main.rs
RUN cargo build --release || true

# Now copy the real source
COPY pit_timer_backend/src ./src
COPY pit_timer_backend/src/tracks ./tracks

# Build the real binary
RUN cargo build --release

# -------- Runtime image --------
FROM debian:bookworm-slim AS runtime
WORKDIR /app
RUN useradd -m appuser

# Install CA certificates and curl (used by healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl netcat-openbsd \
	&& rm -rf /var/lib/apt/lists/*

# Copy binary and assets
COPY --from=builder /app/target/release/pit_timer_backend /app/pit_timer_backend
COPY --from=builder /app/tracks /app/tracks

USER appuser
ENV RUST_LOG=info
ENV BIND_ADDR=0.0.0.0:8765
EXPOSE 8765
CMD ["/app/pit_timer_backend"]
