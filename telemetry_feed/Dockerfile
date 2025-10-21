# Multi-stage build for Rust telemetry feed service

# Stage 1: Build
FROM rust:1.75-slim as builder

WORKDIR /usr/src/app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY tests ./tests

# Build release binary
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /usr/src/app/target/release/telemetry-feed /app/telemetry-feed

# Environment variables (can be overridden)
ENV BACKEND_URL=ws://rust-backend:8765
ENV TEAM=Williams
ENV YEAR=2023
ENV GP=Monaco
ENV SESSION_TYPE=R

# Run the telemetry feed
CMD ["/app/telemetry-feed"]