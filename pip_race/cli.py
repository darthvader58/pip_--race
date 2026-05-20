from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from pip_race.contracts import HpcTelemetryPacket
from pip_race.data import frames_to_jsonl, summarize_frames, write_frames_csv
from pip_race.benchmark import run_pitwit_benchmark
from pip_race.inference import OnnxRunner
from pip_race.pitwit import PitWit
from pip_race.visualization import write_pit_risk_svg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pip-race", description="Telemetry inference runtime utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run telemetry JSONL through the PitWit runtime.")
    infer.add_argument("-i", "--input", default="-", help="Input telemetry JSONL path, or '-' for stdin.")
    infer.add_argument("-o", "--output", default="-", help="Output path, or '-' for stdout.")
    infer.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format.")
    infer.add_argument("--model", default=None, help="Optional ONNX model path.")
    infer.add_argument("--summary", default=None, help="Write summary metrics JSON to this path, or '-' for stderr.")
    infer.add_argument("--svg", default=None, help="Write a pit-risk SVG chart to this path.")
    infer.add_argument("--redis-url", default=None, help="Optionally publish each frame to Redis.")
    infer.set_defaults(func=_cmd_infer)

    bench = subparsers.add_parser("benchmark", help="Measure single-frame PitWit latency.")
    bench.add_argument("--iterations", type=int, default=10_000)
    bench.add_argument("--warmup", type=int, default=1_000)
    bench.add_argument("--model", default=None, help="Optional ONNX model path.")
    bench.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


def _cmd_infer(args: argparse.Namespace) -> None:
    packets = _read_packets(args.input, sys.stdin)
    pitwit = PitWit(runner=OnnxRunner(args.model) if args.model else None)
    frames = pitwit.process_many(packets)

    if args.redis_url:
        from pip_race.streaming import RedisDashboardPublisher

        publisher = RedisDashboardPublisher(args.redis_url)
        for frame in frames:
            publisher.publish(frame)

    _write_frames(frames, output=args.output, output_format=args.format)

    if args.summary:
        _write_summary(summarize_frames(frames), args.summary)

    if args.svg:
        write_pit_risk_svg(frames, args.svg)


def _cmd_benchmark(args: argparse.Namespace) -> None:
    pitwit = PitWit(runner=OnnxRunner(args.model) if args.model else None)
    result = run_pitwit_benchmark(pitwit=pitwit, iterations=args.iterations, warmup=args.warmup)
    print(result.to_json())


def _read_packets(input_path: str, stdin: TextIO) -> list[HpcTelemetryPacket]:
    lines = stdin if input_path == "-" else Path(input_path).open("r", encoding="utf-8")
    try:
        return [
            HpcTelemetryPacket.from_mapping(json.loads(line))
            for line in lines
            if line.strip()
        ]
    finally:
        if input_path != "-":
            lines.close()


def _write_frames(frames, output: str, output_format: str) -> None:
    if output_format == "csv":
        if output == "-":
            raise SystemExit("CSV output requires --output PATH")
        write_frames_csv(frames, output)
        return

    payload = frames_to_jsonl(frames)
    if output == "-":
        if payload:
            print(payload)
    else:
        Path(output).write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _write_summary(summary: dict, output: str) -> None:
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if output == "-":
        print(payload, file=sys.stderr)
    else:
        Path(output).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
