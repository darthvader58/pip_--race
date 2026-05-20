from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from pip_race.contracts import DashboardFrame
from pip_race.data import frames_to_rows


STATUS_COLORS = {
    "GREEN": "#16803c",
    "AMBER": "#b7791f",
    "RED": "#c53030",
    "LOCKED_OUT": "#4a5568",
}


def pit_risk_vega_spec(frames: list[DashboardFrame]) -> dict[str, Any]:
    """Build a Vega-Lite spec for pit risk over lap distance."""

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Pit risk by lap distance.",
        "data": {"values": frames_to_rows(frames)},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "lap_distance_m", "type": "quantitative", "title": "Lap distance (m)"},
            "y": {"field": "pit_risk", "type": "quantitative", "title": "Pit risk", "scale": {"domain": [0, 1]}},
            "color": {"field": "car_id", "type": "nominal", "title": "Car"},
            "tooltip": [
                {"field": "car_id", "type": "nominal"},
                {"field": "lap", "type": "ordinal"},
                {"field": "pit_risk", "type": "quantitative", "format": ".2f"},
                {"field": "tire_degradation", "type": "quantitative", "format": ".2f"},
                {"field": "status", "type": "nominal"},
            ],
        },
    }


def status_bar_vega_spec(frames: list[DashboardFrame]) -> dict[str, Any]:
    """Build a Vega-Lite spec showing alert-state counts."""

    counts: dict[str, int] = {}
    for frame in frames:
        counts[frame.status] = counts.get(frame.status, 0) + 1
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Alert status counts.",
        "data": {"values": [{"status": key, "count": value} for key, value in sorted(counts.items())]},
        "mark": "bar",
        "encoding": {
            "x": {"field": "status", "type": "nominal", "title": "Status"},
            "y": {"field": "count", "type": "quantitative", "title": "Frames"},
            "color": {"field": "status", "type": "nominal", "scale": {"domain": list(STATUS_COLORS), "range": list(STATUS_COLORS.values())}},
        },
    }


def pit_risk_svg(frames: list[DashboardFrame], width: int = 900, height: int = 320) -> str:
    """Render a dependency-free SVG line chart for pit risk."""

    rows = frames_to_rows(frames)
    if not rows:
        return _empty_svg(width, height, "No frames")

    margin_left = 56
    margin_right = 24
    margin_top = 22
    margin_bottom = 42
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_x = min(row["lap_distance_m"] for row in rows)
    max_x = max(row["lap_distance_m"] for row in rows)
    x_span = max(max_x - min_x, 1.0)

    def x_scale(value: float) -> float:
        return margin_left + ((value - min_x) / x_span) * plot_w

    def y_scale(value: float) -> float:
        return margin_top + (1.0 - value) * plot_h

    by_car: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (item["car_id"], item["lap"], item["lap_distance_m"])):
        by_car.setdefault(row["car_id"], []).append(row)

    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    polylines: list[str] = []
    points: list[str] = []
    legend: list[str] = []
    for idx, (car_id, car_rows) in enumerate(by_car.items()):
        color = palette[idx % len(palette)]
        coords = " ".join(f"{x_scale(row['lap_distance_m']):.2f},{y_scale(row['pit_risk']):.2f}" for row in car_rows)
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{coords}" />')
        for row in car_rows:
            points.append(
                f'<circle cx="{x_scale(row["lap_distance_m"]):.2f}" cy="{y_scale(row["pit_risk"]):.2f}" '
                f'r="3.5" fill="{STATUS_COLORS.get(row["status"], color)}"><title>{escape(car_id)} '
                f'pit risk {row["pit_risk"]:.2f}</title></circle>'
            )
        legend_y = margin_top + idx * 18
        legend.append(f'<rect x="{width - 110}" y="{legend_y - 9}" width="10" height="10" fill="{color}" />')
        legend.append(f'<text x="{width - 94}" y="{legend_y}" font-size="12" fill="#1f2937">{escape(car_id)}</text>')

    grid = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_scale(tick)
        grid.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" />')
        grid.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{tick:.2f}</text>')

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
            '<title>Pit risk over lap distance</title>',
            '<rect width="100%" height="100%" fill="white" />',
            *grid,
            f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{height - margin_bottom}" stroke="#111827" />',
            f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{height - margin_bottom}" y2="{height - margin_bottom}" stroke="#111827" />',
            f'<text x="{width / 2:.2f}" y="{height - 10}" text-anchor="middle" font-size="13" fill="#111827">Lap distance (m)</text>',
            f'<text x="16" y="{height / 2:.2f}" transform="rotate(-90 16 {height / 2:.2f})" text-anchor="middle" font-size="13" fill="#111827">Pit risk</text>',
            *polylines,
            *points,
            *legend,
            "</svg>",
        ]
    )


def write_pit_risk_svg(frames: list[DashboardFrame], path: str | Path, width: int = 900, height: int = 320) -> None:
    """Write the dependency-free pit-risk SVG chart to disk."""

    Path(path).write_text(pit_risk_svg(frames, width=width, height=height), encoding="utf-8")


def _empty_svg(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white" />'
        f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" fill="#4b5563">{escape(message)}</text>'
        "</svg>"
    )
