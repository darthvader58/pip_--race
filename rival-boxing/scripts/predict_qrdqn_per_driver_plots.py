#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

import fastf1
from fastf1 import plotting as f1plot

# -------------------- Helpers --------------------

def load_artifacts(artifacts_dir: str) -> Tuple[torch.jit.ScriptModule, Dict, Dict]:
    art = Path(artifacts_dir)
    meta = json.loads((art / "rl" / "meta.json").read_text())
    ts_path = art / "rl" / "qrdqn_torchscript.pt"
    model = torch.jit.load(str(ts_path), map_location="cpu")
    model.eval()

    calib_path = art / "rl" / "calib_platt.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else {"a": 1.0, "b": 0.0}
    return model, meta, calib

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def platt(gap: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * gap + b)))

def ensure_outdir(base: Path, race_tag: str) -> Path:
    out = base / race_tag / "drivers"
    out.mkdir(parents=True, exist_ok=True)
    return out

# -------------------- Feature builder (matches realtime feeder ordering) --------------------

FEAT_ORDER_DEFAULT_26 = [
    "tire_age_laps",
    "stint_no",
    "compound_SOFT",
    "compound_MED",
    "compound_HARD",
    "compound_INTERMEDIATE",
    "compound_WET",
    "last3_avg",
    "last5_slope",
    "last3_var",
    "typical_stint_len",
    "age_vs_typical",
    "age_percentile",
    "overshoot",
    "cheap_stop_flag",
    "cheap_prev1",
    "cheap_prev2",
    "non_green_runlen",
    "pits_prev1",
    "pits_prev2",
    "tire_age_laps",            # duplicated in training meta
    "compound_SOFT",            # duplicated in training meta
    "compound_MED",
    "compound_HARD",
    "compound_INTERMEDIATE",
    "compound_WET",
]

COMPOUND_KEYS = ["SOFT", "MED", "HARD", "INTERMEDIATE", "WET"]

def typical_stint_by_compound(stint_lengths: List[int]) -> float:
    if len(stint_lengths) == 0:
        return 18.0
    return float(np.median(stint_lengths))

def compute_driver_features(laps_df, drv_code: str, feat_list: List[str]) -> Tuple[np.ndarray, List[int], List[int]]:
    """Returns (X [n_laps x in_dim], lap_numbers, pit_laps) for a driver."""
    dlaps = laps_df.pick_driver(drv_code).reset_index(drop=True)
    if len(dlaps) == 0:
        return np.zeros((0, len(feat_list)), dtype=np.float32), [], []

    # Basic series
    lap_numbers = dlaps['LapNumber'].to_numpy(int).tolist()
    lap_time_s = dlaps['LapTime'].dt.total_seconds().fillna(method='ffill').fillna(method='bfill').fillna(0.0).to_numpy(float)
    s1 = dlaps['Sector1Time'].dt.total_seconds().fillna(0.0).to_numpy(float)
    s2 = dlaps['Sector2Time'].dt.total_seconds().fillna(0.0).to_numpy(float)
    s3 = dlaps['Sector3Time'].dt.total_seconds().fillna(0.0).to_numpy(float)

    # Pit flags (actual)
    pit_in = dlaps['PitInTime'].notna().to_numpy(bool)
    pit_laps = [int(n) for n, p in zip(lap_numbers, pit_in) if p]

    # Stint number and tyre age per FastF1
    stint_no = dlaps['Stint'].fillna(method='ffill').fillna(1).to_numpy(int)
    tyre_age = dlaps['TyreLife'].fillna(0).to_numpy(int)

    # Compound one-hots
    comp = dlaps['Compound'].fillna("UNKNOWN").astype(str).str.upper().to_list()
    comp_oh = np.zeros((len(comp), len(COMPOUND_KEYS)), dtype=np.float32)
    comp_idx = {k: i for i, k in enumerate(COMPOUND_KEYS)}
    for i, c in enumerate(comp):
        if c in comp_idx:
            comp_oh[i, comp_idx[c]] = 1.0

    # Rolling stats
    def rolling_avg(x, w):
        y = np.copy(x)
        for i in range(len(x)):
            s = max(0, i - w + 1)
            y[i] = float(np.mean(x[s:i+1]))
        return y

    def rolling_var(x, w):
        y = np.copy(x)
        for i in range(len(x)):
            s = max(0, i - w + 1)
            y[i] = float(np.var(x[s:i+1]))
        return y

    def rolling_slope(x, w):
        y = np.zeros_like(x)
        for i in range(len(x)):
            s = max(0, i - w + 1)
            xs = np.arange(i - s + 1, dtype=np.float64)
            ys = x[s:i+1].astype(np.float64)
            if len(xs) >= 2:
                A = np.vstack([xs, np.ones_like(xs)]).T
                m, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
                y[i] = float(m)
            else:
                y[i] = 0.0
        return y

    last3_avg  = rolling_avg(lap_time_s, 3)
    last3_var  = rolling_var(lap_time_s, 3)
    last5_slope= rolling_slope(lap_time_s, 5)

    # Typical stint length proxy per compound (coarse)
    # Here we approximate by median tyre life observed when a pit happens on that compound.
    # If none, fallback to overall median 18.
    typ_len = np.full(len(dlaps), 18.0, dtype=np.float32)
    # crude by-stint lens:
    stint_lens = {}
    for st in np.unique(stint_no):
        idx = np.where(stint_no == st)[0]
        if len(idx) > 0:
            stint_lens[st] = len(idx)
    # use current stint length as proxy "typical"
    for i in range(len(dlaps)):
        typ_len[i] = float(stint_lens.get(stint_no[i], 18))

    age_vs_typ = tyre_age - typ_len
    with np.errstate(divide='ignore', invalid='ignore'):
        age_pct = np.clip(tyre_age / np.maximum(typ_len, 1e-3), 0.0, 4.0)

    # "overshoot" indicator when past typical
    overshoot = (tyre_age > typ_len).astype(np.float32)

    # Cheap-stop flags: use TrackStatus (0=green), non-green when non-zero
    track_status = dlaps['TrackStatus'].fillna(0).astype(int).to_numpy()
    cheap_flag = (track_status != 0).astype(np.float32)

    # cheap_prev1/prev2: previous laps having non-green
    cheap_prev1 = np.roll(cheap_flag, 1); cheap_prev1[0] = 0.0
    cheap_prev2 = np.roll(cheap_flag, 2); cheap_prev2[:2] = 0.0

    # non_green_runlen: consecutive non-green run length
    non_green_run = np.zeros(len(track_status), dtype=np.float32)
    r = 0
    for i, st in enumerate(track_status):
        if st != 0:
            r += 1
        else:
            r = 0
        non_green_run[i] = float(r)

    # pits_prev1/prev2: recent pit flags
    pit_prev1 = np.roll(pit_in.astype(np.float32), 1); pit_prev1[0] = 0.0
    pit_prev2 = np.roll(pit_in.astype(np.float32), 2); pit_prev2[:2] = 0.0

    # Assemble feature matrix in the exact feat_list order
    cols = {
        "tire_age_laps": tyre_age.astype(np.float32),
        "stint_no": stint_no.astype(np.float32),
        "compound_SOFT": comp_oh[:, 0],
        "compound_MED":  comp_oh[:, 1],
        "compound_HARD": comp_oh[:, 2],
        "compound_INTERMEDIATE": comp_oh[:, 3],
        "compound_WET": comp_oh[:, 4],
        "last3_avg": last3_avg.astype(np.float32),
        "last5_slope": last5_slope.astype(np.float32),
        "last3_var": last3_var.astype(np.float32),
        "typical_stint_len": typ_len.astype(np.float32),
        "age_vs_typical": age_vs_typ.astype(np.float32),
        "age_percentile": age_pct.astype(np.float32),
        "overshoot": overshoot.astype(np.float32),
        "cheap_stop_flag": cheap_flag.astype(np.float32),
        "cheap_prev1": cheap_prev1.astype(np.float32),
        "cheap_prev2": cheap_prev2.astype(np.float32),
        "non_green_runlen": non_green_run.astype(np.float32),
        "pits_prev1": pit_prev1.astype(np.float32),
        "pits_prev2": pit_prev2.astype(np.float32),
    }

    # Some feat names in meta may repeat (as in training). Respect order strictly:
    X = np.zeros((len(dlaps), len(feat_list)), dtype=np.float32)
    for j, name in enumerate(feat_list):
        if name in cols:
            X[:, j] = cols[name]
        else:
            # If duplicate keys like tire_age_laps appear again, still map from cols
            if name in cols:
                X[:, j] = cols[name]
            else:
                # unseen column name — fill 0
                X[:, j] = 0.0

    return X, lap_numbers, pit_laps

def predict_p2_from_torchscript(model, X: np.ndarray, calib: Dict) -> np.ndarray:
    """Run model(X) and turn into calibrated p2 using the QR-DQN gap -> logistic + Platt."""
    with torch.no_grad():
        t = torch.from_numpy(X).float()
        out = model(t)            # [N, A, Q]
        if isinstance(out, (list, tuple)):
            out = out[0]
        # mean over quantiles
        q_mean = out.mean(dim=2)  # [N, A]
        gap = q_mean[:, 1] - q_mean[:, 0]  # pit - stay
        gap = gap.cpu().numpy()
    p = platt(gap, calib.get("a", 1.0), calib.get("b", 0.0))
    return p

def plot_driver(race_tag: str, drv: str, laps: List[int], p2: np.ndarray, pit_laps: List[int], out_dir: Path):
    plt.figure(figsize=(9, 4.2))
    plt.plot(laps, p2, linewidth=2)
    # crosses for pit-in laps
    for L in pit_laps:
        plt.plot([L], [p2[laps.index(L)] if L in laps else 0.0], marker="x", markersize=8)

    plt.ylim(0.0, 1.0)
    plt.xlim(min(laps), max(laps) if len(laps) else 1)
    plt.xlabel("Lap")
    plt.ylabel("P(pit within 2 laps)")
    plt.title(f"{race_tag} — {drv}")
    plt.grid(True, alpha=0.3)
    out_path = out_dir / f"{drv}_p2.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help='e.g. "2023:Monaco"')
    ap.add_argument("--cache", default="data/fastf1_cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--only", default="", help='Comma sep driver codes to filter, e.g. "VER,LEC,HAM"')
    args = ap.parse_args()

    # Artifacts
    model, meta, calib = load_artifacts(args.artifacts)
    feat_list = meta.get("feat_list", FEAT_ORDER_DEFAULT_26)
    in_dim = len(feat_list)

    # FastF1
    fastf1.Cache.enable_cache(args.cache)
    year, gp = args.race.split(":", 1)
    session = fastf1.get_session(int(year), gp.strip(), "R")
    session.load()

    # Select drivers
    all_drv = [session.get_driver(driver)["Abbreviation"] for driver in session.drivers]
    only = [s.strip().upper() for s in args.only.split(",") if s.strip()] if args.only else []
    drivers = [d for d in all_drv if (not only or d in only)]

    race_tag = f"{year}_{gp.strip().replace(' ', '')}"
    out_dir = ensure_outdir(Path(args.out), race_tag)

    # For each driver → build features → predict → plot
    for drv in drivers:
        X, laps, pit_laps = compute_driver_features(session.laps, drv, feat_list)
        if X.shape[0] == 0:
            print(f"[SKIP] {drv}: no laps")
            continue
        # Guard: feature dimension must match
        if X.shape[1] != in_dim:
            print(f"[WARN] {drv}: X.shape[1] ({X.shape[1]}) != in_dim ({in_dim}); skipping")
            continue

        p2 = predict_p2_from_torchscript(model, X, calib)  # np.array len = n_laps
        plot_driver(race_tag, drv, laps, p2, pit_laps, out_dir)
        print(f"[OK] wrote {drv} → {out_dir / (drv + '_p2.png')}")

    print(f"Done. PNGs in {out_dir}")

if __name__ == "__main__":
    main()
