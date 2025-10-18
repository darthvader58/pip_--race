#!/usr/bin/env python3
"""
Multi-driver probability plot for "box within K laps" using QR-DQN artifacts.

- Loads artifacts/rl/qrdqn.pt, meta.json, calib_platt.json
- Builds SAME features as trainer (hazard + tactical) via FastF1
- Computes per-lap probabilities for multiple drivers
- Writes one CSV per driver (optional) and ONE combined matplotlib line chart
  (probability curve per driver; pit markers use same color with 'x')

Usage:
  python scripts/predict_qrdqn_multidriver.py \
    --race "2023:Monaco" \
    [--drivers "VER,HAM,LEC"] \
    --cache data/fastf1_cache \
    --artifacts artifacts \
    --out artifacts/reports \
    --hops 2 \
    [--save_csv --save_png] [--threshold 0.6]
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import fastf1

# ----------------------------- Model (same as trainer) -----------------------------

class QRDQN(nn.Module):
    def __init__(self, in_dim: int, n_actions: int = 2, n_quantiles: int = 51, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions * n_quantiles)

    def forward(self, x):
        z = self.backbone(x)
        q = self.head(z).view(-1, self.n_actions, self.n_quantiles)
        return q

def choose_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ----------------------------- FastF1 helpers (mirror trainer) -----------------------------

COMPOUNDS = ["SOFT","MED","HARD","INTERMEDIATE","WET"]

def enable_cache(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

def load_race_session(race_spec: str):
    year_s, gp = race_spec.split(":")
    ses = fastf1.get_session(int(year_s), gp, "R")
    ses.load(laps=True, telemetry=False, weather=False)
    race_id = f"{year_s}_{gp}"
    track_name = str(ses.event.get("EventName") or ses.event.get("Location") or gp).upper()
    year_evt = int(pd.Timestamp(ses.event["EventDate"]).year)
    return ses, race_id, track_name, year_evt

def _safe_secs(td):
    try: return float(td.total_seconds())
    except Exception: return None

def lap_time_window_cols(laps_df: pd.DataFrame) -> pd.DataFrame:
    df = laps_df.copy()
    if "LapStartTime" not in df.columns or df["LapStartTime"].isna().all():
        df = df.sort_values(["Driver","LapNumber"]).reset_index(drop=True)
        df["LapStartTime"] = df.groupby("Driver")["LapTime"].shift(1).fillna(pd.Timedelta(seconds=0))
        df["LapStartTime"] = df.groupby("Driver")["LapStartTime"].cumsum()
    if "LapTime" not in df.columns:
        df["LapTime"] = pd.Timedelta(seconds=0)
    df["LapEndTime"] = df["LapStartTime"] + df["LapTime"].fillna(pd.Timedelta(seconds=0))
    return df

def non_green_in_window(track_status: pd.DataFrame, start: pd.Timedelta, end: pd.Timedelta) -> int:
    if track_status is None or len(track_status)==0: return 0
    sub = track_status[(track_status["Time"] >= start) & (track_status["Time"] <= end)]
    return int(((sub["Status"].astype(str) != "1")).any())

def build_car_lap_rows(session, race_id: str, track_key: str, year_evt: int) -> pd.DataFrame:
    rows=[]
    laps_all = lap_time_window_cols(session.laps)
    ts = getattr(session, "track_status", None)
    for drv, g in laps_all.groupby("Driver"):
        g=g.sort_values("LapNumber").reset_index(drop=True)
        stint_no=0; last_comp=None; age=0; last_full=[]
        index_map={int(r["LapNumber"]): r for _, r in g.iterrows()}
        for _, row in g.iterrows():
            lap=int(row["LapNumber"])
            comp=str(row.get("Compound","")).upper() or "HARD"
            pit_in=row.get("PitInTime"); pit_out=row.get("PitOutTime")
            if (comp != last_comp) or pd.notnull(pit_out):
                stint_no+=1; age=0; last_comp=comp
            next_lap = index_map.get(lap+1)
            pitted_this = int(pd.notnull(pit_in) or (next_lap is not None and pd.notnull(next_lap.get("PitOutTime"))))
            next2 = index_map.get(lap+2)
            pitted_next = int(next_lap is not None and (pd.notnull(next_lap.get("PitInTime")) or (next2 is not None and pd.notnull(next2.get("PitOutTime")))))
            y_within2 = int(pitted_this or pitted_next)
            start_t=row["LapStartTime"]; end_t=row["LapEndTime"]
            cheap = non_green_in_window(ts, start_t, end_t)
            rows.append({
                "race_id": race_id, "track": track_key, "year": year_evt, "driver": drv, "lap": lap,
                "stint_no": stint_no, "compound": comp, "tire_age_laps": age,
                "last_laps_json": json.dumps(last_full[-5:]),
                "cheap_stop_flag_true": cheap,
                "pitted_this_lap": pitted_this,
                "pitted_within2": y_within2
            })
            lt=row.get("LapTime")
            if pd.notnull(lt):
                s=_safe_secs(lt)
                if s is not None: last_full.append(s)
            age+=1
    return pd.DataFrame(rows)

def median_stint_lengths(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stint = (df.groupby(["race_id","driver","stint_no"], as_index=False)
               .agg({"tire_age_laps":"max","track":"first","compound":"last"}))
    out={}
    for track,g in stint.groupby("track"):
        out.setdefault(track,{})
        for comp,gc in g.groupby("compound"):
            out[track][comp]=float(gc["tire_age_laps"].median())
    return out

def make_per_lap_pit_counts(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["race_id","lap"], as_index=False)["pitted_this_lap"].sum()
             .rename(columns={"pitted_this_lap":"pits_this_lap"}))
    agg["pits_prev1"] = agg.groupby("race_id")["pits_this_lap"].shift(1).fillna(0)
    agg["pits_prev2"] = agg.groupby("race_id")["pits_this_lap"].shift(2).fillna(0)
    return agg[["race_id","lap","pits_prev1","pits_prev2"]]

def parse_last(js, k=5):
    try:
        arr = json.loads(js) if isinstance(js,str) else js
        return [float(x) for x in (arr or [])][-k:]
    except Exception:
        return []

def slope(vals):
    if len(vals)<3: return 0.0
    x = np.arange(len(vals), dtype=float); y = np.array(vals, dtype=float)
    x -= x.mean(); y -= y.mean()
    d = (x**2).sum()
    return float((x*y).sum()/d) if d else 0.0

def var3(vals):
    if len(vals)<3: return 0.0
    return float(np.var(vals[-3:]))

def hazard_features(df: pd.DataFrame, priors: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    feats=pd.DataFrame(index=df.index)
    feats["tire_age_laps"]=df["tire_age_laps"].fillna(0).clip(lower=0).astype(float)
    feats["stint_no"]=df["stint_no"].fillna(1).astype(int)
    comp=df["compound"].str.upper().fillna("HARD")
    for c in COMPOUNDS: feats[f"compound_{c}"]=(comp==c).astype(int)
    last=df["last_laps_json"].apply(parse_last)
    feats["last3_avg"]=last.apply(lambda a: float(np.mean(a[-3:])) if len(a) else 0.0)
    feats["last5_slope"]=last.apply(slope)
    feats["last3_var"]=last.apply(var3)
    med=df.apply(lambda r: priors.get(r["track"],{}).get(str(r["compound"]).upper(),None), axis=1)
    feats["typical_stint_len"]=med.fillna(0).astype(float)
    feats["age_vs_typical"]=feats["tire_age_laps"]-feats["typical_stint_len"]
    feats["age_percentile"]=(feats["tire_age_laps"]/(feats["typical_stint_len"]+1e-6)).clip(upper=1.4)
    feats["overshoot"]=(feats["tire_age_laps"]-feats["typical_stint_len"]).clip(lower=0)
    return feats

def tactical_features(df: pd.DataFrame, perlap: pd.DataFrame) -> pd.DataFrame:
    feats=pd.DataFrame(index=df.index)
    cheap=df["cheap_stop_flag_true"].astype(int)
    feats["cheap_stop_flag"]=cheap
    grp=df.groupby("race_id")
    feats["cheap_prev1"]=grp["cheap_stop_flag_true"].shift(1).fillna(0).astype(int)
    feats["cheap_prev2"]=grp["cheap_stop_flag_true"].shift(2).fillna(0).astype(int)
    # run length per race
    run_series=pd.Series(0,index=df.index,dtype=int)
    for rid, idx in df.sort_values(["race_id","lap"]).groupby("race_id").groups.items():
        sub=df.loc[idx]
        arr=sub["cheap_stop_flag_true"].astype(int).to_numpy()
        out=np.zeros_like(arr,dtype=int); c=0
        for i,v in enumerate(arr):
            c = c+1 if v==1 else 0
            out[i]=c
        run_series.loc[idx]=out
    feats["non_green_runlen"]=run_series.astype(int)
    # per-lap pit behavior
    joined=df.merge(perlap, how="left", on=["race_id","lap"]).fillna({"pits_prev1":0,"pits_prev2":0})
    feats["pits_prev1"]=joined["pits_prev1"].astype(float)
    feats["pits_prev2"]=joined["pits_prev2"].astype(float)
    # due markers
    feats["tire_age_laps"]=df["tire_age_laps"].fillna(0).clip(lower=0).astype(float)
    comp=df["compound"].str.upper().fillna("HARD")
    for c in COMPOUNDS: feats[f"compound_{c}"]=(comp==c).astype(int)
    return feats

def build_state_matrix(base: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    priors=median_stint_lengths(base)
    perlap=make_per_lap_pit_counts(base)
    H=hazard_features(base, priors)
    T=tactical_features(base, perlap)
    X=pd.concat([H,T], axis=1)
    feat_list=list(X.columns)
    return X, feat_list

# ----------------------------- Artifacts & inference -----------------------------

def load_artifacts(artifacts_dir: str):
    rl_dir = Path(artifacts_dir) / "rl"
    ckpt = torch.load(rl_dir / "qrdqn.pt", map_location="cpu")
    meta = json.loads((rl_dir / "meta.json").read_text())
    calib = json.loads((rl_dir / "calib_platt.json").read_text())
    # optional threshold (if exists)
    metrics_path = Path(artifacts_dir) / "reports" / "rl_metrics.json"
    auto_th = None
    if metrics_path.exists():
        try:
            reps = json.loads(metrics_path.read_text())
            auto_th = float(reps.get("platt_best_threshold", None))
        except Exception:
            pass
    return ckpt, meta, calib, auto_th

def build_net_from_ckpt(ckpt):
    net = QRDQN(
        in_dim=ckpt["in_dim"],
        n_actions=ckpt["n_actions"],
        n_quantiles=ckpt["n_quantiles"],
        hidden=ckpt["hidden"],
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net

def platt(scores: np.ndarray, calib: dict) -> np.ndarray:
    z = calib["coef"] * scores + calib["intercept"]
    return 1.0 / (1.0 + np.exp(-z))

def score_gap(net, device, X_block: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(X_block, dtype=torch.float32, device=device)
        qz = net(x).cpu().numpy()
        exp = qz.mean(axis=-1)
        return exp[:,1] - exp[:,0]

def prob_within2(net, device, X_block: np.ndarray, calib: dict) -> np.ndarray:
    return platt(score_gap(net, device, X_block), calib)

def prob_within3(net, device, X_block: np.ndarray, calib: dict) -> np.ndarray:
    p2_t  = platt(score_gap(net, device, X_block), calib)
    X_next = np.vstack([X_block[1:], X_block[-1:]])
    p2_t1 = platt(score_gap(net, device, X_next), calib)
    return 1.0 - (1.0 - p2_t) * (1.0 - p2_t1)

# ----------------------------- Core routine -----------------------------

def compute_probs_for_all_drivers(race_spec: str, cache_dir: str, artifacts_dir: str,
                                  hops: int, only_drivers: List[str] = None) -> Dict[str, pd.DataFrame]:
    enable_cache(cache_dir)
    ses, race_id, track, year = load_race_session(race_spec)

    # Build base across ALL drivers (keep original index!)
    base_all = build_car_lap_rows(ses, race_id, track, year)

    # Features from full grid
    X_all, feat_list = build_state_matrix(base_all)

    # Load artifacts and enforce feature order
    ckpt, meta, calib, auto_th = load_artifacts(artifacts_dir)
    model_feats = meta.get("feat_list", feat_list)

    # Ensure ALL model_feats exist in X_all (add zeros for any missing), then reorder
    for col in model_feats:
        if col not in X_all.columns:
            X_all[col] = 0.0
    # If X_all has extra columns, that’s fine; select only model feats in order:
    X_all = X_all[model_feats]

    device = choose_device()
    net = build_net_from_ckpt(ckpt).to(device)

    # Driver selection
    drivers = sorted(base_all["driver"].unique().tolist())
    if only_drivers:
        target = {d.strip().upper() for d in only_drivers}
        drivers = [d for d in drivers if str(d).upper() in target]

    results = {}
    for drv in drivers:
        # USE ORIGINAL INDICES, not reset()
        sub = base_all.loc[base_all["driver"] == drv].sort_values("lap")
        idx = sub.index  # original positions into X_all
        if len(idx) == 0:
            continue

        X_drv = X_all.loc[idx].to_numpy(dtype=np.float32)
        if hops == 2:
            probs = prob_within2(net, device, X_drv, calib)
        else:
            probs = prob_within3(net, device, X_drv, calib)

        out = sub[["lap", "cheap_stop_flag_true", "pitted_this_lap", "pitted_within2"]].copy()
        out[f"prob_within{hops}"] = probs
        results[str(drv)] = out.reset_index(drop=True)  # safe to reset for pretty output

    return results, auto_th


# ----------------------------- Plotting -----------------------------

def plot_multi_driver(results: Dict[str, pd.DataFrame], hops: int, title: str,
                      save_path: str = None, threshold: float = None, max_drivers_in_legend: int = 24):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # Plot probability line per driver and draw a "cross" at actual pit laps
    for drv, df in results.items():
        laps = df["lap"].to_numpy()
        probs = df[f"prob_within{hops}"].to_numpy()

        # main probability curve
        (line,) = ax.plot(laps, probs, linewidth=1.8, label=drv)
        color = line.get_color()

        # mark actual pit laps with a big "x" at the top (so it’s super visible)
        if "pitted_this_lap" in df.columns:
            pit_mask = df["pitted_this_lap"].to_numpy(dtype=bool)
            if pit_mask.any():
                # place crosses slightly above the plot so they don't hide the curve
                y_cross = np.full(pit_mask.sum(), 1.02)
                ax.scatter(
                    laps[pit_mask],
                    y_cross,
                    marker="x",
                    s=80,
                    linewidths=2.0,
                    color=color,
                    clip_on=False,
                    zorder=5,
                    label=f"{drv} pit"
                )
                # small vertical tick for extra readability
                ax.vlines(laps[pit_mask], 0.98, 1.02, colors=color, linewidth=1.0, alpha=0.7)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"threshold={threshold:.2f}")

    ax.set_xlabel("Lap")
    ax.set_ylabel(f"P(box ≤ {hops} laps)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # expand ylim to show the pit crosses above 1.0
    ax.set_ylim(0.0, 1.06)

    # Trim legend so it doesn’t explode with “pit” labels; keep at most N drivers + threshold
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > max_drivers_in_legend:
        kept = []
        seen = set()
        for h, lab in zip(handles, labels):
            if "pit" in lab:
                continue  # hide per-driver "pit" entries to keep legend compact
            if lab not in seen:
                kept.append((h, lab)); seen.add(lab)
            if len(kept) >= max_drivers_in_legend:
                break
        if threshold is not None:
            kept.append((plt.Line2D([0], [0], linestyle="--", color="black"), f"threshold={threshold:.2f}"))
        if kept:
            handles, labels = zip(*kept)
        else:
            handles, labels = [], []
    ax.legend(handles, labels, ncols=4 if len(labels) > 12 else 1, fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help='e.g. "2023:Monaco"')
    ap.add_argument("--drivers", default=None, help='Comma-separated driver codes to include (e.g., "VER,HAM,LEC"). Omit for all drivers.')
    ap.add_argument("--cache", default="data/fastf1_cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts/reports")
    ap.add_argument("--hops", type=int, default=2, choices=[2,3])
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="Horizontal threshold line; if omitted and metrics exist, auto-uses platt_best_threshold")
    args = ap.parse_args()

    only = [s.strip() for s in args.drivers.split(",")] if args.drivers else None

    results, auto_th = compute_probs_for_all_drivers(
        race_spec=args.race,
        cache_dir=args.cache,
        artifacts_dir=args.artifacts,
        hops=args.hops,
        only_drivers=only
    )

    if not results:
        print("No driver data to plot.")
        return

    # Save per-driver CSVs
    if args.save_csv:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for drv, df in results.items():
            csv_path = out_dir / f"{args.race.replace(':','_')}_{drv}_within{args.hops}_probs.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")

    # Plot
    th = args.threshold if args.threshold is not None else auto_th
    png_path = None
    if args.save_png:
        fname = f"{args.race.replace(':','_')}_ALL_{'SEL' if args.drivers else 'DRIVERS'}_within{args.hops}_probs.png"
        png_path = str(Path(args.out) / fname)

    title = f"{args.race} — P(box ≤ {args.hops} laps) — {'Selected drivers' if args.drivers else 'All drivers'}"
    plot_multi_driver(results, args.hops, title, save_path=png_path, threshold=th)

if __name__ == "__main__":
    main()

