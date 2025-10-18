#!/usr/bin/env python3
"""
Predict pit probability (within 2–3 laps) for any driver using QR-DQN artifacts.

- Loads artifacts from your trainer: artifacts/rl/qrdqn.pt, meta.json, calib_platt.json
- Rebuilds the SAME features used in training (hazard + tactical)
- Produces per-lap probability that the selected driver will box within the next K laps
  * K=2: model-native (trained horizon)
  * K=3: heuristic composition using next-lap state (see _prob_within3)

Outputs (optional):
- CSV of [lap, prob_withinK, cheap_flag, pitted_this_lap, pitted_within2, ...]
- PNG chart of probability curve (with threshold and actual pit markers)

Usage:
  python scripts/predict_qrdqn.py \
    --race "2023:Monaco" --driver VER \
    --cache data/fastf1_cache --artifacts artifacts --out artifacts/reports \
    --hops 2 --save_csv --save_png
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

# ----------------------------- FastF1 helpers (mirrors trainer) -----------------------------

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

# ----------------------------- Loading artifacts -----------------------------

def load_artifacts(artifacts_dir: str):
    rl_dir = Path(artifacts_dir) / "rl"
    ckpt = torch.load(rl_dir / "qrdqn.pt", map_location="cpu")
    meta = json.loads((rl_dir / "meta.json").read_text())
    calib = json.loads((rl_dir / "calib_platt.json").read_text())
    return ckpt, meta, calib

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

# ----------------------------- Probability helpers -----------------------------

def _score_to_prob(scores: np.ndarray, calib: dict) -> np.ndarray:
    # Platt scaling
    z = calib["coef"] * scores + calib["intercept"]
    return 1.0 / (1.0 + np.exp(-z))

def _prob_within2(net, device, X_block: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor(X_block, dtype=torch.float32, device=device)
        qz = net(x).cpu().numpy()              # (N,2,Q)
        exp = qz.mean(axis=-1)                 # (N,2)
        scores = exp[:,1] - exp[:,0]           # BOX - NOBOX
    return scores

def _prob_within3(net, device, X_block: np.ndarray, calib: dict) -> np.ndarray:
    """
    Simple heuristic to extend horizon:
    P(within3 @ t) = 1 - (1 - p2(t)) * (1 - p2(t+1))
    where p2 is the calibrated within-2 probability. For t+1, we use the next state's features.
    Last lap uses only p2(t).
    """
    scores_t = _prob_within2(net, device, X_block)
    p2_t = _score_to_prob(scores_t, calib)

    # shift X by one for t+1
    X_next = np.vstack([X_block[1:], X_block[-1:]])   # duplicate last for boundary
    scores_t1 = _prob_within2(net, device, X_next)
    p2_t1 = _score_to_prob(scores_t1, calib)

    p3 = 1.0 - (1.0 - p2_t) * (1.0 - p2_t1)
    return p3

# ----------------------------- Main prediction routine -----------------------------

def predict_for_driver(race_spec: str, driver_code: str, cache_dir: str, artifacts_dir: str,
                       hops: int = 2) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      lap, prob_withinK, cheap_stop_flag_true, pitted_this_lap, pitted_within2
    """
    enable_cache(cache_dir)
    ses, race_id, track, year = load_race_session(race_spec)

    # Build the full race base table (all drivers), then filter to driver
    base_all = build_car_lap_rows(ses, race_id, track, year)
    base = base_all[base_all["driver"].astype(str).str.upper() == driver_code.upper()].copy()
    base = base.sort_values("lap").reset_index(drop=True)

    # Features (use global priors/per-lap computed on full grid to match trainer)
    X_all, feat_list = build_state_matrix(base_all)
    X_drv = X_all.loc[base.index, feat_list].to_numpy(dtype=np.float32)

    # Load model + calib
    ckpt, meta, calib = load_artifacts(artifacts_dir)
    # Safety: enforce feature order from meta if present
    meta_feats = meta.get("feat_list", feat_list)
    # Reindex X columns to meta order if needed
    if meta_feats != feat_list:
        # Build mapped frame to meta order
        X_all_meta = X_all[meta_feats]
        X_drv = X_all_meta.loc[base.index].to_numpy(dtype=np.float32)

    device = choose_device()
    net = build_net_from_ckpt(ckpt).to(device)

    # Scores -> prob
    if hops == 2:
        scores = _prob_within2(net, device, X_drv)
        probs = _score_to_prob(scores, calib)
    elif hops == 3:
        probs = _prob_within3(net, device, X_drv, calib)
    else:
        raise ValueError("--hops must be 2 or 3")

    out = base[["lap","cheap_stop_flag_true","pitted_this_lap","pitted_within2"]].copy()
    colname = f"prob_within{hops}"
    out[colname] = probs
    return out

# ----------------------------- Plotting -----------------------------

def plot_probability(df: pd.DataFrame, hops: int, title: str, save_path: str = None,
                     threshold: float = None):
    col = f"prob_within{hops}"
    laps = df["lap"].values
    probs = df[col].values

    plt.figure(figsize=(12,4))
    plt.plot(laps, probs, linewidth=2, label=f"P(box ≤ {hops} laps)")
    if threshold is not None:
        plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")
    # ground truth markers
    if "pitted_this_lap" in df.columns:
        pit_laps = df.loc[df["pitted_this_lap"]==1, "lap"].values
        if len(pit_laps):
            plt.scatter(pit_laps, np.clip(probs[df["pitted_this_lap"]==1], 0, 1),
                        marker="x", s=60, label="actual pit (this lap)")
    plt.xlabel("Lap")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help='e.g. "2023:Monaco"')
    ap.add_argument("--driver", required=True, help='Driver code, e.g. VER, HAM, LEC')
    ap.add_argument("--cache", default="data/fastf1_cache")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts/reports")
    ap.add_argument("--hops", type=int, default=2, choices=[2,3], help="Within-2 (model-native) or within-3 (heuristic)")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="Optional horizontal threshold line on the chart")
    args = ap.parse_args()

    df = predict_for_driver(
        race_spec=args.race,
        driver_code=args.driver,
        cache_dir=args.cache,
        artifacts_dir=args.artifacts,
        hops=args.hops
    )

    # save CSV
    if args.save_csv:
        csv_path = Path(args.out) / f"{args.race.replace(':','_')}_{args.driver}_within{args.hops}_probs.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    # plot
    if args.save_png:
        png_path = Path(args.out) / f"{args.race.replace(':','_')}_{args.driver}_within{args.hops}_probs.png"
        title = f"{args.race} — {args.driver} — P(box ≤ {args.hops} laps)"
        plot_probability(df, args.hops, title, save_path=str(png_path), threshold=args.threshold)
    else:
        # print a quick head if not saving
        print(df.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
