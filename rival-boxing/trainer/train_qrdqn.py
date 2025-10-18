#!/usr/bin/env python3
"""
QR-DQN trainer for "Will rival box within next 2 laps?"

Upgrades in this version:
- Reward shaping (+cheap-stop boost, softer false-positive penalty)
- Positive oversampling flag (--oversample_pos)
- Recall-biased thresholding (F2) + optional recall floor (--target_recall)
- Replay-dimension safe model init

Outputs:
  artifacts/rl/qrdqn.pt
  artifacts/rl/meta.json
  artifacts/rl/calib_platt.json
  artifacts/reports/rl_metrics.json
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, fbeta_score
)

import fastf1

print("QRDQN v1.2 — reward shaping + oversample + F2 threshold (replay-dim safe)")

# ----------------------------- FastF1 helpers -----------------------------

COMPOUNDS = ["SOFT","MED","HARD","INTERMEDIATE","WET"]

def enable_fastf1_cache(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

def load_race_session(spec: str):
    year_s, gp = spec.split(":")
    year = int(year_s)
    ses = fastf1.get_session(year, gp, "R")
    ses.load(laps=True, telemetry=False, weather=False)
    race_id = f"{year}_{gp}"
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

# ----------------------------- Featureization -----------------------------

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

# ----------------------------- Replay construction (2-step, shaped rewards) -----------------------------

def build_replay(
    base: pd.DataFrame,
    X: pd.DataFrame,
    feat_list: List[str],
    pos_reward: float = 2.0,
    neg_reward: float = -0.05,
    cheap_boost: float = 1.3
) -> Tuple[dict, dict]:
    """
    Tuples: (s_t, a_t, r_t, s_{t+2}, done)
    Reward shaping:
      +pos_reward (× cheap_boost if non-green) when action==BOX and pit within 2 laps
      neg_reward when action==BOX and no pit within 2 laps
      0 otherwise
    """
    tuples = []
    groups = []

    y_this = base["pitted_this_lap"].astype(int).values
    y_within2 = base["pitted_within2"].astype(int).values

    X_np = X[feat_list].to_numpy(dtype=np.float32)

    sorted_base = base.sort_values(["race_id","driver","lap"])
    for (rid, drv), g in sorted_base.groupby(["race_id","driver"], sort=False):
        idxs = g.index.to_list()
        n = len(idxs)
        for k in range(n):
            i = idxs[k]
            k2 = k+2
            if k2 >= n:
                done = True
                s_tp2 = np.zeros_like(X_np[i])
            else:
                done = False
                s_tp2 = X_np[idxs[k2]]

            s_t = X_np[i]
            a_t = 1 if y_this[i]==1 else 0
            will_pit_in2 = y_within2[i]==1

            cheap = int(base.loc[i, "cheap_stop_flag_true"]) if "cheap_stop_flag_true" in base.columns else 0
            r = 0.0
            if a_t==1 and will_pit_in2:
                r = pos_reward * (cheap_boost if cheap==1 else 1.0)
            elif a_t==1 and not will_pit_in2:
                r = neg_reward

            tuples.append((s_t, a_t, r, s_tp2, done))
            groups.append(rid)

    replay = {"tuples": tuples, "feat_list": feat_list}
    meta = {"groups": groups}
    return replay, meta

# ----------------------------- Model -----------------------------

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
        q = self.head(z)
        return q.view(-1, self.n_actions, self.n_quantiles)

def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0):
    u = target.unsqueeze(1) - pred.unsqueeze(2)  # (B,Q,Q)
    abs_u = torch.abs(u)
    huber = torch.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = torch.abs((u.detach() < 0).float() - tau)
    return (weight * huber).mean()

# ----------------------------- Data utils -----------------------------

class OfflineBuffer(Dataset):
    def __init__(self, tuples: List[tuple], oversample_pos: int = 8):
        self.base = tuples
        pos = [t for t in tuples if t[2] > 0]
        neg = [t for t in tuples if t[2] <= 0]
        self.items = neg + pos * max(1, oversample_pos)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        s, a, r, s2, done = self.items[idx]
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(s2, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

def choose_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ----------------------------- Training -----------------------------

def train_qrdqn(
    races: List[str],
    cache_dir: str,
    out_dir: str,
    n_quantiles: int = 101,
    gamma: float = 0.98,
    batch_size: int = 512,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden: int = 256,
    cql_alpha: float = 0.0,
    oversample_pos: int = 8,
    pos_reward: float = 2.0,
    neg_reward: float = -0.05,
    cheap_boost: float = 1.3,
    target_recall: float = 0.0,  # 0 -> disabled; else enforce recall floor when choosing threshold
    seed: int = 42,
):
    torch.manual_seed(seed); np.random.seed(seed)
    enable_fastf1_cache(cache_dir)

    # Build base rows
    all_rows=[]
    for spec in races:
        ses, rid, track, yr = load_race_session(spec)
        all_rows.append(build_car_lap_rows(ses, rid, track, yr))
    base = pd.concat(all_rows, ignore_index=True)

    # Features
    X, feat_list = build_state_matrix(base)

    # Replay w/ shaped rewards
    replay, meta = build_replay(
        base, X, feat_list,
        pos_reward=pos_reward, neg_reward=neg_reward, cheap_boost=cheap_boost
    )
    tuples = replay["tuples"]
    groups = np.array(meta["groups"])

    # Sanity on dims
    replay_dim = len(tuples[0][0])
    for j in (len(tuples)//3, 2*len(tuples)//3, len(tuples)-1):
        assert len(tuples[j][0]) == replay_dim
        assert len(tuples[j][3]) == replay_dim
    print(f"[QRDQN] feat_list={len(feat_list)} ; replay_dim={replay_dim}")

    # Split by race (last race as val)
    uniq = np.unique(groups)
    if len(uniq) >= 2:
        val_race = uniq[-1]
        train_idx = [i for i,g in enumerate(groups) if g != val_race]
        val_idx   = [i for i,g in enumerate(groups) if g == val_race]
    else:
        N = len(tuples); perm=np.random.permutation(N); cut=int(0.8*N)
        train_idx, val_idx = perm[:cut], perm[cut:]

    train_tuples = [tuples[i] for i in train_idx]
    val_tuples   = [tuples[i] for i in val_idx]

    train_ds = OfflineBuffer(train_tuples, oversample_pos=oversample_pos)
    val_ds   = OfflineBuffer(val_tuples,   oversample_pos=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # Model
    device = choose_device()
    in_dim = replay_dim
    net = QRDQN(in_dim, 2, n_quantiles, hidden).to(device)
    tgt = QRDQN(in_dim, 2, n_quantiles, hidden).to(device)
    tgt.load_state_dict(net.state_dict())

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    taus = torch.tensor([(i+0.5)/n_quantiles for i in range(n_quantiles)], dtype=torch.float32, device=device)

    target_tau = 200
    step = 0

    # Train
    for ep in range(epochs):
        net.train()
        for (s, a, r, s2, done) in train_loader:
            s = s.to(device); a=a.to(device); r=r.to(device); s2=s2.to(device); done=done.to(device)
            B = s.size(0)

            with torch.no_grad():
                qz_next_online = net(s2)
                exp_next = qz_next_online.mean(dim=-1)
                a_star = exp_next.argmax(dim=1)

                qz_next_target = tgt(s2)
                target_z = qz_next_target[torch.arange(B), a_star]
                Tz = r.unsqueeze(1) + (1.0 - done.unsqueeze(1)) * (gamma**2) * target_z

            qz = net(s)
            qz_a = qz[torch.arange(B), a]
            loss = quantile_huber_loss(qz_a, Tz, taus, kappa=1.0)

            if cql_alpha > 0.0:
                exp_all = qz.mean(dim=-1)
                logsum = torch.logsumexp(exp_all, dim=1)
                q_beh = exp_all[torch.arange(B), a]
                loss = loss + cql_alpha * (logsum - q_beh).mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            step += 1
            if step % target_tau == 0:
                tgt.load_state_dict(net.state_dict())

        # quick directional val (AUC/AP on reward-proxy labels)
        net.eval()
        with torch.no_grad():
            val_states = np.stack([t[0] for t in val_tuples], axis=0)
            s_tensor = torch.tensor(val_states, dtype=torch.float32, device=device)
            qz = net(s_tensor).cpu().numpy()
            exp = qz.mean(axis=-1)
            scores = exp[:,1] - exp[:,0]
            val_labels = np.array([1.0 if (t[2] > 0.5) else 0.0 for t in val_tuples], dtype=float)
            try:
                auc = roc_auc_score(val_labels, scores)
                ap  = average_precision_score(val_labels, scores)
            except Exception:
                auc, ap = float("nan"), float("nan")
        print(f"[epoch {ep+1}/{epochs}] val AUC={auc:.3f} AP={ap:.3f}")

    # Save model + meta
    rl_dir = Path(out_dir) / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": in_dim, "n_actions": 2,
        "n_quantiles": n_quantiles, "hidden": hidden, "gamma": gamma,
        "feat_list": feat_list,
    }, rl_dir / "qrdqn.pt")

    (rl_dir / "meta.json").write_text(json.dumps({
        "feat_list": feat_list,
        "n_quantiles": n_quantiles,
        "gamma": gamma,
        "hidden": hidden,
        "actions": ["NO_BOX","BOX"],
        "note": "Expectation gap (BOX - NO_BOX) -> Platt -> prob",
    }, indent=2))

    # Platt calibration
    with torch.no_grad():
        ss = torch.tensor(np.stack([t[0] for t in val_tuples], axis=0), dtype=torch.float32, device=device)
        qz = net(ss).cpu().numpy()
        exp = qz.mean(axis=-1)
        scores = exp[:,1] - exp[:,0]
    val_labels = np.array([1.0 if (t[2] > 0.5) else 0.0 for t in val_tuples], dtype=float)

    lr_platt = LogisticRegression(max_iter=500, class_weight="balanced")
    lr_platt.fit(scores.reshape(-1,1), val_labels)
    calib = {"coef": float(lr_platt.coef_[0,0]), "intercept": float(lr_platt.intercept_[0])}
    (rl_dir / "calib_platt.json").write_text(json.dumps(calib, indent=2))

    # Threshold selection — F2 priority, optional recall floor
    probs = 1.0 / (1.0 + np.exp(-(calib["coef"]*scores + calib["intercept"])))

    best = {"th": 0.5, "f2": -1.0, "pr": 0.0, "rc": 0.0, "f1": 0.0}
    grid = np.linspace(0.2, 0.8, 121)

    # If recall floor requested, pre-filter candidates
    if target_recall > 0.0:
        candidates = []
        for th in grid:
            pred = (probs >= th).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(val_labels, pred, average="binary", zero_division=0)
            if rc >= target_recall:
                candidates.append((th, pr, rc, f1, fbeta_score(val_labels, pred, beta=2.0, average="binary", zero_division=0)))
        if candidates:
            # choose best precision among those meeting recall floor
            th, pr, rc, f1, f2 = max(candidates, key=lambda x: (x[1], x[4]))
            best = {"th": th, "f2": f2, "pr": pr, "rc": rc, "f1": f1}
        else:
            # fall back to plain F2 maximization
            for th in grid:
                pred = (probs >= th).astype(int)
                f2 = fbeta_score(val_labels, pred, beta=2.0, average="binary", zero_division=0)
                if f2 > best["f2"]:
                    pr, rc, f1, _ = precision_recall_fscore_support(val_labels, pred, average="binary", zero_division=0)
                    best = {"th": th, "f2": f2, "pr": pr, "rc": rc, "f1": f1}
    else:
        for th in grid:
            pred = (probs >= th).astype(int)
            f2 = fbeta_score(val_labels, pred, beta=2.0, average="binary", zero_division=0)
            if f2 > best["f2"]:
                pr, rc, f1, _ = precision_recall_fscore_support(val_labels, pred, average="binary", zero_division=0)
                best = {"th": th, "f2": f2, "pr": pr, "rc": rc, "f1": f1}

    reports = Path(out_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metrics = {
        "val_auc_proxy": float(roc_auc_score(val_labels, scores)) if len(np.unique(val_labels))>1 else float("nan"),
        "val_ap_proxy":  float(average_precision_score(val_labels, scores)) if len(np.unique(val_labels))>1 else float("nan"),
        "platt_best_threshold": float(best["th"]),
        "platt_best_precision": float(best["pr"]),
        "platt_best_recall": float(best["rc"]),
        "platt_best_f1": float(best["f1"]),
        "platt_best_f2": float(best["f2"]),
        "n_val": int(len(val_labels)),
        "pos_rate_proxy": float(np.mean(val_labels)),
        "replay_dim": int(replay_dim),
        "feat_cols": int(len(feat_list)),
        "oversample_pos": int(oversample_pos),
        "pos_reward": float(pos_reward),
        "neg_reward": float(neg_reward),
        "cheap_boost": float(cheap_boost),
        "target_recall": float(target_recall),
    }
    (reports / "rl_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("== QR-DQN training complete ==")
    print("Saved:", str(rl_dir / 'qrdqn.pt'), str(rl_dir / 'meta.json'), str(rl_dir / 'calib_platt.json'))
    print("Val (proxy) metrics:", json.dumps(metrics, indent=2))

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", required=True, help="Comma-separated list like 2023:Monaco,2023:Monza")
    ap.add_argument("--cache", default="data/fastf1_cache")
    ap.add_argument("--out",   default="artifacts")
    # model/train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--n_quantiles", type=int, default=101)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cql_alpha", type=float, default=0.0)
    # imbalance knobs
    ap.add_argument("--oversample_pos", type=int, default=8)
    ap.add_argument("--pos_reward", type=float, default=2.0)
    ap.add_argument("--neg_reward", type=float, default=-0.05)
    ap.add_argument("--cheap_boost", type=float, default=1.3)
    # thresholding
    ap.add_argument("--target_recall", type=float, default=0.0, help="Set >0 (e.g., 0.8) to enforce a recall floor when picking threshold")
    args = ap.parse_args()

    races = [s.strip() for s in args.races.split(",") if s.strip()]
    train_qrdqn(
        races=races,
        cache_dir=args.cache,
        out_dir=args.out,
        n_quantiles=args.n_quantiles,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        cql_alpha=args.cql_alpha,
        oversample_pos=args.oversample_pos,
        pos_reward=args.pos_reward,
        neg_reward=args.neg_reward,
        cheap_boost=args.cheap_boost,
        target_recall=args.target_recall,
    )

if __name__ == "__main__":
    main()
