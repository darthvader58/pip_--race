#!/usr/bin/env python3
import argparse, json, time, sys
from pathlib import Path
from collections import defaultdict, deque

import requests
import numpy as np
import pandas as pd
import fastf1

# --------------- helpers ----------------

def nz(x, default=0.0):
    try:
        if x is None:
            return float(default)
        if hasattr(x, "total_seconds"):
            x = x.total_seconds()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def load_feat_list(meta_path):
    meta = json.loads(Path(meta_path).read_text())
    fl = meta.get("feat_list", [])
    if not fl or not isinstance(fl, list):
        print(f"[ERR] meta.feat_list missing/empty in {meta_path}", file=sys.stderr)
        sys.exit(1)
    return fl

def compound_flags(raw):
    # Normalise text
    c = (raw or "").upper()
    # FastF1 typically uses 'SOFT','MEDIUM','HARD','INTERMEDIATE','WET'
    # Your feats want 'MED' not 'MEDIUM'
    is_soft = 1 if c == "SOFT" else 0
    is_med  = 1 if c in ("MED", "MEDIUM") else 0
    is_hard = 1 if c == "HARD" else 0
    is_int  = 1 if c in ("INTERMEDIATE","INT") else 0
    is_wet  = 1 if c == "WET" else 0
    return is_soft, is_med, is_hard, is_int, is_wet

def build_track_status_timeline(session):
    """
    Returns a DataFrame with columns: ['Time','StatusCode','is_green']
    StatusCode is int if possible; else 1 for green by default.
    """
    try:
        ts = session.api.track_status_data  # cached when session.load() ran
        # Columns: Time, Status, Message (varies by FastF1 version)
        df = ts.copy()
        # Status can be strings "1","2",... We convert best-effort to int.
        def to_code(s):
            try:
                return int(str(s).strip())
            except Exception:
                return 1
        df["StatusCode"] = df["Status"].map(to_code)
        # Heuristic: 1=green, others=not green.
        df["is_green"] = (df["StatusCode"] == 1).astype(int)
        # Keep only what we need
        return df[["Time","StatusCode","is_green"]].sort_values("Time").reset_index(drop=True)
    except Exception:
        # Fallback: always green
        return pd.DataFrame({"Time":[0.0], "StatusCode":[1], "is_green":[1]})

def status_at_time(ts_df, t_seconds):
    # Find last status change at or before t_seconds
    if ts_df is None or ts_df.empty:
        return 1, 1
    # ts_df["Time"] is Timedelta in many cases
    if hasattr(ts_df["Time"].iloc[0], "total_seconds"):
        tvals = ts_df["Time"].apply(lambda x: x.total_seconds())
    else:
        tvals = ts_df["Time"].astype(float)
    idx = tvals.searchsorted(t_seconds, side="right") - 1
    idx = max(0, min(idx, len(ts_df)-1))
    sc = int(ts_df["StatusCode"].iloc[idx])
    green = int(ts_df["is_green"].iloc[idx])
    return sc, green

def typical_stint_len_by_comp(laps_df):
    """
    Compute a typical (median) stint length per compound in this race,
    using laps where PitIn==True as stint ends.
    """
    comp_med = {}
    if "PitIn" in laps_df.columns and "TyreLife" in laps_df.columns and "Compound" in laps_df.columns:
        # pick rows that end a stint (pit in) and have TyreLife > 0
        enders = laps_df[(laps_df["PitIn"] == True) & (~laps_df["TyreLife"].isna()) & (laps_df["TyreLife"] > 0)]
        if len(enders):
            grp = enders.groupby(enders["Compound"].astype(str).str.upper())["TyreLife"].median()
            comp_med = {k: float(v) for k, v in grp.to_dict().items()}
    # sane defaults if missing
    defaults = {"SOFT": 15.0, "MEDIUM": 22.0, "MED": 22.0, "HARD": 30.0, "INTERMEDIATE": 35.0, "WET": 45.0}
    def lookup(c):
        u = (c or "").upper()
        if u in comp_med:
            return comp_med[u]
        if u == "MED":
            return comp_med.get("MEDIUM", defaults["MED"])
        return defaults.get(u, 20.0)
    return lookup

# --------------- feature builder (EXACT names) ----------------

def compute_features_for_row(row, prev_row, ts_df, driver_state, typical_len_fn):
    """
    Returns a dict with exactly the keys in your feat_list.
    Uses row (current lap), prev_row (previous lap of same driver),
    track-status timeline, rolling driver_state (keeps last laps), and
    typical_len_fn(compound)->typical_len.
    """
    lap_no = int(row.get("LapNumber", 0))
    stint_no = int(row.get("Stint", 0))
    tire_age = int(row.get("TyreLife", 0))
    comp_raw = row.get("Compound", None)
    comp_soft, comp_med, comp_hard, comp_int, comp_wet = compound_flags(comp_raw)

    # lap times & rolling stats
    lt = nz(row.get("LapTime"))
    s1 = nz(row.get("Sector1Time"))
    s2 = nz(row.get("Sector2Time"))
    s3 = nz(row.get("Sector3Time"))
    t_sess = nz(row.get("SessionTime"))

    # update driver rolling buffers
    lbuf = driver_state["last_laps"]
    lbuf.append(lt)
    if len(lbuf) > 10:
        lbuf.popleft()

    # last3_avg, last3_var
    if len(lbuf) >= 3:
        arr3 = np.array(list(lbuf)[-3:], dtype=float)
        last3_avg = float(np.nanmean(arr3))
        last3_var = float(np.nanvar(arr3))
    else:
        last3_avg = lt if lt > 0 else 0.0
        last3_var = 0.0

    # last5_slope: simple linear regression slope over last up-to-5 points
    if len(lbuf) >= 2:
        tail = list(lbuf)[-5:]
        xs = np.arange(len(tail), dtype=float)
        ys = np.array(tail, dtype=float)
        # if any NaNs, fallback 0
        if np.any(~np.isfinite(ys)):
            last5_slope = 0.0
        else:
            # slope = cov(x,y)/var(x)
            vx = np.var(xs)
            if vx == 0:
                last5_slope = 0.0
            else:
                last5_slope = float(np.cov(xs, ys, bias=True)[0,1] / vx)
    else:
        last5_slope = 0.0

    # typical stint length & derived
    typical_len = float(typical_len_fn(comp_raw))
    age_vs_typical = float(tire_age - typical_len)
    # "percentile": crude: age / typical_len (clipped 0..2) then rescale to 0..1 by /2
    age_percentile = float(np.clip(tire_age / (typical_len + 1e-6), 0.0, 2.0) / 2.0)
    overshoot = float(max(0.0, age_vs_typical))

    # Track status & cheap stop flags
    scode, green = status_at_time(ts_df, t_sess)
    cheap_flag = 0 if green == 1 else 1

    # previous cheap flags (per driver)
    cheap_prev1 = driver_state["cheap_prev1"]
    cheap_prev2 = driver_state["cheap_prev2"]
    driver_state["cheap_prev2"] = cheap_prev1
    driver_state["cheap_prev1"] = cheap_flag

    # non_green_runlen: consecutive laps with !green (kept in state)
    if cheap_flag == 1:
        driver_state["non_green_runlen"] += 1
    else:
        driver_state["non_green_runlen"] = 0
    non_green_runlen = driver_state["non_green_runlen"]

    # previous pit flags
    pit_in = 1 if row.get("PitIn", False) else 0
    pits_prev1 = driver_state["pits_prev1"]
    pits_prev2 = driver_state["pits_prev2"]
    driver_state["pits_prev2"] = pits_prev1
    driver_state["pits_prev1"] = pit_in

    # Assemble EXACT feature names (26), including duplicates at the end
    feats = {
        "tire_age_laps": float(tire_age),
        "stint_no": float(stint_no),
        "compound_SOFT": float(comp_soft),
        "compound_MED": float(comp_med),
        "compound_HARD": float(comp_hard),
        "compound_INTERMEDIATE": float(comp_int),
        "compound_WET": float(comp_wet),
        "last3_avg": float(last3_avg),
        "last5_slope": float(last5_slope),
        "last3_var": float(last3_var),
        "typical_stint_len": float(typical_len),
        "age_vs_typical": float(age_vs_typical),
        "age_percentile": float(age_percentile),
        "overshoot": float(overshoot),
        "cheap_stop_flag": float(cheap_flag),
        "cheap_prev1": float(cheap_prev1),
        "cheap_prev2": float(cheap_prev2),
        "non_green_runlen": float(non_green_runlen),
        "pits_prev1": float(pits_prev1),
        "pits_prev2": float(pits_prev2),
        # duplicates:
        "tire_age_laps": float(tire_age),
        "compound_SOFT": float(comp_soft),
        "compound_MED": float(comp_med),
        "compound_HARD": float(comp_hard),
        "compound_INTERMEDIATE": float(comp_int),
        "compound_WET": float(comp_wet),
    }

    return feats

# --------------- FastF1 access ----------------

def load_fastf1_session(race_spec, cache_dir):
    year_s, gp_name = race_spec.split(":", 1)
    year = int(year_s)
    fastf1.Cache.enable_cache(cache_dir)
    ses = fastf1.get_session(year, gp_name, "R")
    ses.load(telemetry=False, weather=False, messages=False)
    return ses

def iter_laps_stream(session, only_codes=None):
    laps = session.laps
    laps = laps[(~laps["LapTime"].isna()) & (~laps["Driver"].isna())]
    if only_codes:
        only = set([x.strip().upper() for x in only_codes.split(",") if x.strip()])
        laps = laps[laps["Driver"].str.upper().isin(only)]
    laps = laps.sort_values(by=["LapStartTime", "Driver"])
    return laps

# --------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help='e.g. "2023:Monaco"')
    ap.add_argument("--cache", default="data/fastf1_cache")
    ap.add_argument("--url", default="http://localhost:8080/ingest")
    ap.add_argument("--meta", required=True, help="artifacts/rl/meta.json")
    ap.add_argument("--sleep", type=float, default=0.10)
    ap.add_argument("--only", default="", help="CSV of 3-letter driver codes")
    ap.add_argument("--echo", action="store_true")
    # NEW: Bridge service for WebSocket broadcasting to frontend
    ap.add_argument("--bridge", default="http://localhost:8081/update", 
                    help="Bridge service URL for WebSocket broadcasting")
    args = ap.parse_args()

    feat_list = load_feat_list(args.meta)
    print(f"Loaded feat_list[{len(feat_list)}] from {args.meta}")

    print(f"Loading FastF1 {args.race} (Race) from cache: {args.cache}")
    ses = load_fastf1_session(args.race, args.cache)
    status_df = build_track_status_timeline(ses)
    laps = iter_laps_stream(ses, args.only)
    drivers = sorted(list(set(laps["Driver"].str.upper().values.tolist())))
    print("Drivers found:", drivers)

    # Typical stint length per compound (median at pit-in)
    typical_len_fn = typical_stint_len_by_comp(laps)

    # Per-driver state
    state_by_driver = defaultdict(lambda: {
        "last_laps": deque(maxlen=10),
        "cheap_prev1": 0,
        "cheap_prev2": 0,
        "non_green_runlen": 0,
        "pits_prev1": 0,
        "pits_prev2": 0,
        "prev_row": None,
    })

    posted = 0
    bridge_failures = 0
    
    for _, lap in laps.iterrows():
        drv = str(lap["Driver"]).upper()
        lapnum = int(lap["LapNumber"])

        row = {
            "LapNumber": lapnum,
            "Stint": int(lap.get("Stint", 0)),
            "TyreLife": int(nz(lap.get("TyreLife", 0))),
            "Compound": lap.get("Compound", None),
            "PitIn": bool(lap.get("PitIn", False)),
            "PitOut": bool(lap.get("PitOut", False)),
            "LapTime": nz(lap["LapTime"]),
            "Sector1Time": nz(lap.get("Sector1Time", 0.0)),
            "Sector2Time": nz(lap.get("Sector2Time", 0.0)),
            "Sector3Time": nz(lap.get("Sector3Time", 0.0)),
            "SessionTime": nz(lap.get("Time", 0.0)),
        }

        drv_state = state_by_driver[drv]
        feats = compute_features_for_row(row, drv_state["prev_row"], status_df, drv_state, typical_len_fn)

        # sanity: check coverage and variability
        matched = sum(1 for k in feat_list if k in feats)
        if matched != len(feat_list):
            missing = [k for k in feat_list if k not in feats]
            print(f"[WRN] {drv} L{lapnum} missing {len(missing)} feats: {missing[:4]}...", flush=True)

        # form flat payload: exactly the feature keys
        payload = {"driver": drv, "lap": lapnum}
        for k in feat_list:  # keep explicit order on server side
            payload[k] = nz(feats.get(k, 0.0))

        try:
            # Send to rt_predictor for inference
            r = requests.post(args.url, json=payload, timeout=2.0)
            if r.status_code != 200:
                print(f"[WARN] POST {r.status_code}: {r.text[:200]}", flush=True)
            else:
                out = r.json()
                if args.echo:
                    p2 = out.get("p2", None)
                    p3 = out.get("p3", None)
                    print(f"[OK] {drv} L {lapnum:2d}: p2={p2:.3f} p3={p3:.3f}", flush=True)
                
                # NEW: Send to bridge service for WebSocket broadcasting to frontend
                try:
                    bridge_payload = {
                        'driver': drv,
                        'lap': lapnum,
                        'p2': out.get('p2', 0.0),
                        'p3': out.get('p3', 0.0),
                        't': out.get('t', int(time.time() * 1000))
                    }
                    bridge_r = requests.post(args.bridge, json=bridge_payload, timeout=1.0)
                    if bridge_r.status_code != 200:
                        bridge_failures += 1
                        if bridge_failures % 50 == 0:
                            print(f"[WARN] Bridge POST failed: {bridge_r.status_code} (total failures: {bridge_failures})", flush=True)
                except requests.exceptions.RequestException as bridge_e:
                    # Don't fail the main flow if bridge is down
                    bridge_failures += 1
                    if bridge_failures == 1:
                        print(f"[WARN] Bridge service unavailable at {args.bridge}. Predictions won't reach frontend.", flush=True)
                        print(f"       Make sure bridge service is running: cd bridge-service && npm start", flush=True)
                    elif bridge_failures % 100 == 0:
                        print(f"[WARN] Bridge connection issues (failures: {bridge_failures})", flush=True)
                    
        except Exception as e:
            print(f"[ERR] POST to rt_predictor failed: {e}", flush=True)

        drv_state["prev_row"] = row
        posted += 1
        if posted % 50 == 0:
            print(f"Posted {posted} packets... (bridge failures: {bridge_failures})", flush=True)

        time.sleep(args.sleep if args.sleep >= 0 else 0.0)

    print(f"\n=== Summary ===")
    print(f"Total packets posted to rt_predictor: {posted}")
    if bridge_failures > 0:
        print(f"Bridge service failures: {bridge_failures}")
        print(f"Note: Start bridge service with: cd bridge-service && npm start")
    else:
        print(f"Bridge service: All packets sent successfully ✓")

if __name__ == "__main__":
    main()