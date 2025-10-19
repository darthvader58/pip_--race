import argparse, json, time, requests, datetime as dt
from pathlib import Path

# Example: load your per-timestamp predictions you already compute
# Or just call your in-memory model and format the result here.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8080/publish")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between posts")
    ap.add_argument("--demo", action="store_true", help="send a few dummy frames")
    args = ap.parse_args()

    if args.demo:
        frames = [
            {"1":0.05,"11":0.08,"16":0.10,"55":0.06},
            {"1":0.06,"11":0.11,"16":0.12,"55":0.07},
            {"1":0.08,"11":0.15,"16":0.16,"55":0.09},
        ]
    else:
        raise SystemExit("Plug your model outputs here or use --demo.")

    for i, probs in enumerate(frames, start=1):
        payload = {
            "race_id": args.race,
            "ts": dt.datetime.utcnow().isoformat()+"Z",
            "probs": probs,
            "meta": {"lap": i, "hops": 2}
        }
        r = requests.post(args.endpoint, json=payload, timeout=2)
        r.raise_for_status()
        print("posted", i, r.status_code)
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
