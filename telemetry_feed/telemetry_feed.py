import fastf1
import asyncio
import json
import websockets
import os
from speed_profile_calculator import SpeedProfileCalculator

# Enable FastF1 caching
fastf1.Cache.enable_cache('fastf1_cache')

async def stream_telemetry(ws, team="Williams", year=2023, gp="Monaco", session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    # Filter laps by team
    laps = session.laps[session.laps['Team'] == team]
    
    if laps.empty:
        print(f"No laps found for team {team} in {gp} {year}")
        return

    print(f"Loaded {len(laps)} laps for team {team}")

    # Initialize speed profile calculator with a 50-sample sliding window
    # and 500m lookahead (adjust these based on your track/needs)
    profile_calc = SpeedProfileCalculator(window_size=50, lookahead_m=500.0)

    # Flatten all telemetry rows into a single stream
    telemetry_rows = []
    for _, lap in laps.iterrows():
        tel = lap.get_car_data().add_distance()
        for idx, row in tel.iterrows():
            telemetry_rows.append(row)

    print(f"Streaming {len(telemetry_rows)} telemetry rows at 5Hz...")
    for row in telemetry_rows:
        lap_distance_m = float(row["Distance"])
        speed_kph = float(row["Speed"])
        profile_calc.add_sample(lap_distance_m, speed_kph)
        speed_profile = profile_calc.get_lookahead_profile(lap_distance_m)
        payload = {
            "lap_distance_m": lap_distance_m,
            "speed_kph": speed_kph,
            "speed_profile": speed_profile
        }
        await ws.send(json.dumps(payload))
        await asyncio.sleep(0.2)  # 5 Hz stream
    print("Telemetry stream complete.")

async def main():
    backend_url = os.getenv("BACKEND_URL", "ws://rust-backend:8765")
    print(f"Connecting to backend at {backend_url}")

    # Reconnect loop: if connection drops or handshake fails, retry with a short backoff
    while True:
        try:
            async with websockets.connect(
                backend_url,
                ping_interval=20,   # seconds between keepalive pings
                ping_timeout=20,    # wait this long for a pong
                close_timeout=5,    # graceful close
                max_queue=None      # unbounded recv queue to avoid backpressure disconnects
            ) as ws:
                print("WebSocket connected. Streaming telemetry...")
                await stream_telemetry(ws)
                print("Stream finished, closing connection.")
                break  # Completed streaming successfully
        except Exception as e:
            print(f"WebSocket error: {e}. Reconnecting in 1s...")
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(main())
