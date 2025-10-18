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

    for _, lap in laps.iterrows():
        tel = lap.get_car_data().add_distance()
        for idx, row in tel.iterrows():
            lap_distance_m = float(row["Distance"])
            speed_kph = float(row["Speed"])
            
            # Add current sample to the sliding window
            profile_calc.add_sample(lap_distance_m, speed_kph)
            
            # Generate speed profile looking ahead from current position
            # This will be used by Rust backend for integration-based time estimation
            speed_profile = profile_calc.get_lookahead_profile(lap_distance_m)
            
            payload = {
                "lap_distance_m": lap_distance_m,
                "speed_kph": speed_kph,
                "speed_profile": speed_profile  # Will be None initially, then a list of {x_m, v_mps}
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.1)  # 10 Hz stream
    print("Telemetry stream complete.")

async def main():
    backend_url = os.getenv("BACKEND_URL", "ws://rust-backend:8765")
    print(f"Connecting to backend at {backend_url}")
    async with websockets.connect(backend_url) as ws:
        await stream_telemetry(ws)

if __name__ == "__main__":
    asyncio.run(main())
