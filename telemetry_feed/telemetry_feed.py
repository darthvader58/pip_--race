import fastf1
import asyncio
import json
import websockets
import os

# Enable FastF1 caching
fastf1.Cache.enable_cache('fastf1_cache')

async def stream_telemetry(ws, driver="VER", year=2023, gp="Monaco", session_type="R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    laps = session.laps.pick_driver(driver)
    print(f"Loaded {len(laps)} laps for {driver}")

    for _, lap in laps.iterrows():
        tel = lap.get_car_data().add_distance()
        for _, row in tel.iterrows():
            payload = {
                "lap_distance_m": float(row["Distance"]),
                "speed_kph": float(row["Speed"])
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
