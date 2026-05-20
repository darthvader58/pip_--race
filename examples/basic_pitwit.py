from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pip_race import HpcTelemetryPacket, PitWit
from pip_race.data import frames_to_rows, summarize_frames
from pip_race.visualization import write_pit_risk_svg


packets = [
    HpcTelemetryPacket(car_id="ALB", lap=42, lap_distance_m=2300, speed_kph=188, tire_age_laps=27, compound="MEDIUM"),
    HpcTelemetryPacket(car_id="ALB", lap=42, lap_distance_m=2410, speed_kph=178, tire_age_laps=29, compound="MEDIUM", track_status="VSC"),
    HpcTelemetryPacket(car_id="SAR", lap=42, lap_distance_m=2390, speed_kph=181, tire_age_laps=18, compound="HARD"),
]

pitwit = PitWit()
frames = pitwit.process_many(packets)

print(summarize_frames(frames))
print(frames_to_rows(frames))
write_pit_risk_svg(frames, "pit_risk.svg")
