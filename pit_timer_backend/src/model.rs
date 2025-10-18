use serde::Deserialize;
use crate::config::TimerConfig;

#[derive(Deserialize, Debug)]
pub struct TelemetryPacket {
    pub lap_distance_m: f64,
    pub speed_kph: f64,
}

pub fn time_to_call(d: &TelemetryPacket, cfg: &TimerConfig) -> (f64, f64, &'static str) {
    let call_at_m = cfg.pit_entry_m - cfg.call_offset_m;
    let d_rem = (call_at_m - d.lap_distance_m).max(0.0);
    let v_mps = (d.speed_kph / 3.6).max(1.0);
    let t_call = d_rem / v_mps;
    let t_safe = t_call - cfg.buffer_s;

    let status = if t_safe < 0.0 {
        "LOCKED_OUT"
    } else if t_safe < 2.0 {
        "RED"
    } else if t_safe < 5.0 {
        "AMBER"
    } else {
        "GREEN"
    };

    (t_call, t_safe, status)
}
