use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct TelemetryPacket {
    pub driver: String,
    pub lap: i32,
    // fields we use to rebuild features (extend as you add more features):
    pub compound: Option<String>,        // e.g., "C1","C2","C3","C4","C5","I","W"
    pub tyre_laps: Option<i32>,          // age in laps
    pub gap_front: Option<f32>,          // to car ahead (s)
    pub track_status_code: Option<i32>,  // 1=green; others non-green
    pub pit_window_lap: Option<i32>,     // nominal planned pit window (lap)
    pub pitted_this_lap: Option<bool>,   // optional ground truth marker (for debug)
}

#[derive(Debug, Serialize, Clone)]
pub struct PredictionOut {
    pub driver: String,
    pub lap: i32,
    pub prob_box_within2: f32,
    pub prob_box_within3: f32,
    pub ts_ms: i64,
}