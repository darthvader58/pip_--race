use serde::Deserialize;
use std::fs;

#[derive(Deserialize, Debug)]
pub struct TimerConfig {
    pub pit_entry_m: f64,
    pub call_offset_m: f64,
    pub buffer_s: f64,
}

impl TimerConfig {
    pub fn load(path: &str) -> Self {
        let data = fs::read_to_string(path).expect("Config file not found");
        serde_json::from_str(&data).expect("Invalid config JSON")
    }
}
