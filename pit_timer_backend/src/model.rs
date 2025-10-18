use serde::Deserialize;
use crate::config::TimerConfig;

/// Optional speed sample for integrating time-to-call over a short distance.
/// x_m is the lap distance in meters; v_mps is instantaneous speed in m/s at that x.
#[derive(Deserialize, Debug, Clone, Copy)]
pub struct SpeedSample {
    pub x_m: f64,
    pub v_mps: f64,
}

#[derive(Deserialize, Debug)]
pub struct TelemetryPacket {
    pub lap_distance_m: f64,
    pub speed_kph: f64,
    /// Optional recent speed profile vs. distance, used to integrate 1/v(x) dx
    /// for a better estimate of time-to-call. If absent, we fall back to
    /// instantaneous speed t_call ≈ d_rem / v.
    pub speed_profile: Option<Vec<SpeedSample>>,
}

pub fn time_to_call(d: &TelemetryPacket, cfg: &TimerConfig) -> (f64, f64, &'static str) {
    // Core math per spec:
    // d_rem = max((pit_entry_m - call_offset_m) - lap_distance_m, 0)
    let call_at_m = cfg.pit_entry_m - cfg.call_offset_m;
    let start_x = d.lap_distance_m;
    let end_x = call_at_m;
    let d_rem = (end_x - start_x).max(0.0);

    // Prefer integrated time over a provided speed profile (trapezoidal rule),
    // otherwise fall back to instantaneous speed estimate.
    let v_inst_mps = (d.speed_kph / 3.6).max(0.1); // avoid div-by-zero; tiny epsilon
    let t_call = integrate_time_over_profile(start_x, end_x, d.speed_profile.as_deref(), v_inst_mps)
        .unwrap_or_else(|| d_rem / v_inst_mps);

    // Latest safe radio moment: t_safe = t_call - buffer_s
    let t_safe = t_call - cfg.buffer_s;

    // let status = if t_safe < 0.0 {
    //     "LOCKED_OUT"
    // } else if t_safe < 2.0 {
    //     "RED"
    // } else if t_safe < 5.0 {
    //     "AMBER"
    // } else {
    //     "GREEN"
    // };
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

/// Trapezoidal integration of dt = ∫ (1 / v(x)) dx from start_x to end_x.
/// Returns None if there are no usable samples in-range; caller should fall back.
fn integrate_time_over_profile(
    start_x: f64,
    end_x: f64,
    profile: Option<&[SpeedSample]>,
    fallback_v_mps: f64,
) -> Option<f64> {
    let Some(samples) = profile else { return None };
    if end_x <= start_x {
        return Some(0.0);
    }

    // Collect in-range samples [start_x, end_x] and include start/end guards
    let mut pts: Vec<(f64, f64)> = samples
        .iter()
        .filter_map(|s| {
            if s.x_m >= start_x && s.x_m <= end_x {
                Some((s.x_m, s.v_mps))
            } else {
                None
            }
        })
        .collect();

    if pts.is_empty() {
        // No in-range samples; let caller fall back to instantaneous
        return None;
    }

    // Add start and end points using fallbacks if not already present
    if pts.first().map(|(x, _)| *x) > Some(start_x) {
        pts.insert(0, (start_x, fallback_v_mps));
    } else if pts.first().map(|(x, _)| *x) != Some(start_x) {
        // Ensure exact start point exists for stable integration
        pts.insert(0, (start_x, pts.first().map(|(_, v)| *v).unwrap_or(fallback_v_mps)));
    }
    if pts.last().map(|(x, _)| *x) < Some(end_x) {
        pts.push((end_x, fallback_v_mps));
    } else if pts.last().map(|(x, _)| *x) != Some(end_x) {
        let last_v = pts.last().map(|(_, v)| *v).unwrap_or(fallback_v_mps);
        pts.push((end_x, last_v));
    }

    // Sort by x, deduplicate any identical x keeping the last value
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup_by(|a, b| a.0 == b.0);

    // Trapezoidal integrate of 1/v(x)
    let eps = 0.1_f64; // minimum speed to avoid blow-ups
    let mut area = 0.0;
    for w in pts.windows(2) {
        let (x0, v0) = w[0];
        let (x1, v1) = w[1];
        if x1 <= x0 { continue; }
        let v0 = v0.max(eps);
        let v1 = v1.max(eps);
        let inv0 = 1.0 / v0;
        let inv1 = 1.0 / v1;
        area += (x1 - x0) * 0.5 * (inv0 + inv1);
    }

    Some(area)
}
