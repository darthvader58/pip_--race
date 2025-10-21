use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedSample {
    /// Distance along lap in meters
    pub x_m: f64,
    /// Speed in meters per second
    pub v_mps: f64,
}

/// Maintains a sliding window of recent telemetry and generates
/// speed profiles for integration-based time estimation.
pub struct SpeedProfileCalculator {
    /// Number of recent samples to keep in the sliding window
    window_size: usize,
    /// Distance ahead (meters) to include in the profile for integration
    lookahead_m: f64,
    /// Sliding window of recent telemetry samples
    window: VecDeque<SpeedSample>,
}

impl SpeedProfileCalculator {
    /// Create a new speed profile calculator.
    ///
    /// # Arguments
    /// * `window_size` - Number of recent samples to keep in the sliding window
    /// * `lookahead_m` - Distance ahead (meters) to include in the profile for integration
    pub fn new(window_size: usize, lookahead_m: f64) -> Self {
        Self {
            window_size,
            lookahead_m,
            window: VecDeque::with_capacity(window_size),
        }
    }

    /// Add a new telemetry sample to the sliding window.
    ///
    /// # Arguments
    /// * `lap_distance_m` - Current lap distance in meters
    /// * `speed_kph` - Current speed in km/h
    pub fn add_sample(&mut self, lap_distance_m: f64, speed_kph: f64) {
        let speed_mps = speed_kph / 3.6; // Convert to m/s
        
        let sample = SpeedSample {
            x_m: lap_distance_m,
            v_mps: speed_mps,
        };

        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(sample);
    }

    /// Generate a speed profile from current position toward target.
    ///
    /// Returns samples in the range [current_distance_m, target_distance_m]
    /// or slightly beyond if we have historical data that covers that range.
    ///
    /// # Arguments
    /// * `current_distance_m` - Current lap distance in meters
    /// * `target_distance_m` - Target call point distance in meters
    ///
    /// # Returns
    /// `Some(Vec<SpeedSample>)` if sufficient data exists, `None` otherwise
    pub fn get_profile(&self, current_distance_m: f64, target_distance_m: f64) -> Option<Vec<SpeedSample>> {
        if self.window.len() < 2 {
            return None;
        }

        // Filter samples in the range [current_distance, target_distance]
        // Include a small buffer before/after for smoother integration
        const BUFFER_M: f64 = 50.0;
        let min_x = current_distance_m - BUFFER_M;
        let max_x = target_distance_m + BUFFER_M;

        let mut profile: Vec<SpeedSample> = self.window
            .iter()
            .filter(|sample| sample.x_m >= min_x && sample.x_m <= max_x)
            .cloned()
            .collect();

        // Need at least 2 points for meaningful integration
        if profile.len() < 2 {
            return None;
        }

        // Sort by distance to ensure monotonic increasing x
        profile.sort_by(|a, b| a.x_m.partial_cmp(&b.x_m).unwrap());

        Some(profile)
    }

    /// Generate a speed profile looking ahead from current position.
    ///
    /// Uses lookahead_m to determine the target range.
    ///
    /// # Arguments
    /// * `current_distance_m` - Current lap distance in meters
    ///
    /// # Returns
    /// List of speed samples ahead of current position, or None if insufficient
    pub fn get_lookahead_profile(&self, current_distance_m: f64) -> Option<Vec<SpeedSample>> {
        if self.window.len() < 2 {
            return None;
        }

        let target_distance_m = current_distance_m + self.lookahead_m;
        self.get_profile(current_distance_m, target_distance_m)
    }

    /// Clear the sliding window.
    pub fn reset(&mut self) {
        self.window.clear();
    }

    /// Get the current window size
    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_profile_generation() {
        let mut calc = SpeedProfileCalculator::new(20, 200.0);

        // Simulate telemetry: car accelerating from 1000m to 1500m
        for distance in (1000..1500).step_by(10) {
            // Speed increases linearly from 50 to 100 kph
            let speed_kph = 50.0 + (distance - 1000) as f64 / 10.0;
            calc.add_sample(distance as f64, speed_kph);
        }

        // Now at distance=1490m, ask for lookahead profile
        let current_distance = 1490.0;
        let profile = calc.get_lookahead_profile(current_distance);

        assert!(profile.is_some(), "Profile should be generated");
        let profile = profile.unwrap();
        
        println!("✓ Generated profile with {} samples", profile.len());
        println!("  First sample: x={:.1}m, v={:.2}m/s", profile[0].x_m, profile[0].v_mps);
        println!("  Last sample:  x={:.1}m, v={:.2}m/s", profile.last().unwrap().x_m, profile.last().unwrap().v_mps);

        // Verify samples are in expected range
        for sample in &profile {
            assert!(sample.x_m >= current_distance - 50.0 && sample.x_m <= current_distance + 200.0 + 50.0,
                "Profile sample outside expected range");
        }

        // Verify monotonic increasing x
        for i in 1..profile.len() {
            assert!(profile[i].x_m >= profile[i-1].x_m, "Profile not sorted by distance");
        }

        println!("✓ All assertions passed");
    }

    #[test]
    fn test_target_range_profile() {
        let mut calc = SpeedProfileCalculator::new(30, 300.0);

        // Add samples from 2000m to 2500m
        for distance in (2000..2500).step_by(5) {
            calc.add_sample(distance as f64, 80.0); // Constant speed
        }

        // Request profile from 2200m to 2400m
        let profile = calc.get_profile(2200.0, 2400.0);

        assert!(profile.is_some(), "Target range profile should be generated");
        let profile = profile.unwrap();

        println!("✓ Target range profile: {} samples from {:.1}m to {:.1}m", 
            profile.len(), profile[0].x_m, profile.last().unwrap().x_m);

        // Verify all samples are around the target range (with buffer tolerance)
        assert!(profile[0].x_m >= 2150.0, "Start too far back");
        assert!(profile.last().unwrap().x_m <= 2450.0, "End too far forward");

        // Verify speed conversion (80 kph = 22.22 m/s)
        let expected_mps = 80.0 / 3.6;
        for sample in &profile {
            assert!((sample.v_mps - expected_mps).abs() < 0.01, "Speed conversion error");
        }

        println!("✓ Target range test passed");
    }

    #[test]
    fn test_insufficient_data() {
        let mut calc = SpeedProfileCalculator::new(10, 100.0);

        // Add only 1 sample
        calc.add_sample(1000.0, 50.0);

        let profile = calc.get_lookahead_profile(1000.0);
        assert!(profile.is_none(), "Should return None with insufficient data");
        println!("✓ Insufficient data handling correct");
    }

    #[test]
    fn test_json_serialization() {
        let mut calc = SpeedProfileCalculator::new(15, 150.0);

        for distance in (3000..3200).step_by(10) {
            calc.add_sample(distance as f64, 75.0);
        }

        let profile = calc.get_lookahead_profile(3180.0);
        
        assert!(profile.is_some(), "Profile should be generated");
        let profile = profile.unwrap();

        // Test JSON serialization
        let json_str = serde_json::to_string_pretty(&profile).expect("Should serialize to JSON");
        println!("✓ JSON serialization successful ({} chars)", json_str.len());
        println!("Sample JSON:");
        let preview = if json_str.len() > 200 {
            format!("{}...", &json_str[..200])
        } else {
            json_str.clone()
        };
        println!("{}", preview);

        // Verify round-trip
        let decoded: Vec<SpeedSample> = serde_json::from_str(&json_str).expect("Should deserialize from JSON");
        assert_eq!(decoded.len(), profile.len(), "Round-trip failed");
        println!("✓ JSON round-trip successful");
    }

    #[test]
    fn test_window_size_limit() {
        let mut calc = SpeedProfileCalculator::new(5, 100.0);

        // Add more samples than window size
        for i in 0..10 {
            calc.add_sample(i as f64 * 10.0, 50.0);
        }

        // Window should only contain last 5 samples
        assert_eq!(calc.window_len(), 5, "Window should be limited to window_size");
    }

    #[test]
    fn test_reset() {
        let mut calc = SpeedProfileCalculator::new(10, 100.0);

        for i in 0..5 {
            calc.add_sample(i as f64 * 10.0, 50.0);
        }

        assert_eq!(calc.window_len(), 5);
        
        calc.reset();
        assert_eq!(calc.window_len(), 0, "Window should be empty after reset");
    }
}