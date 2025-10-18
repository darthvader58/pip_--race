# Speed Profile Integration

This module provides speed profile calculation for the pit timer backend, enabling more accurate time-to-call estimates through integration over the speed profile rather than instantaneous speed approximation.

## Overview

The `speed_profile_calculator.py` module maintains a sliding window of recent telemetry samples and generates speed profiles suitable for trapezoidal integration in the Rust backend's `time_to_call` function.

### Mathematical Background

Instead of the simple instantaneous estimate:
```
t_call ≈ d_rem / v_instantaneous
```

We integrate over the speed profile:
```
t_call = ∫[x=current to x=target] (1/v(x)) dx
```

This accounts for speed variations (braking, cornering, acceleration) along the path to the call point, providing a more accurate time estimate.

## Architecture

### Python Side (`telemetry_feed`)

1. **`speed_profile_calculator.py`**: Core calculator module
   - `SpeedProfileCalculator`: Maintains sliding window of samples
   - Generates profiles as `List[{"x_m": float, "v_mps": float}]`
   - Configurable window size and lookahead distance

2. **`telemetry_feed.py`**: Updated to include profiles
   - Creates `SpeedProfileCalculator` instance
   - Adds each sample to the sliding window
   - Includes `speed_profile` in JSON payload sent to Rust backend

### Rust Side (`pit_timer_backend`)

1. **`model.rs`**: Updated `time_to_call` function
   - Accepts optional `speed_profile` field in `TelemetryPacket`
   - Uses `integrate_time_over_profile` for trapezoidal integration
   - Falls back to instantaneous estimate if no profile provided
   - Backward compatible: works with or without speed_profile

## Configuration

### SpeedProfileCalculator Parameters

- **`window_size`** (default: 50): Number of recent samples to keep
  - Larger = more historical context, higher memory
  - Smaller = more responsive to changes, less context
  - Recommended: 30-100 samples

- **`lookahead_m`** (default: 500.0): Distance ahead to include in profile
  - Should cover distance from current position to call point
  - Larger = more comprehensive, but may include irrelevant data
  - Recommended: 300-800 meters depending on track

### Example Usage

```python
from speed_profile_calculator import SpeedProfileCalculator

# Initialize
calc = SpeedProfileCalculator(window_size=50, lookahead_m=500.0)

# Add samples as they arrive
for distance, speed in telemetry_stream:
    calc.add_sample(lap_distance_m=distance, speed_kph=speed)
    
    # Generate profile for current position
    profile = calc.get_lookahead_profile(distance)
    
    # Send to backend
    payload = {
        "lap_distance_m": distance,
        "speed_kph": speed,
        "speed_profile": profile  # None initially, then list of samples
    }
```

## Testing

Run the test suite to verify the calculator:

```bash
cd telemetry_feed
python test_speed_profile.py
```

Tests cover:
- Basic profile generation from sliding window
- Target range profile extraction
- Insufficient data handling
- JSON serialization for websocket transmission
- Integration payload demonstration

## JSON Schema

The `speed_profile` field in telemetry packets:

```json
{
  "lap_distance_m": 2200.5,
  "speed_kph": 68.0,
  "speed_profile": [
    {"x_m": 2180.0, "v_mps": 18.5},
    {"x_m": 2190.0, "v_mps": 18.7},
    {"x_m": 2200.0, "v_mps": 18.9},
    ...
  ]
}
```

- `x_m`: Lap distance in meters
- `v_mps`: Speed in meters per second (converted from kph)
- Array is sorted by increasing `x_m`
- May be `null` if insufficient data

## Performance Considerations

### Python Side
- Sliding window uses `deque` with O(1) append and pop
- Profile generation is O(n) where n = window_size
- Typical overhead: <1ms per sample at 50-100 window size

### Rust Side
- Trapezoidal integration is O(m) where m = profile length
- Fallback to instantaneous is O(1)
- No allocation if profile is None

### Network
- Profile adds ~20-50 bytes per sample to JSON payload
- At 50 samples: ~1-2.5 KB per packet
- At 10 Hz stream: ~10-25 KB/s additional bandwidth
- Negligible for local websocket connections

## Troubleshooting

### No profile generated (always None)
- **Cause**: Insufficient samples in window (<2)
- **Fix**: Wait for window to fill; ensure telemetry is streaming

### Profile doesn't include future samples
- **Cause**: Lookahead distance too small or no future data available
- **Fix**: Increase `lookahead_m` or ensure telemetry includes upcoming data

### Integration not used (logs show instantaneous estimate)
- **Cause**: Profile samples don't cover [current, target] range
- **Fix**: Verify `lookahead_m` >= distance to call point; check profile range in logs

### Large JSON payload size
- **Cause**: Window size too large
- **Fix**: Reduce `window_size` to 20-30 samples; profile only needs recent data

## Future Enhancements

- [ ] Exponential smoothing for noise reduction
- [ ] Track-specific profiles (use historical sector data)
- [ ] Adaptive window sizing based on speed variance
- [ ] Compression for network efficiency
- [ ] Multi-lap profile stitching for long predictions
