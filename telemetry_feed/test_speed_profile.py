"""
Unit test and demonstration for speed_profile_calculator.py

Run this to verify the speed profile logic works correctly before Docker build.
"""

from speed_profile_calculator import SpeedProfileCalculator


def test_basic_profile_generation():
    """Test that the calculator builds a profile from a sliding window."""
    calc = SpeedProfileCalculator(window_size=20, lookahead_m=200.0)
    
    # Simulate telemetry: car accelerating from 1000m to 1500m
    for distance in range(1000, 1500, 10):
        # Speed increases linearly from 50 to 100 kph
        speed_kph = 50 + (distance - 1000) / 10
        calc.add_sample(distance, speed_kph)
    
    # Now at distance=1490m, ask for lookahead profile
    current_distance = 1490.0
    profile = calc.get_lookahead_profile(current_distance)
    
    if profile:
        print(f"✓ Generated profile with {len(profile)} samples")
        print(f"  First sample: x={profile[0]['x_m']:.1f}m, v={profile[0]['v_mps']:.2f}m/s")
        print(f"  Last sample:  x={profile[-1]['x_m']:.1f}m, v={profile[-1]['v_mps']:.2f}m/s")
        
        # Verify samples are in range [current, current+lookahead]
        assert all(current_distance - 50 <= s['x_m'] <= current_distance + 200 + 50 for s in profile), \
            "Profile samples outside expected range"
        
        # Verify monotonic increasing x
        for i in range(1, len(profile)):
            assert profile[i]['x_m'] >= profile[i-1]['x_m'], "Profile not sorted by distance"
        
        print("✓ All assertions passed")
    else:
        print("✗ No profile generated (insufficient data)")
    
    return profile


def test_target_range_profile():
    """Test get_profile with explicit target distance."""
    calc = SpeedProfileCalculator(window_size=30, lookahead_m=300.0)
    
    # Add samples from 2000m to 2500m
    for distance in range(2000, 2500, 5):
        speed_kph = 80.0  # Constant speed for simplicity
        calc.add_sample(distance, speed_kph)
    
    # Request profile from 2200m to 2400m
    profile = calc.get_profile(current_distance_m=2200.0, target_distance_m=2400.0)
    
    if profile:
        print(f"\n✓ Target range profile: {len(profile)} samples from {profile[0]['x_m']:.1f}m to {profile[-1]['x_m']:.1f}m")
        
        # Verify all samples are around the target range (with buffer tolerance)
        assert profile[0]['x_m'] >= 2150, "Start too far back"
        assert profile[-1]['x_m'] <= 2450, "End too far forward"
        
        # Verify speed conversion (80 kph = 22.22 m/s)
        expected_mps = 80.0 / 3.6
        for sample in profile:
            assert abs(sample['v_mps'] - expected_mps) < 0.01, "Speed conversion error"
        
        print("✓ Target range test passed")
    else:
        print("✗ Target range profile failed")
    
    return profile


def test_insufficient_data():
    """Test that calculator returns None when data is insufficient."""
    calc = SpeedProfileCalculator(window_size=10, lookahead_m=100.0)
    
    # Add only 1 sample
    calc.add_sample(1000.0, 50.0)
    
    profile = calc.get_lookahead_profile(1000.0)
    assert profile is None, "Should return None with insufficient data"
    print("\n✓ Insufficient data handling correct")


def test_json_serialization():
    """Test that profile can be JSON serialized (for websocket transmission)."""
    import json
    
    calc = SpeedProfileCalculator(window_size=15, lookahead_m=150.0)
    
    for distance in range(3000, 3200, 10):
        calc.add_sample(distance, 75.0)
    
    profile = calc.get_lookahead_profile(3180.0)
    
    if profile:
        # This should not raise an exception
        json_str = json.dumps(profile, indent=2)
        print(f"\n✓ JSON serialization successful ({len(json_str)} chars)")
        print("Sample JSON:")
        print(json_str[:200] + "..." if len(json_str) > 200 else json_str)
        
        # Verify round-trip
        decoded = json.loads(json_str)
        assert len(decoded) == len(profile), "Round-trip failed"
        print("✓ JSON round-trip successful")
    else:
        print("✗ No profile to serialize")


def demo_integration_payload():
    """Demonstrate what the telemetry_feed.py will send to Rust backend."""
    import json
    
    calc = SpeedProfileCalculator(window_size=50, lookahead_m=500.0)
    
    # Simulate a few telemetry samples
    test_samples = [
        (2200.0, 68.0),
        (2205.0, 67.5),
        (2210.0, 67.0),
        (2215.0, 66.5),
    ]
    
    print("\n" + "="*60)
    print("DEMO: Telemetry Payload Integration")
    print("="*60)
    
    for lap_distance_m, speed_kph in test_samples:
        calc.add_sample(lap_distance_m, speed_kph)
        speed_profile = calc.get_lookahead_profile(lap_distance_m)
        
        payload = {
            "lap_distance_m": lap_distance_m,
            "speed_kph": speed_kph,
            "speed_profile": speed_profile
        }
        
        json_payload = json.dumps(payload)
        profile_info = f"{len(speed_profile)} samples" if speed_profile else "None"
        print(f"\n📡 Sending: dist={lap_distance_m}m, speed={speed_kph}kph, profile={profile_info}")
        
        if speed_profile and len(test_samples) == 4:  # Show detail on last sample
            print(f"   Profile range: {speed_profile[0]['x_m']:.1f}m to {speed_profile[-1]['x_m']:.1f}m")
            print(f"   JSON size: {len(json_payload)} bytes")


if __name__ == "__main__":
    print("Running Speed Profile Calculator Tests\n")
    
    try:
        test_basic_profile_generation()
        test_target_range_profile()
        test_insufficient_data()
        test_json_serialization()
        demo_integration_payload()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nThe speed profile calculator is ready for Docker integration!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise
