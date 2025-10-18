"""
Speed profile calculator for pit timer integration.

Given a sliding window of telemetry samples, computes a speed profile
suitable for trapezoidal integration of time-to-call in the Rust backend.
"""

from typing import List, Dict, Optional
from collections import deque
import pandas as pd


class SpeedProfileCalculator:
    """
    Maintains a sliding window of recent telemetry and generates
    speed profiles for integration-based time estimation.
    """
    
    def __init__(self, window_size: int = 50, lookahead_m: float = 500.0):
        """
        Initialize the speed profile calculator.
        
        Args:
            window_size: Number of recent samples to keep in the sliding window
            lookahead_m: Distance ahead (meters) to include in the profile for integration
        """
        self.window_size = window_size
        self.lookahead_m = lookahead_m
        self.window = deque(maxlen=window_size)
    
    def add_sample(self, lap_distance_m: float, speed_kph: float):
        """
        Add a new telemetry sample to the sliding window.
        
        Args:
            lap_distance_m: Current lap distance in meters
            speed_kph: Current speed in km/h
        """
        speed_mps = speed_kph / 3.6  # Convert to m/s
        self.window.append({
            'x_m': lap_distance_m,
            'v_mps': speed_mps
        })
    
    def get_profile(self, current_distance_m: float, target_distance_m: float) -> Optional[List[Dict[str, float]]]:
        """
        Generate a speed profile from current position toward target.
        
        Returns samples in the range [current_distance_m, target_distance_m]
        or slightly beyond if we have historical data that covers that range.
        
        Args:
            current_distance_m: Current lap distance in meters
            target_distance_m: Target call point distance in meters
            
        Returns:
            List of {'x_m': float, 'v_mps': float} dicts, or None if insufficient data
        """
        if len(self.window) < 2:
            return None
        
        # Filter samples in the range [current_distance, target_distance]
        # Include a small buffer before/after for smoother integration
        buffer_m = 50.0
        min_x = current_distance_m - buffer_m
        max_x = target_distance_m + buffer_m
        
        profile = []
        for sample in self.window:
            x = sample['x_m']
            if min_x <= x <= max_x:
                profile.append({
                    'x_m': sample['x_m'],
                    'v_mps': sample['v_mps']
                })
        
        # Need at least 2 points for meaningful integration
        if len(profile) < 2:
            return None
        
        # Sort by distance to ensure monotonic increasing x
        profile.sort(key=lambda s: s['x_m'])
        
        return profile
    
    def get_lookahead_profile(self, current_distance_m: float) -> Optional[List[Dict[str, float]]]:
        """
        Generate a speed profile looking ahead from current position.
        
        Uses lookahead_m to determine the target range.
        
        Args:
            current_distance_m: Current lap distance in meters
            
        Returns:
            List of speed samples ahead of current position, or None if insufficient
        """
        if len(self.window) < 2:
            return None
        
        target_distance_m = current_distance_m + self.lookahead_m
        return self.get_profile(current_distance_m, target_distance_m)
    
    def reset(self):
        """Clear the sliding window."""
        self.window.clear()


def calculate_speed_profile_from_dataframe(
    telemetry_df: pd.DataFrame,
    current_idx: int,
    window_size: int = 50,
    lookahead_m: float = 500.0
) -> Optional[List[Dict[str, float]]]:
    """
    Static helper: compute speed profile from a DataFrame slice.
    
    Useful for batch processing or when working directly with FastF1 data.
    
    Args:
        telemetry_df: DataFrame with 'Distance' and 'Speed' columns
        current_idx: Current row index in the DataFrame
        window_size: Number of samples to look back
        lookahead_m: Distance ahead to include
        
    Returns:
        List of speed profile samples, or None if insufficient data
    """
    if current_idx < 1 or current_idx >= len(telemetry_df):
        return None
    
    current_distance = telemetry_df.iloc[current_idx]['Distance']
    target_distance = current_distance + lookahead_m
    
    # Slice window: look back from current position
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(telemetry_df), current_idx + window_size // 2)  # Include some forward samples if available
    
    profile = []
    for idx in range(start_idx, end_idx):
        row = telemetry_df.iloc[idx]
        x_m = float(row['Distance'])
        speed_kph = float(row['Speed'])
        
        # Only include samples in relevant range
        if current_distance <= x_m <= target_distance:
            profile.append({
                'x_m': x_m,
                'v_mps': speed_kph / 3.6
            })
    
    if len(profile) < 2:
        return None
    
    # Sort by distance
    profile.sort(key=lambda s: s['x_m'])
    
    return profile
