import React from 'react';

export default function PitProbabilities({ probabilities }) {
  const getBarColor = (p2) => {
    // Using p2 (2-lap probability) for color coding
    if (p2 >= 0.7) return '#e74c3c'; // High probability - red
    if (p2 >= 0.5) return '#f39c12'; // Medium - orange
    return '#2ecc71'; // Low - green
  };

  const getTrendIcon = (trend) => {
    if (trend === 'up') return '↑';
    if (trend === 'down') return '↓';
    return '•';
  };

  const getTrendColor = (trend) => {
    if (trend === 'up') return '#e74c3c';
    if (trend === 'down') return '#2ecc71';
    return '#95a5a6';
  };

  return (
    <div className="pit-probabilities-panel">
      <h2 className="panel-title">Pit Stop Probabilities - Next 2 Laps</h2>
      
      <div className="probabilities-list">
        {probabilities.map((item, i) => (
          <div key={i} className="prob-row">
            <div className="prob-driver">
              <span className="prob-driver-name">{item.driver}</span>
              <span className="prob-team">{item.team}</span>
            </div>
            
            <div className="prob-bar-container">
              <div 
                className="prob-bar"
                style={{ 
                  width: `${item.p2 * 100}%`,
                  backgroundColor: getBarColor(item.p2)
                }}
              />
            </div>
            
            <span className="prob-value">{(item.p2 * 100).toFixed(0)}%</span>
            
            <span 
              className="prob-trend"
              style={{ color: getTrendColor(item.trend) }}
            >
              {getTrendIcon(item.trend)}
            </span>
          </div>
        ))}
      </div>
      
      <div className="panel-note">
        Probabilities based on QR-DQN model trained on Monaco 2023 race data
      </div>
    </div>
  );
}