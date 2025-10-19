import React from 'react';

export default function CarCard({ car }) {
  const getTireColor = (tire) => {
    const colors = {
      'Soft': '#e74c3c',
      'Medium': '#f39c12',
      'Hard': '#f1f1f1'
    };
    return colors[tire] || '#95a5a6';
  };

  const getTireHealth = (laps) => {
    // Rough health based on typical stint lengths
    if (laps > 35) return { percentage: 20, color: '#e74c3c' };
    if (laps > 20) return { percentage: 50, color: '#f39c12' };
    return { percentage: 80, color: '#2ecc71' };
  };

  const health = getTireHealth(car.tireLaps);

  return (
    <div className="car-card">
      <div className="car-number">#{car.number}</div>
      <h2 className="driver-name">{car.driver}</h2>
      
      <div className="stats-grid">
        <div className="stat">
          <span className="stat-label">Position</span>
          <span className="stat-value">P{car.position}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Lap Time</span>
          <span className="stat-value">{car.lapTime}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Gap to Leader</span>
          <span className="stat-value">{car.gap}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Tire Compound</span>
          <span 
            className="stat-value" 
            style={{ color: getTireColor(car.tire) }}
          >
            {car.tire}
          </span>
        </div>
      </div>
      
      <div className="tire-life">
        <div className="tire-header">
          <span className="stat-label">Tire Life: {car.tireLaps} laps</span>
          <span className="tire-percentage">{health.percentage}%</span>
        </div>
        <div className="tire-bar">
          <div 
            className="tire-progress" 
            style={{ 
              width: `${health.percentage}%`,
              backgroundColor: health.color
            }}
          />
        </div>
      </div>
    </div>
  );
}