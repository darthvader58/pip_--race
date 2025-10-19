import React from 'react';

export default function RaceInfo() {
  const raceData = {
    track: 'Circuit de Monaco',
    totalLaps: 78,
    currentLap: 45,
    weather: 'Dry',
    temperature: '24°C',
    trackStatus: 'Green Flag'
  };

  return (
    <div className="race-info-grid">
      <div className="info-card">
        <div className="info-label">Circuit</div>
        <div className="info-value">{raceData.track}</div>
      </div>
      
      <div className="info-card">
        <div className="info-label">Laps</div>
        <div className="info-value">{raceData.currentLap} / {raceData.totalLaps}</div>
      </div>
      
      <div className="info-card">
        <div className="info-label">Weather</div>
        <div className="info-value">{raceData.weather}</div>
      </div>
      
      <div className="info-card">
        <div className="info-label">Temperature</div>
        <div className="info-value">{raceData.temperature}</div>
      </div>
      
      <div className="info-card">
        <div className="info-label">Track Status</div>
        <div className="info-value status-green">{raceData.trackStatus}</div>
      </div>
    </div>
  );
}