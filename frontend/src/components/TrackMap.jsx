import React from 'react';

export default function TrackMap() {
  return (
    <div className="track-map-panel">
      <h3 className="panel-title">Circuit de Monaco</h3>
      
      <div className="track-container">
        {/* Monaco track image from public folder */}
        <img 
          src="/monaco-track.png" 
          alt="Circuit de Monaco"
          className="track-image"
          onError={(e) => {
            // Fallback if image not found
            e.target.style.display = 'none';
            e.target.nextSibling.style.display = 'flex';
          }}
        />
        
        {/* Placeholder if image doesn't load */}
        <div className="track-placeholder" style={{ display: 'none' }}>
          <div className="placeholder-content">
            <div className="placeholder-icon">🏁</div>
            <div className="placeholder-text">Monaco Track Map</div>
            <div className="placeholder-note">
              Place your track image at: <code>public/monaco_track.png</code>
            </div>
          </div>
        </div>
      </div>
      
      <div className="track-info">
        <div className="track-stat">
          <span className="stat-label">Length:</span>
          <span className="stat-value">3.337 km</span>
        </div>
        <div className="track-stat">
          <span className="stat-label">Turns:</span>
          <span className="stat-value">19</span>
        </div>
        <div className="track-stat">
          <span className="stat-label">Pit Entry:</span>
          <span className="stat-value">~2,850m</span>
        </div>
      </div>
    </div>
  );
}