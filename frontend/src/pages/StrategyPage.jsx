import React from 'react';
import PitProbabilities from '../components/PitProbabilities';
import BoxWindow from '../components/BoxWindow';
import TrackMap from '../components/TrackMap';

export default function StrategyPage({ timerData, pitProbs, connected }) {
  return (
    <div className="page strategy-page">
      <div className="strategy-header">
        <h1 className="page-title">Strategy Control</h1>
        <div className="telemetry-info">
          {timerData && (
            <>
              <span className="telem-item">
                Distance: <strong>{timerData.lap_distance_m?.toFixed(0) || 0}m</strong>
              </span>
              <span className="telem-item">
                Speed: <strong>{timerData.speed_kph?.toFixed(0) || 0} km/h</strong>
              </span>
            </>
          )}
        </div>
      </div>
      
      <div className="strategy-grid">
        <div className="left-column">
          <PitProbabilities probabilities={pitProbs} />
        </div>
        
        <div className="right-column">
          <BoxWindow 
            status={timerData?.status || 'INIT'}
            timeRemaining={timerData?.t_safe || 0}
          />
          <TrackMap />
        </div>
      </div>
    </div>
  );
}