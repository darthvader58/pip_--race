import React, { useState, useEffect } from 'react';

export default function BoxWindow({ status, timeRemaining }) {
  const [display, setDisplay] = useState(timeRemaining || 0);

  // Update display when new time comes in
  useEffect(() => {
    setDisplay(timeRemaining || 0);
  }, [timeRemaining]);

  // Countdown timer (updates every 30ms for smooth display)
  useEffect(() => {
    let lastTime = Date.now();
    
    const interval = setInterval(() => {
      setDisplay(prev => {
        if (prev <= 0) return 0;
        const now = Date.now();
        const elapsed = (now - lastTime) / 1000;
        lastTime = now;
        return Math.max(0, prev - elapsed);
      });
    }, 30);

    return () => clearInterval(interval);
  }, []);

  const formatTime = (seconds) => {
    if (!isFinite(seconds)) return '00.00';
    const s = Math.floor(seconds);
    const hundredths = Math.floor((seconds - s) * 100);
    return `${String(s).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
  };

  const getStatusColor = () => {
    switch (status?.toUpperCase()) {
      case 'GREEN':
        return { color: '#2ecc71', text: 'CLEAR TO PIT' };
      case 'AMBER':
        return { color: '#f39c12', text: 'PREPARE TO PIT' };
      case 'RED':
        return { color: '#e74c3c', text: 'PIT NOW!' };
      case 'LOCKED_OUT':
        return { color: '#95a5a6', text: 'PIT WINDOW CLOSED' };
      default:
        return { color: '#00539f', text: 'INITIALIZING...' };
    }
  };

  const statusInfo = getStatusColor();

  return (
    <div className="box-window-panel">
      <div className="box-header">
        <h3 className="panel-title">Box Window</h3>
        <span 
          className="status-badge"
          style={{ 
            backgroundColor: `${statusInfo.color}33`,
            color: statusInfo.color,
            border: `1px solid ${statusInfo.color}66`
          }}
        >
          {status}
        </span>
      </div>
      
      <div 
        className="box-timer"
        style={{
          color: statusInfo.color,
          textShadow: `0 0 20px ${statusInfo.color}88`
        }}
      >
        {formatTime(display)}
      </div>
      
      <div 
        className="box-message"
        style={{ color: statusInfo.color }}
      >
        {statusInfo.text}
      </div>
      
      <div className="box-info">
        <div className="info-item">
          <span className="info-label">t_call:</span>
          <span className="info-value">{formatTime(timeRemaining + 3.5)}s</span>
        </div>
        <div className="info-item">
          <span className="info-label">t_safe:</span>
          <span className="info-value">{formatTime(timeRemaining)}s</span>
        </div>
      </div>
    </div>
  );
}