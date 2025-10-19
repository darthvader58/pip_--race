import React from 'react';
import useRouter from './hooks/useRouter';
import useWebSocket from './hooks/useWebSocket';
import Navigation from './components/Navigation';
import LiveRacePage from './pages/LiveRacePage';
import StrategyPage from './pages/StrategyPage';
import AboutPage from './pages/AboutPage';

export default function App() {
  const { route, navigate } = useRouter();
  
  // Connect to pit timer backend (box window + telemetry)
  // This connects to your Rust backend at pit_timer_backend/src/main.rs
  const wsUrl = import.meta.env.VITE_PIT_TIMER_WS || 'ws://localhost:8765';
  const { data: timerData, connected, error } = useWebSocket(wsUrl);

  // Mock pit probabilities data structure
  // In production, integrate with rt_predictor (see PIT_PROBABILITIES_INTEGRATION.md)
  const [pitProbs, setPitProbs] = React.useState([
    { driver: 'M. Verstappen', team: 'Red Bull', p2: 0.87, p3: 0.73, trend: 'up' },
    { driver: 'C. Leclerc', team: 'Ferrari', p2: 0.73, p3: 0.68, trend: 'up' },
    { driver: 'S. Perez', team: 'Red Bull', p2: 0.68, p3: 0.62, trend: 'down' },
    { driver: 'L. Hamilton', team: 'Mercedes', p2: 0.62, p3: 0.58, trend: 'up' },
    { driver: 'F. Alonso', team: 'Aston Martin', p2: 0.58, p3: 0.52, trend: 'stable' },
    { driver: 'G. Russell', team: 'Mercedes', p2: 0.54, p3: 0.48, trend: 'up' },
    { driver: 'C. Sainz', team: 'Ferrari', p2: 0.45, p3: 0.42, trend: 'down' },
    { driver: 'E. Ocon', team: 'Alpine', p2: 0.38, p3: 0.35, trend: 'stable' },
  ]);

  // Mock Williams car data (could also come from telemetry)
  const williamsData = {
    car23: {
      driver: 'A. Albon',
      number: '23',
      position: 12,
      lapTime: '1:14.523',
      gap: '+24.5s',
      tire: 'Medium',
      tireLaps: 28
    },
    car2: {
      driver: 'L. Sargeant',
      number: '2',
      position: 16,
      lapTime: '1:15.102',
      gap: '+38.2s',
      tire: 'Hard',
      tireLaps: 32
    }
  };

  return (
    <div className="app">
      <Navigation 
        currentRoute={route} 
        onNavigate={navigate} 
        connected={connected}
        error={error}
      />
      
      <main className="main-content">
        {route === '/' && (
          <LiveRacePage williamsData={williamsData} />
        )}
        {route === '/strategy' && (
          <StrategyPage 
            timerData={timerData}
            pitProbs={pitProbs}
            connected={connected}
            error={error}
          />
        )}
        {route === '/about' && (
          <AboutPage />
        )}
      </main>
    </div>
  );
}