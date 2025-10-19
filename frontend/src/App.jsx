import React from 'react';
import useRouter from './hooks/useRouter';
import useWebSocket from './hooks/useWebSocket';
import usePitProbabilities from './hooks/usePitProbabilities';
import Navigation from './components/Navigation';
import LiveRacePage from './pages/LiveRacePage';
import StrategyPage from './pages/StrategyPage';
import AboutPage from './pages/AboutPage';

export default function App() {
  const { route, navigate } = useRouter();
  
  // Connect to pit timer backend (box window + telemetry)
  const timerWsUrl = import.meta.env.VITE_PIT_TIMER_WS || 'ws://localhost:8765';
  const { data: timerData, connected: timerConnected, error: timerError } = useWebSocket(timerWsUrl);

  // Connect to bridge service for pit probabilities
  const probsWsUrl = import.meta.env.VITE_PROBS_WS || 'ws://localhost:8081';
  const { probabilities, connected: probsConnected, error: probsError } = usePitProbabilities(probsWsUrl);

  // Combined connection status
  const connected = timerConnected && probsConnected;
  const error = timerError || probsError;

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
            pitProbs={probabilities}
            connected={connected}
            timerConnected={timerConnected}
            probsConnected={probsConnected}
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