import { useState, useEffect, useRef } from 'react';

/**
 * WebSocket hook for pit probabilities from bridge service
 * Connects to ws://localhost:8081 and receives:
 * {
 *   type: 'UPDATE' | 'FULL_STATE',
 *   prediction: { driver, lap, p2, p3, trend, timestamp }
 *   OR
 *   probabilities: [{ driver, lap, p2, p3, trend }]
 * }
 */

// Driver code to team mapping
const DRIVER_TEAMS = {
  'VER': 'Red Bull', 'PER': 'Red Bull',
  'HAM': 'Mercedes', 'RUS': 'Mercedes',
  'LEC': 'Ferrari', 'SAI': 'Ferrari',
  'NOR': 'McLaren', 'PIA': 'McLaren',
  'ALO': 'Aston Martin', 'STR': 'Aston Martin',
  'GAS': 'Alpine', 'OCO': 'Alpine',
  'BOT': 'Alfa Romeo', 'ZHO': 'Alfa Romeo',
  'MAG': 'Haas', 'HUL': 'Haas',
  'TSU': 'AlphaTauri', 'RIC': 'AlphaTauri',
  'ALB': 'Williams', 'SAR': 'Williams',
  'DEV': 'AlphaTauri', 'LAW': 'Williams'
};

function getTeamForDriver(driverCode) {
  const code = driverCode.toUpperCase().slice(0, 3);
  return DRIVER_TEAMS[code] || 'Unknown';
}

function getDriverName(driverCode) {
  const names = {
    'VER': 'M. Verstappen', 'PER': 'S. Perez',
    'HAM': 'L. Hamilton', 'RUS': 'G. Russell',
    'LEC': 'C. Leclerc', 'SAI': 'C. Sainz',
    'NOR': 'L. Norris', 'PIA': 'O. Piastri',
    'ALO': 'F. Alonso', 'STR': 'L. Stroll',
    'GAS': 'P. Gasly', 'OCO': 'E. Ocon',
    'BOT': 'V. Bottas', 'ZHO': 'G. Zhou',
    'MAG': 'K. Magnussen', 'HUL': 'N. Hulkenberg',
    'TSU': 'Y. Tsunoda', 'RIC': 'D. Ricciardo',
    'ALB': 'A. Albon', 'SAR': 'L. Sargeant',
  };
  const code = driverCode.toUpperCase().slice(0, 3);
  return names[code] || driverCode;
}

export default function usePitProbabilities(url) {
  const [probabilities, setProbabilities] = useState([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);

  useEffect(() => {
    if (!url) {
      console.warn('[PitProbs] No WebSocket URL provided');
      return;
    }

    let isAlive = true;
    const maxReconnectAttempts = 10;
    const baseReconnectDelay = 1000;

    const connect = () => {
      if (!isAlive) return;

      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch (e) {
          // Ignore
        }
      }

      console.log(`[PitProbs] Connecting to ${url}...`);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`[PitProbs] Connected to ${url}`);
        setConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
      };

      ws.onclose = (event) => {
        console.log(`[PitProbs] Disconnected (code: ${event.code})`);
        setConnected(false);

        if (isAlive && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
            30000
          );
          
          console.log(
            `[PitProbs] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`
          );
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError('Bridge service unavailable. Check if bridge is running.');
          console.error('[PitProbs] Max reconnection attempts reached');
        }
      };

      ws.onerror = (error) => {
        console.error('[PitProbs] WebSocket error:', error);
        setError('Bridge connection error');
        setConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          
          if (msg.type === 'FULL_STATE' && Array.isArray(msg.probabilities)) {
            // Initial full state from bridge
            const formatted = msg.probabilities.map(p => ({
              driver: getDriverName(p.driver),
              driverCode: p.driver,
              team: getTeamForDriver(p.driver),
              lap: p.lap,
              p2: p.p2,
              p3: p.p3,
              trend: p.trend || 'stable',
              timestamp: p.timestamp
            }));
            setProbabilities(formatted);
            console.log(`[PitProbs] Loaded ${formatted.length} drivers`);
            
          } else if (msg.type === 'UPDATE' && msg.prediction) {
            // Single driver update
            const pred = msg.prediction;
            const formatted = {
              driver: getDriverName(pred.driver),
              driverCode: pred.driver,
              team: getTeamForDriver(pred.driver),
              lap: pred.lap,
              p2: pred.p2,
              p3: pred.p3,
              trend: pred.trend || 'stable',
              timestamp: pred.timestamp
            };
            
            setProbabilities(prev => {
              const existing = prev.find(p => p.driverCode === pred.driver);
              if (existing) {
                // Update existing
                return prev.map(p => 
                  p.driverCode === pred.driver ? formatted : p
                );
              } else {
                // Add new driver
                return [...prev, formatted].sort((a, b) => b.p2 - a.p2);
              }
            });
          }
        } catch (e) {
          console.error('[PitProbs] Failed to parse message:', e, event.data);
        }
      };
    };

    connect();

    return () => {
      console.log('[PitProbs] Cleaning up connection');
      isAlive = false;
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch (e) {
          // Ignore
        }
      }
    };
  }, [url]);

  return { probabilities, connected, error };
}