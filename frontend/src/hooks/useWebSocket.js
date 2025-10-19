import { useState, useEffect, useRef } from 'react';

/**
 * WebSocket hook for pit_timer_backend connection
 * Connects to ws://localhost:8765 and receives:
 * {
 *   t_call: f64,
 *   t_safe: f64,
 *   status: "GREEN" | "AMBER" | "RED" | "LOCKED_OUT",
 *   lap_distance_m: f64,
 *   speed_kph: f64
 * }
 */
export default function useWebSocket(url) {
  const [data, setData] = useState({
    t_call: null,
    t_safe: null,
    status: 'INIT',
    lap_distance_m: 0,
    speed_kph: 0
  });
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);

  useEffect(() => {
    if (!url) {
      console.warn('No WebSocket URL provided');
      return;
    }

    let isAlive = true;
    const maxReconnectAttempts = 10;
    const baseReconnectDelay = 1000; // 1 second

    const connect = () => {
      if (!isAlive) return;

      // Clear any existing connection
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch (e) {
          // Ignore close errors
        }
      }

      console.log(`[WebSocket] Connecting to ${url}...`);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`[WebSocket] Connected to ${url}`);
        setConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0; // Reset on successful connection
      };

      ws.onclose = (event) => {
        console.log(`[WebSocket] Disconnected from ${url} (code: ${event.code})`);
        setConnected(false);

        // Auto-reconnect with exponential backoff
        if (isAlive && reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
            30000 // Max 30 seconds
          );
          
          console.log(
            `[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})...`
          );
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            connect();
          }, delay);
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError('Max reconnection attempts reached. Please refresh the page.');
          console.error('[WebSocket] Max reconnection attempts reached');
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setError('WebSocket connection error');
        setConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          
          // Validate the message structure
          if (typeof msg === 'object' && msg !== null) {
            // Log first few messages for debugging
            if (reconnectAttemptsRef.current === 0) {
              console.log('[WebSocket] Received data:', msg);
            }
            
            // Update state with new data
            if (isAlive) {
              setData({
                t_call: msg.t_call ?? null,
                t_safe: msg.t_safe ?? null,
                status: msg.status || 'INIT',
                lap_distance_m: msg.lap_distance_m ?? 0,
                speed_kph: msg.speed_kph ?? 0
              });
            }
          } else {
            console.warn('[WebSocket] Received invalid message format:', msg);
          }
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e, event.data);
        }
      };
    };

    // Initial connection
    connect();

    // Cleanup function
    return () => {
      console.log('[WebSocket] Cleaning up connection');
      isAlive = false;
      
      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      // Close WebSocket
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch (e) {
          // Ignore close errors
        }
      }
    };
  }, [url]);

  return { data, connected, error };
}