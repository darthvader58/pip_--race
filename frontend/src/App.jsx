import React, { useEffect, useMemo, useRef, useState } from 'react'

function useTimerSocket(url) {
  const [data, setData] = useState({ t_call: null, t_safe: null, status: 'INIT', lap_distance_m: 0, speed_kph: 0 })
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    if (!url) return
    let alive = true
    const ws = new WebSocket(url)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        // Expecting { t_call, t_safe, status, lap_distance_m, speed_kph }
        if (alive) setData(msg)
      } catch (e) {
        // ignore
      }
    }
    return () => {
      alive = false
      try { ws.close() } catch {}
    }
  }, [url])

  return { data, connected }
}

function formatTime(seconds) {
  if (seconds == null || !isFinite(seconds)) return '00:00.00'
  const s = Math.max(0, Math.floor(seconds))
  const mm = String(Math.floor(s / 60)).padStart(2, '0')
  const ss = String(s % 60).padStart(2, '0')
  // Show hundredths (2 digits)
  const hundredths = Math.max(0, Math.floor((seconds - Math.floor(seconds)) * 100))
  const hsStr = String(hundredths).padStart(2, '0')
  return `${mm}:${ss}.${hsStr}`
}

function StatusBadge({ status }) {
  const cls = status === 'GREEN' ? 'green' : status === 'AMBER' ? 'yellow' : status === 'RED' ? 'red' : 'neutral'
  return <span className={`status-badge ${cls}`}>{status}</span>
}

function PitCountdown({ tSafe, status }) {
  const [display, setDisplay] = useState(tSafe ?? 0)

  useEffect(() => {
    setDisplay(tSafe ?? 0)
  }, [tSafe])

  // Update timer every 30ms for smooth hundredths
  useEffect(() => {
    let last = Date.now()
    const id = setInterval(() => {
      setDisplay((prev) => {
        if (prev <= 0) return 0
        const now = Date.now()
        const elapsed = (now - last) / 1000
        last = now
        return Math.max(0, prev - elapsed)
      })
    }, 30)
    return () => clearInterval(id)
  }, [])

  // Map status to color class
  let colorClass = '';
  switch ((status || '').toUpperCase()) {
    case 'GREEN':
      colorClass = 'timer-green';
      break;
    case 'AMBER':
      colorClass = 'timer-amber';
      break;
    case 'RED':
      colorClass = 'timer-red';
      break;
    case 'LOCKED':
    case 'LOCKED OUT':
    case 'LOCKED_OUT':
      colorClass = 'timer-locked';
      break;
    default:
      colorClass = 'timer-neutral';
  }
  return (
    <section className="panel pit-countdown-panel">
      <h2>PIT ENTRY COUNTDOWN <StatusBadge status={status} /></h2>
      <div className={`countdown-display ${colorClass}`} id="pitCountdown">{formatTime(display)}</div>
    </section>
  )
}

export default function App() {
  // Use env var VITE_BACKEND_WS or default to ws://localhost:8765
  const wsUrl = useMemo(() => {
    return import.meta.env.VITE_BACKEND_WS || 'ws://localhost:8765'
  }, [])

  const { data, connected } = useTimerSocket(wsUrl)

  return (
    <>
      <header className="dashboard-header">
        <div className="logo">
          <span className="f1-text">F1</span> DASHBOARD {connected ? '• online' : '• offline'}
        </div>
      </header>

      <div className="dashboard-container">
        <main className="dashboard-grid">
          <section className="panel rival-boxing-panel">
            <h2>RIVAL BOXING LIKELIHOOD</h2>
            <div className="teams-list">
              {[
                { name: 'M. Verstappen', car: 'Red Bull', icon: 'red-bull-icon', p: 83 },
                { name: 'F. Alonso', car: 'Aston Martin', icon: 'aston-martin-icon', p: 62 },
                { name: 'E. Ocon', car: 'Alpine', icon: 'alpine-icon', p: 42 },
                { name: 'L. Hamilton', car: 'Mercedes', icon: 'mercedes-icon', p: 75 },
                { name: 'G. Russell', car: 'Mercedes', icon: 'mercedes-icon', p: 78 },
                { name: 'C. Leclerc', car: 'Ferrari', icon: 'ferrari-icon', p: 55 },
              ].map((t) => (
                <div key={t.name} className="team-item">
                  <div className="team-name">
                    <span className={`team-icon ${t.icon}`}></span>
                    <div className="driver-info">
                      <span className="driver-name">{t.name}</span>
                      <span className="car-name">{t.car}</span>
                    </div>
                  </div>
                  <div className="progress-bar-container">
                    <div
                      className={`progress-bar ${t.p >= 80 ? 'green' : t.p >= 50 ? 'yellow' : 'red'}`}
                      style={{ width: `${t.p}%` }}
                    />
                  </div>
                  <span className="likelihood-value">{t.p}%</span>
                  <i className={`fas ${t.p >= 60 ? 'fa-arrow-up trend-up' : 'fa-arrow-down trend-down'}`}></i>
                </div>
              ))}
            </div>
          </section>

          <section className="panel circuit-map-panel">
            <img src="/circuit_map.png" alt="Circuit Map" className="circuit-map" />
          </section>

          <PitCountdown tSafe={data.t_safe} status={data.status} />
        </main>
      </div>
    </>
  )
}
