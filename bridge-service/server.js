/**
 * WebSocket Bridge Service
 * 
 * Receives predictions from feeder_fastf1_cache.py via HTTP POST
 * Broadcasts them to frontend via WebSocket
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Store latest predictions by driver
const predictions = new Map();

// Allow JSON body parsing
app.use(express.json());

// WebSocket connection handler
wss.on('connection', (ws, req) => {
  console.log(`[Bridge] Frontend connected from ${req.socket.remoteAddress}`);
  
  // Send current state immediately on connection
  const currentState = Array.from(predictions.values());
  ws.send(JSON.stringify({
    type: 'FULL_STATE',
    probabilities: currentState,
    timestamp: Date.now()
  }));
  
  ws.on('close', () => {
    console.log('[Bridge] Frontend disconnected');
  });
});

// Broadcast to all connected WebSocket clients
function broadcast(data) {
  const message = JSON.stringify(data);
  let clientCount = 0;
  
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
      clientCount++;
    }
  });
  
  if (clientCount > 0) {
    console.log(`[Bridge] Broadcasted to ${clientCount} client(s)`);
  }
}

/**
 * Endpoint to receive predictions from rt_predictor via feeder
 * 
 * Expected payload from modified feeder_fastf1_cache.py:
 * {
 *   "driver": "VER",
 *   "lap": 45,
 *   "p2": 0.87,
 *   "p3": 0.73,
 *   "t": 1234567890
 * }
 */
app.post('/update', (req, res) => {
  const { driver, lap, p2, p3, t } = req.body;
  
  if (!driver || p2 === undefined || p3 === undefined) {
    return res.status(400).json({ 
      error: 'Missing required fields: driver, p2, p3' 
    });
  }
  
  // Store/update prediction
  const existing = predictions.get(driver);
  const prediction = {
    driver,
    lap: lap || 0,
    p2: parseFloat(p2),
    p3: parseFloat(p3),
    timestamp: t || Date.now(),
    // Calculate trend if we have previous data
    trend: existing ? calculateTrend(existing.p2, p2) : 'stable'
  };
  
  predictions.set(driver, prediction);
  
  console.log(
    `[Bridge] ${driver} L${lap}: p2=${(p2 * 100).toFixed(0)}%, p3=${(p3 * 100).toFixed(0)}%`
  );
  
  // Broadcast update to all frontends
  broadcast({
    type: 'UPDATE',
    prediction
  });
  
  res.json({ ok: true, driver, lap });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    activeConnections: wss.clients.size,
    driversTracked: predictions.size,
    uptime: process.uptime()
  });
});

// Get all current predictions
app.get('/predictions', (req, res) => {
  res.json({
    probabilities: Array.from(predictions.values()),
    count: predictions.size
  });
});

function calculateTrend(oldP2, newP2) {
  const diff = newP2 - oldP2;
  if (diff > 0.05) return 'up';
  if (diff < -0.05) return 'down';
  return 'stable';
}

const PORT = process.env.PORT || 8081;

server.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('🔗 Williams Racing - Pit Probability Bridge Service');
  console.log('='.repeat(60));
  console.log(`HTTP Server:    http://localhost:${PORT}`);
  console.log(`WebSocket:      ws://localhost:${PORT}`);
  console.log(`Health Check:   http://localhost:${PORT}/health`);
  console.log('='.repeat(60));
  console.log('\nWaiting for predictions from feeder...\n');
});