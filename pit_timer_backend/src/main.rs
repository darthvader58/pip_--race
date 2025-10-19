mod model;
mod config;

use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::{Error as WsError, error::ProtocolError};
use tokio::net::TcpListener;
use futures_util::{StreamExt, SinkExt};
use std::path::PathBuf;
use tokio::sync::broadcast;
use serde::Serialize;

#[derive(Serialize, Debug, Clone)]
struct TimerOut {
    t_call: f64,
    t_safe: f64,
    status: &'static str,
    lap_distance_m: f64,
    speed_kph: f64,
}

#[tokio::main]
async fn main() {
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8765".to_string());
    let listener = match TcpListener::bind(&bind_addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind TCP listener at {}: {}", bind_addr, e);
            return;
        }
    };
    eprintln!("🚀 Rust backend listening on ws://{}", bind_addr);

    // Broadcast channel to fan out computed results to all connected clients
    let (tx, _rx) = broadcast::channel::<TimerOut>(128);

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let tx_clone = tx.clone();
                let rx = tx.subscribe();
                tokio::spawn(handle_connection(stream, addr.to_string(), tx_clone, rx));
            }
            Err(e) => {
                eprintln!("Accept error: {}", e);
                // small delay to avoid tight loop in case of persistent errors
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        }
    }
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_addr: String,
    tx: broadcast::Sender<TimerOut>,
    mut rx: broadcast::Receiver<TimerOut>,
) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => {
            eprintln!("Client connected: {}", peer_addr);
            ws
        }
        Err(WsError::Protocol(ProtocolError::HandshakeIncomplete)) => {
            // Likely a TCP probe (e.g., healthcheck) that connected and closed without a WS handshake.
            // Suppress noisy logging.
            return;
        }
        Err(e) => {
            eprintln!("WebSocket handshake failed from {}: {}", peer_addr, e);
            return;
        }
    };
    let (mut write, mut read) = ws_stream.split();

    // Try to resolve config path robustly: prefer workspace-relative, then CWD.
    let cfg_path = resolve_config_path();
    let cfg = config::TimerConfig::load(cfg_path.to_str().unwrap());

    // Writer task: forwards broadcast messages to this websocket client
    let mut write_task = tokio::spawn(async move {
        while let Ok(out) = rx.recv().await {
            if let Ok(text) = serde_json::to_string(&out) {
                if write.send(tokio_tungstenite::tungstenite::Message::Text(text)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Reader loop: process incoming telemetry, compute, and broadcast
    while let Some(msg) = read.next().await {
        if let Ok(msg) = msg {
            if msg.is_text() {
                if let Ok(data) = serde_json::from_str::<model::TelemetryPacket>(&msg.to_string()) {
                    let (t_call, t_safe, status) = model::time_to_call(&data, &cfg);

                    let out = TimerOut {
                        t_call,
                        t_safe,
                        status,
                        lap_distance_m: data.lap_distance_m,
                        speed_kph: data.speed_kph,
                    };

                    // Log a compact line for observability
                    println!(
                        "lap={:.1}m speed={:.1}kph t_call={:.2}s t_safe={:.2}s status={}",
                        out.lap_distance_m, out.speed_kph, out.t_call, out.t_safe, out.status
                    );

                    // Ignore send errors (no subscribers)
                    let _ = tx.send(out);
                }
            }
        }
    }

    // Ensure writer task stops when reader ends
    write_task.abort();
}

fn resolve_config_path() -> PathBuf {
    // Common run path: project root: pip_--race/pit_timer_backend
    // Config lives at src/../tracks/monaco.json or tracks/monaco.json at root
    let candidates = [
        PathBuf::from("src/tracks/monaco.json"),
        PathBuf::from("tracks/monaco.json"),
        PathBuf::from("./tracks/monaco.json"),
        {
            let mut p = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
            p.pop(); // exe dir
            p.push("tracks/monaco.json");
            p
        },
    ];

    for c in candidates {
        if c.exists() {
            return c;
        }
    }

    // Fallback to default relative path; load() will error
    PathBuf::from("tracks/monaco.json")
}
