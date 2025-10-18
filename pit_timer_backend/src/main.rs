mod model;
mod config;

use tokio_tungstenite::accept_async;
use tokio::net::TcpListener;
use futures_util::StreamExt;
use std::path::PathBuf;

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

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                tokio::spawn(handle_connection(stream));
            }
            Err(e) => {
                eprintln!("Accept error: {}", e);
                // small delay to avoid tight loop in case of persistent errors
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        }
    }
}

async fn handle_connection(stream: tokio::net::TcpStream) {
    let ws_stream = accept_async(stream).await.unwrap();
    let (_, mut read) = ws_stream.split();

    // Try to resolve config path robustly: prefer workspace-relative, then CWD.
    let cfg_path = resolve_config_path();
    let cfg = config::TimerConfig::load(cfg_path.to_str().unwrap());

    while let Some(msg) = read.next().await {
        if let Ok(msg) = msg {
            if msg.is_text() {
                if let Ok(data) = serde_json::from_str::<model::TelemetryPacket>(&msg.to_string()) {
                    let (t_call, t_safe, status) = model::time_to_call(&data, &cfg);
                    println!(
                        "[ lap ] dist={:.1}m  speed={:.1}kph  t_call={:.2}s  t_safe={:.2}s  STATUS={}",
                        data.lap_distance_m, data.speed_kph, t_call, t_safe, status
                    );
                }
            }
        }
    }
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
