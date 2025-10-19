use axum::{routing::post, extract::State, Json, http::StatusCode};
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

mod model;

// ---------- Request/Response types ----------

// FLAT request: driver, lap, PLUS all features as top-level keys
#[derive(Deserialize, Debug)]
struct IngestFlat {
    driver: String,
    lap: i32,
    #[serde(flatten)]
    features: HashMap<String, f32>,
}

// Response: only p2 and p3 as requested
#[derive(serde::Serialize)]
struct Out {
    t: i64,
    driver: String,
    lap: i32,
    p2: f32,
    p3: f32,
}

// ---------- Server state ----------

#[derive(Clone)]
struct AppState {
    mdl: Arc<model::Model>,
    feat_list: Arc<Vec<String>>, // authoritative input order
}

// ---------- Feature ordering utility ----------

fn order_from_flat(map: &HashMap<String, f32>, feat_list: &[String]) -> Vec<f32> {
    let mut v = Vec::with_capacity(feat_list.len());
    for k in feat_list {
        v.push(*map.get(k).unwrap_or(&0.0));
    }
    v
}

// ---------- Handler ----------

async fn ingest(
    State(state): State<AppState>,
    Json(payload): Json<IngestFlat>,
) -> Result<Json<Out>, (StatusCode, Json<serde_json::Value>)> {
    // Map incoming flat map -> ordered vector
    let vec = order_from_flat(&payload.features, &state.feat_list);

    // Debug signal so we can confirm we're not sending all-zeros
    if std::env::var("LOG_PRED").ok().as_deref() == Some("1") {
        let nz = vec.iter().filter(|x| **x != 0.0).count();
        let mean = if vec.is_empty() { 0.0 } else { vec.iter().sum::<f32>() / (vec.len() as f32) };
        let std = if vec.len() < 2 {
            0.0
        } else {
            let m = mean;
            (vec.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (vec.len() as f32)).sqrt()
        };
        let mut sample = vec![];
        for (i, name) in state.feat_list.iter().take(6).enumerate() {
            sample.push(format!("{}={:.3}", name, vec[i]));
        }
        tracing::info!(
            "recv driver={} lap={} in_dim={} nonzero={} mean={:.3} std={:.3} sample=[{}]",
            payload.driver, payload.lap, vec.len(), nz, mean, std, sample.join(", ")
        );
    }

    // Run inference (returns p2/p3)
    let (p2, p3) = state
        .mdl
        .predict_probs(&vec, state.feat_list.len())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))))?;

    let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
    Ok(Json(Out {
        t: now_ms,
        driver: payload.driver,
        lap: payload.lap,
        p2,
        p3,
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::var("MODEL_PATH").expect("MODEL_PATH not set");
    let meta_path  = std::env::var("META_PATH").expect("META_PATH not set");
    let port: u16  = std::env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8080);

    let (mdl, in_dim, feat_list) = model::Model::new(&model_path, &meta_path)?;
    if in_dim != feat_list.len() {
        tracing::warn!("meta.in_dim ({}) != feat_list.len() ({}); using feat_list.len()", in_dim, feat_list.len());
    }
    // Warmup to ensure JIT is happy
    let _ = mdl.predict_probs(&vec![0.0; feat_list.len()], feat_list.len())?;
    tracing::info!("warmup forward ok");

    tracing::info!("loaded model; feat_list[{}]: {:?}", feat_list.len(), &feat_list);

    let state = AppState {
        mdl: Arc::new(mdl),
        feat_list: Arc::new(feat_list),
    };

    let app = axum::Router::new()
        .route("/ingest", post(ingest))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
