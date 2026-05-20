# pip --race (Pressure Is Privilege) 

A real-time Formula 1 race strategy system combining pit stop probability prediction and optimal pit window timing for Williams Racing. The system integrates two sophisticated backend models with a live frontend dashboard to provide actionable race strategy insights.

---

## System Overview

This system provides real-time strategic intelligence during F1 races by answering two critical questions:

1. **Which rival cars are likely to pit soon?** (Rival-Boxing Predictor)
2. **When should we call our driver to pit?** (Pit-Timer)

The architecture consists of:
- **Two specialized backend models** (QR-DQN classifier + Rust pit timer)
- **Bridge service** for WebSocket communication
- **React frontend** for live visualization
- **FastF1 data ingestion pipeline**

---

## Model 1: Rival-Boxing Predictor (QR-DQN)

### Architecture

**Type**: Quantile Regression Deep Q-Network (QR-DQN)  
**Framework**: PyTorch  
**Purpose**: Predict probability that any driver will pit within next 2-3 laps

```
Input (26 features) 
    ↓
Linear(26 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 2 × 101)  [2 actions × 101 quantiles]
    ↓
Reshape → [Batch, 2 Actions, 101 Quantiles]
    ↓
Mean over quantiles → Q(NO_PIT), Q(PIT)
    ↓
Gap = Q(PIT) - Q(NO_PIT)
    ↓
Platt Scaling: σ(coef × gap + intercept)
    ↓
Output: P(box within 2 laps)
```

**Network Dimensions:**
- **Input**: 26 features (hazard + tactical state)
- **Hidden layers**: 2 × 256 neurons with ReLU activation
- **Output**: 2 actions × 101 quantile estimates
- **Parameters**: ~200K trainable weights

### Input Features (26 dimensions)

**Hazard Features (14)** — tire degradation signals:
- `tire_age_laps`: Current tire age in laps
- `stint_no`: Which stint (1st, 2nd, 3rd...)
- `compound_SOFT`, `compound_MED`, `compound_HARD`, `compound_INTERMEDIATE`, `compound_WET`: One-hot tire type
- `last3_avg`: Average lap time over last 3 laps
- `last5_slope`: Linear regression slope of last 5 lap times (degradation rate)
- `last3_var`: Variance in last 3 lap times (consistency)
- `typical_stint_len`: Track-specific median stint length for this compound
- `age_vs_typical`: Difference from typical stint length
- `age_percentile`: Normalized age (0-1 scale relative to typical)
- `overshoot`: How many laps past typical stint length (clipped at 0)

**Tactical Features (12)** — strategic environment:
- `cheap_stop_flag`: Is track under yellow/VSC/SC (cheap pit opportunity)?
- `cheap_prev1`, `cheap_prev2`: Cheap flags in previous 1-2 laps
- `non_green_runlen`: Consecutive laps under non-green conditions
- `pits_prev1`, `pits_prev2`: Number of grid-wide pits in previous 1-2 laps
- `tire_age_laps` (duplicated): Used twice in feature vector for emphasis
- `compound_*` (duplicated × 5): Tire type one-hot repeated

### Mathematical Formula

#### Quantile Huber Loss (Training Objective)

For each state-action pair (s, a), the model learns a distribution of Q-values via quantiles:

```
τᵢ = (i + 0.5) / N_quantiles,  i ∈ {0, 1, ..., 100}

u = TZ - Zθ(s,a)    [TD error for each quantile]

ρᵏ(u) = {  0.5 u²                if |u| ≤ κ
        {  κ(|u| - 0.5κ)         otherwise     [Huber loss]

ℒ_QR = 𝔼[ Σᵢ |τᵢ - 𝟙(u < 0)| · ρᵏ(uᵢ) ]
```

Where:
- `TZ = r + γ² · Zθ'(s', argmax_a' Q̄(s',a'))` — target quantiles (2-step bootstrapped)
- `Zθ(s,a)` — predicted quantiles for action a
- `κ = 1.0` — Huber threshold
- `γ = 0.98` — discount factor

#### Reward Shaping

```
r(s, a, outcome) = {
    +2.0 × boost     if a = PIT and actual_pit_within_2_laps = True
    -0.05            if a = PIT and actual_pit_within_2_laps = False
     0.0             if a = NO_PIT
}

boost = {  1.3   if cheap_stop_flag = 1  (yellow/VSC/SC)
        {  1.0   otherwise
```

**Rationale**: 
- Heavily reward correct pit predictions (+2.0 base, +2.6 during yellows)
- Small penalty for false alarms to encourage precision
- Zero reward for NO_PIT to focus learning on pit decisions

#### Probability Calibration (Platt Scaling)

After training, raw Q-gap scores are calibrated to probabilities:

```
score(s) = Q̄_PIT(s) - Q̄_NO_PIT(s)    [mean over 101 quantiles]

P(box | s) = σ(w · score + b) = 1 / (1 + exp(-(w · score + b)))
```

Where `(w, b)` are fitted via logistic regression on validation set to maximize F2 score (recall-biased).

#### 3-Lap Horizon (Heuristic Extension)

```
P(box within 3 laps | s_t) = 1 - (1 - p₂(t)) × (1 - p₂(t+1))
```

Where:
- `p₂(t)` = calibrated 2-lap probability at current state
- `p₂(t+1)` = calibrated 2-lap probability at next-lap features

### Output

**Primary**: `p2` — Probability driver will pit within next 2 laps (0.0 to 1.0)  
**Secondary**: `p3` — Probability driver will pit within next 3 laps (0.0 to 1.0)

**Example Output**:
```json
{
  "driver": "VER",
  "lap": 45,
  "p2": 0.87,
  "p3": 0.92,
  "t": 1730000000000
}
```

### Tech Stack

- **Training**: Python 3.11, PyTorch 2.x, FastF1 3.x
- **Inference Server**: Rust (Axum web framework, `tch-rs` for PyTorch JIT)
- **Model Format**: TorchScript (`qrdqn_torchscript.pt`)
- **Data**: Monaco 2023 race (primary training set)
- **Artifacts**: 
  - `artifacts/rl/qrdqn.pt` — full checkpoint
  - `artifacts/rl/qrdqn_torchscript.pt` — production model
  - `artifacts/rl/meta.json` — feature list & hyperparameters
  - `artifacts/rl/calib_platt.json` — Platt scaling coefficients

---

## ⏱️ Model 2: Pit-Timer Backend

### Architecture

**Type**: Real-time physics-based state machine  
**Framework**: Rust (Tokio async runtime)  
**Purpose**: Compute optimal pit call timing based on track position and speed

```
Telemetry Stream (WebSocket)
    ↓
{lap_distance_m, speed_kph, speed_profile}
    ↓
Load Track Config (pit_entry_m, call_offset_m, buffer_s)
    ↓
Compute: d_rem = max(0, (pit_entry_m - call_offset_m) - lap_distance_m)
    ↓
Integrate: t_call = ∫[x_current to x_call] (1/v(x)) dx
    ↓
Apply Buffer: t_safe = t_call - buffer_s
    ↓
Status Logic:
  if t_safe < 0        → LOCKED_OUT
  elif t_safe < 2s     → RED (PIT NOW!)
  elif t_safe < 5s     → AMBER (PREPARE)
  else                 → GREEN (CLEAR TO PIT)
    ↓
Broadcast {t_call, t_safe, status, lap_distance_m, speed_kph}
```

### Input

**Telemetry Packet** (JSON via WebSocket):
```json
{
  "lap_distance_m": 2450.5,
  "speed_kph": 187.3,
  "speed_profile": [
    {"x_m": 2400.0, "v_mps": 51.2},
    {"x_m": 2425.0, "v_mps": 52.1},
    {"x_m": 2450.0, "v_mps": 52.0}
  ]
}
```

**Track Configuration** (Monaco example):
```json
{
  "pit_entry_m": 2700.0,
  "call_offset_m": 180.0,
  "buffer_s": 0.8
}
```

### Mathematical Formula

#### Core Timing Calculation

```
x_current = lap_distance_m                    [current position on lap]
x_call = pit_entry_m - call_offset_m          [optimal call point]
d_rem = max(0, x_call - x_current)            [distance remaining to call point]
```

#### Time Integration (Trapezoidal Rule)

Instead of naive `t = d/v`, integrate over variable speed profile:

```
t_call = ∫[x_current to x_call] (1 / v(x)) dx

Discretized (trapezoidal):
t_call ≈ Σᵢ (xᵢ₊₁ - xᵢ) · 0.5 · (1/vᵢ + 1/vᵢ₊₁)
```

Where:
- `v(x)` is interpolated from `speed_profile` samples
- Handles acceleration/deceleration zones more accurately
- Falls back to instantaneous speed if no profile available: `t_call = d_rem / v_inst`

#### Safety Buffer

```
t_safe = t_call - buffer_s
```

**buffer_s** = radio transmission delay + driver reaction time (typically 0.8-1.5s)

#### Status Thresholds

```
status = {
    "LOCKED_OUT"   if t_safe < 0      [too late, missed the window]
    "RED"          if 0 ≤ t_safe < 2  [critical: pit NOW!]
    "AMBER"        if 2 ≤ t_safe < 5  [prepare: get ready to pit]
    "GREEN"        if t_safe ≥ 5      [clear: optimal window open]
}
```

### Output

**Real-time Broadcast** (JSON via WebSocket):
```json
{
  "t_call": 3.85,
  "t_safe": 3.05,
  "status": "AMBER",
  "lap_distance_m": 2450.5,
  "speed_kph": 187.3
}
```

- `t_call` — seconds until optimal radio call point
- `t_safe` — seconds until latest safe call (with buffer)
- `status` — GREEN | AMBER | RED | LOCKED_OUT
- Telemetry echo for frontend display

### Tech Stack

- **Language**: Rust 2021 edition
- **Async Runtime**: Tokio 1.39
- **WebSocket**: `tokio-tungstenite` 0.23
- **Serialization**: `serde` + `serde_json`
- **Concurrency**: `broadcast` channel for fanout to multiple clients
- **Config**: JSON track profiles (`tracks/monaco.json`)

---

## 🌐 System Integration

### Data Flow

```
FastF1 Cache (Monaco 2023)
    ↓
feeder_fastf1_cache.py  [computes 26 features per lap]
    ↓
    ├──→ HTTP POST: rt_predictor (Rust) → {p2, p3}
    │        ↓
    │    bridge-service (Node.js) → WebSocket fanout
    │        ↓
    └──→ Frontend: PitProbabilities component
    
Telemetry Stream (FastF1 simulated 5Hz)
    ↓
telemetry_feed.py → speed_profile_calculator
    ↓
WebSocket: pit_timer_backend (Rust) → {t_call, t_safe, status}
    ↓
Frontend: BoxWindow component
```

### Bridge Service (Node.js)

**Purpose**: Decouple feeder → predictor from frontend WebSocket clients

```javascript
// Receives from feeder via HTTP POST
app.post('/update', (req, res) => {
  const { driver, lap, p2, p3, t } = req.body;
  predictions.set(driver, { driver, lap, p2, p3, timestamp: t });
  broadcast({ type: 'UPDATE', prediction: { driver, lap, p2, p3 } });
});

// Broadcasts to all WebSocket clients
wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ 
    type: 'FULL_STATE', 
    probabilities: Array.from(predictions.values()) 
  }));
});
```

**Why?** 
- Feeder runs at lap-rate (~80-100 seconds per update)
- Predictor is stateless HTTP service
- Frontend needs persistent WebSocket connection with immediate updates

### Frontend Components

**React SPA** with 3 pages:

1. **Live Race** (`/`) — Williams car telemetry cards
2. **Strategy** (`/strategy`) — Combined pit probabilities + box window timer
3. **About** (`/about`) — Team information

**Key Hooks**:
- `useWebSocket(timerUrl)` — connects to pit_timer_backend
- `usePitProbabilities(bridgeUrl)` — connects to bridge-service
- `useRouter()` — hash-based routing

---

## Performance Metrics

### Rival-Boxing Predictor (QR-DQN)

**Validation Set (Monaco 2023, last 20% of race)**:
- **AUC-ROC**: 0.87 (proxy reward labels)
- **Average Precision**: 0.82
- **Optimal Threshold**: 0.65 (F2-optimized)
- **Precision**: 0.71 @ 0.65 threshold
- **Recall**: 0.84 @ 0.65 threshold
- **F2 Score**: 0.81 (recall-weighted)

**Production Latency**:
- Inference: ~2-5ms per driver (CPU)
- End-to-end (feeder → frontend): <50ms

### Pit-Timer Backend

**Timing Accuracy**:
- Integration error vs. naive `d/v`: ±0.1-0.3s improvement in variable-speed zones
- Status updates: 30ms frontend smoothing (countdown timer)

**Throughput**:
- Handles 5Hz telemetry stream per car
- Broadcast fanout: 10+ concurrent WebSocket clients
- Latency: <10ms per packet

---

## 🚀 Usage

### Training the QR-DQN Model

```bash
cd rival-boxing
python trainer/train_qrdqn.py \
  --races "2023:Monaco" \
  --epochs 20 \
  --batch_size 512 \
  --n_quantiles 101 \
  --hidden 256 \
  --oversample_pos 8 \
  --pos_reward 2.0 \
  --cheap_boost 1.3 \
  --target_recall 0.8
```

### Exporting to TorchScript

```bash
python scripts/export_torchscript_26.py \
  --ckpt artifacts/rl/qrdqn.pt \
  --meta artifacts/rl/meta.json \
  --out_ts artifacts/rl/qrdqn_torchscript.pt \
  --update_meta
```

### Running Inference Server

```bash
cd rival-boxing/rt_predictor
export MODEL_PATH=../artifacts/rl/qrdqn_torchscript.pt
export META_PATH=../artifacts/rl/meta.json
export PORT=8080
cargo run --release
```

### Starting Bridge Service

```bash
cd bridge-service
npm install
npm start  # Runs on port 8081
```

### Running Pit Timer Backend

```bash
cd pit_timer_backend
cargo run --release  # Runs on port 8765
```

### Streaming Telemetry

```bash
cd telemetry_feed
pip install -r requirements.txt
export BACKEND_URL=ws://localhost:8765
python telemetry_feed.py
```

### Feeding Predictions

```bash
cd rival-boxing/ingest
python feeder_fastf1_cache.py \
  --race "2023:Monaco" \
  --cache ../data/fastf1_cache \
  --url http://localhost:8080/ingest \
  --bridge http://localhost:8081/update \
  --meta ../artifacts/rl/meta.json \
  --sleep 0.1
```

### Running Frontend

```bash
cd frontend
npm install
npm run dev  # Vite dev server on http://localhost:5173
```

**Environment Variables** (`.env`):
```
VITE_PIT_TIMER_WS=ws://localhost:8765
VITE_PROBS_WS=ws://localhost:8081
```

---

## Key Design Decisions

### Why QR-DQN over Classification?

1. **Distributional RL**: Captures uncertainty in pit timing (quantiles vs. point estimate)
2. **Offline Learning**: Learns from historical race data without live interaction
3. **Reward Shaping**: Explicitly values cheap pit opportunities (yellows/VSC)
4. **Calibration**: Platt scaling maps Q-gaps to well-calibrated probabilities
5. **Performance Metric**: Less false negatives to gain an edge. Focus on Recall and AUC.

### Why Separate Pit Timer Backend?

1. **Decoupling**: Predictor (lap-rate updates) vs. timer (high-frequency position updates)
2. **Physics-based**: Integration over speed profiles for accuracy in acceleration zones
3. **Low Latency**: Rust async processing for <10ms broadcast cycles
4. **Track-specific**: Configurable per circuit (pit entry points, call offsets)

### Why Bridge Service?

1. **Protocol Translation**: HTTP (feeder) → WebSocket (frontend)
2. **State Management**: Caches latest predictions for new client connections
3. **Fanout**: Broadcasts to multiple dashboard instances without predictor load

---
## Deployment
**Vercel** . Deployed on __pip-race.vercel.app__ . Backend server exists locally. Future work involves AWS/Convex Integration for Cloud based servers.

---

## 🛠️ Dependencies

### Python (Training/Ingestion)
```
torch>=2.0
fastf1>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
requests>=2.31
```

### Rust (Inference/Timer)
```
axum = 0.7         # Web framework
tokio = 1.39       # Async runtime
tch = 0.15         # PyTorch bindings
serde = 1.0        # Serialization
tokio-tungstenite = 0.23  # WebSocket
```

### Node.js (Bridge)
```
express = 4.x
ws = 8.x           # WebSocket server
```

### Frontend
```
react = 18.x
vite = 5.x         # Build tool
```

---

## Future Enhancements

1. **Multi-Race Training**: Expand beyond Monaco to generalize across circuits
2. **Tire Compound Dynamics**: Integrate F1 tire model for degradation curves
3. **Weather Integration**: Add rain probability to tactical features
4. **Opponent Modeling**: Predict rival team strategies (undercut/overcut timing)
5. **DRS Detection**: Factor in DRS zones for pit loss calculations
6. **Real-Time Tuning**: Adaptive thresholds based on race phase (first stint vs. last 10 laps)
7. USE IT WITHIN ACTUAL **HPCs** (HIGH PERFORMANCE COMPUTING)

---

## License & Attribution

**Data Source**: FastF1 (https://github.com/theOehrly/Fast-F1)  
**Models**: Custom implementations for Williams Racing strategy optimization  
**Disclaimer**: This is a demonstration project for educational purposes. Not affiliated with Williams Racing or Formula 1.

---

**PRESSURE IS PRIVILEGE! 🏁**
