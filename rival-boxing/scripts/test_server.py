#!/usr/bin/env python3
from flask import Flask, request, jsonify
import torch
import json
import sys
import os
from datetime import datetime

app = Flask(__name__)

# Load model
try:
    model = torch.jit.load('artifacts/rl/qrdqn_torchscript.pt')
    model.eval()
    print("✓ Model loaded: artifacts/rl/qrdqn_torchscript.pt")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Load meta
try:
    meta = json.load(open('artifacts/rl/meta.json'))
    feat_list = meta['feat_list']
    print(f"✓ Loaded {len(feat_list)} features from meta.json")
    
    if os.getenv('LOG_PRED') == '1':
        print("\n📋 Feature list:")
        for i, f in enumerate(feat_list):
            print(f"   [{i:2d}] {f}")
        print()
except Exception as e:
    print(f"❌ Failed to load meta: {e}")
    sys.exit(1)

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.json
    driver = data.get('driver', 'UNK')
    lap = data.get('lap', 0)
    
    # Extract features in the correct order
    vec = [float(data.get(k, 0.0)) for k in feat_list]
    
    # Check for all zeros (common bug)
    nonzero = sum(1 for v in vec if v != 0.0)
    if nonzero == 0:
        print(f"⚠️  All-zero input for {driver} lap {lap}")
    
    # Run inference
    with torch.no_grad():
        input_t = torch.tensor([vec], dtype=torch.float32)
        output = model(input_t)  # [1, 2, 101]
        
        # Mean over quantiles -> [1, 2]
        q_mean = output.mean(dim=-1)
        
        # Gap between Q(pit) and Q(no_pit)
        gap = q_mean[0, 1] - q_mean[0, 0]
        
        # Map gap to probabilities via sigmoid
        p2 = float(torch.sigmoid(gap))
        p3 = float(torch.sigmoid(gap * 1.25))
    
    # Detailed logging if enabled
    if os.getenv('LOG_PRED') == '1':
        sample = [f"{feat_list[i]}={vec[i]:.1f}" for i in range(min(6, len(vec)))]
        print(f"📥 {driver:3s} L{lap:2d}: nz={nonzero:2d}/{len(vec)} [{', '.join(sample)}...] | 📤 p2={p2:.3f} p3={p3:.3f}")
    
    return jsonify({
        't': int(datetime.now().timestamp() * 1000),
        'driver': driver,
        'lap': lap,
        'p2': p2,
        'p3': p3
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 RT Predictor Test Server")
    print("="*70)
    print(f"   Listening on: http://0.0.0.0:8080")
    print(f"   Model: artifacts/rl/qrdqn_torchscript.pt")
    print(f"   Features: {len(feat_list)}")
    print(f"\n💡 Tip: Set LOG_PRED=1 for detailed request/response logs")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
