## Pit-timer backend

`cd telemetry_feed
BACKEND_URL=ws://localhost:8765 python telemetry_feed.py`
(Local Setup)
Listening to localhost/8080

## Rival-Driver Box Classifier
`python feeder_fastf1_cache.py \
  --race "2023:Monaco" \
  --cache ../data/fastf1_cache \
  --meta ../artifacts/rl/meta.json \
  --url http://localhost:8080/ingest \
  --bridge http://localhost:8081/update \
  --sleep 0.1 \
  --echo`
  (Local Setup)
  Listening to localhost/2716

  ## Frontend
  `npm build`
  `npm run dev`
  (Local Setup)
  

  
