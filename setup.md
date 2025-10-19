## Pit-timer backend

`cd telemetry_feed
BACKEND_URL=ws://localhost:8765 python telemetry_feed.py`
(Local Setup)
Listening to localhost/8080

## Rival-Driver Box Classifier
`python feeder_fastf1_cache.py \` <br>
`  --race "2023:Monaco" \` <br>
` --cache ../data/fastf1_cache \` <br>
`  --meta ../artifacts/rl/meta.json \`<br>
`  --url http://localhost:8080/ingest \`<br>
`  --bridge http://localhost:8081/update \`<br>
`  --sleep 0.1 \`<br>
`  --echo`<br>
  (Local Setup)
  Listening to localhost/2716

  ## Frontend
  `npm build`
  `npm run dev`
  (Local Setup)
  

  
