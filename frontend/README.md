# F1 Dashboard (React + Vite)

## Dev

- Install dependencies
- Start dev server

Commands (run from repo root on Windows PowerShell):

```
# install
npm install --prefix frontend

# run
npm run dev --prefix frontend
```

The app expects a WebSocket backend at `ws://localhost:8765`. Override via env:

- Create `.env` in `frontend/`:

```
VITE_BACKEND_WS=ws://localhost:8765
```

Then reload the dev server.

## Backend + feed (Docker)

The provided `docker-compose.yml` runs:
- Rust WebSocket backend on `ws://localhost:8765`
- Telemetry feed that streams FastF1 data to the backend

Run from repo root:

```
docker compose up --build
```

Open the frontend at the Vite URL (default http://localhost:5173).

## Notes
- The legacy `script.js` is deprecated; logic lives in React components now.
- Styles are in `frontend/style.css`.
- To change backend URL dynamically, use `VITE_BACKEND_WS`.
