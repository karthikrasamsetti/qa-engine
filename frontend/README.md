# QA Engine — Frontend (Stage C)

Three-panel live dashboard that consumes the backend SSE trajectory stream.

## Run it
```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```
The backend must be running on `:8000` (Vite proxies `/runs` and `/health` to it,
so EventSource works same-origin — no CORS issues in dev).

## Layout
- **Input panel (left):** user story / Jira ID + target URL + Run.
- **Trajectory panel (center):** live console. Each event is timestamped, tagged
  by type (think / act / tool / result / decide / error / human / done) and
  colored per agent so handoffs are legible. When the backend emits an
  `hitl_request`, an inline answer box appears that calls `/resume`.
- **Output panel (right):** tabs for the Playwright script (fills at Phase 3) and
  the report (fills at Phase 5). Until then it shows the latest INVEST verdict.

## Signature element
The phase rail under the header fills 1→5 as the run advances, with the active
phase pulsing — a glanceable progress instrument.

## Contract sync
`src/lib/api.ts` mirrors the backend's `TrajectoryEvent` shape and the run
endpoints. If you change `app/streaming/events.py` on the backend, update the
`TrajectoryEvent` interface here to match. Note the resume field is `response`.

## What's wired vs. waiting
Works now against Phase 1: streaming, per-agent coloring, HITL answer/resume,
status pill, live cost readout (from `/runs/{id}/cost`), phase rail.
Code/report panels are built but stay empty until Phases 3 & 5 populate
`data.script` and `data.report` in the event stream.
