# QA Engine — Stage A Starter (backend skeleton)

Walking-skeleton backend: full LangGraph wiring with node **stubs** and live SSE
trajectory streaming. The stubs take the happy path so the skeleton runs
end-to-end; later stages swap in real agent logic without changing the contracts.

## Run the smoke test
```bash
cd backend
pip install -r requirements.txt
pytest -v
```

## Run the server + watch the stream
```bash
cd backend
cp .env.example .env          # add your ANTHROPIC_API_KEY (not needed for stubs)
uvicorn app.main:app --reload

# in another terminal:
RUN=$(curl -s -X POST localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"raw_input":"As a user I want to log in","target_url":"http://localhost:3000"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
curl -N localhost:8000/runs/$RUN/stream
```
You should see ten stub events stream by, ending with a `complete` event.

## The two frozen contracts
- `app/graph/state.py` — `QAState`
- `app/streaming/events.py` — `TrajectoryEvent`

Do not change these once you start Stage B; every node and the frontend depend on them.

## What's stubbed (and where Stage B+ takes over)
| Node | File | Becomes real in |
|---|---|---|
| Context / Guardrail / INVEST | `app/graph/nodes/stubs.py` | Stage B |
| Analyst / UI Mapper | same | Stage D |
| Scaffolder / Critic | same | Stage E |
| Execution / Self-Heal | same | Stage F |
| Synthesis | same | Stage G |

In Stage B, split `stubs.py` into one file per node under `app/graph/nodes/`
and implement the real `invest_node` (which may set `status="paused_hitl"` and
interrupt). The conditional edges in `builder.py` are already wired for HITL, the
reflection loop, and self-heal — you only replace node bodies.
