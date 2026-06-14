# Intelligent QA Engine — Claude Code Implementation Plan

A staged build plan for a multi-agent QA automation platform. Hand each stage to
Claude Code as a separate session/prompt. Do **not** ask Claude Code to build all
five phases at once — the streaming + LangGraph state integration is the riskiest
part and is much easier to debug in isolation.

---

## 0. Strategy

**Approach: Walking skeleton, then expand.**

Build the full repo scaffold + one fully working vertical slice (Phase 1 with live
SSE streaming to the UI). Once thoughts stream end-to-end from a LangGraph node to
the React terminal panel, every later phase is just "add another node" — the hard
plumbing is already proven.

**Build order**
1. Stage A — Monorepo scaffold + LangGraph state + node stubs + SSE plumbing
2. Stage B — Phase 1 working slice (Context → Guardrail → INVEST → HITL) streaming live
3. Stage C — Frontend: 3-panel dashboard consuming the real SSE stream
4. Stage D — Phase 2 (Requirements Analyst + UI Mapper via Browser Use)
5. Stage E — Phase 3 (Scaffolder ↔ Critic reflection loop, Playwright output)
6. Stage F — Phase 4 (Execution in Docker + self-healing loop)
7. Stage G — Phase 5 (Synthesis report + Vector DB memory / RAG)
8. Stage H — Governance: evalset, LLM-as-judge, audit logging, model routing

Each stage below is written so you can paste it more or less directly.

---

## Target Architecture

```
NLP user story / Jira ID
        |
   [Context Agent] --MCP--> Jira (epics, history)
        |
   [Guardrail] (prompt-injection sanitize)
        |
   [INVEST Reviewer] --vague?--> [HITL pause] --human--> resume
        |  (valid)
   [Requirements Analyst] (prompt-chained test plan)
        |
   [UI Mapper] --Browser Use--> DOM -> CSS/XPath locators
        |
   [Scaffolder] <--reflection loop--> [Critic]   (Playwright script)
        |  (approved)
   [Execution Agent] --Docker sandbox--> run
        |  fail (ElementNotFound)
        +--> [Self-Heal] re-scan DOM, fix locator, retry (N times) -> HITL
        |  pass
   [Synthesis Agent] -> failure-vs-bug report
        |
   [Memory] -> Vector DB (scripts, locators, embeddings) for RAG
```

State carried through every node is a single `QAState` TypedDict. Routing decisions
(vague story, critic rejects, test fails) are conditional edges in LangGraph.

---

## Tech Stack (pin these for Claude Code)

| Layer | Choice |
|---|---|
| Orchestration | Python 3.11+, LangGraph |
| LLM | Anthropic API — Sonnet for reasoning, Haiku for parsing/extraction |
| Streaming | FastAPI + SSE (`sse-starlette`), LangGraph `astream_events` |
| Frontend | React + Vite + TypeScript + Tailwind + shadcn/ui |
| Browser/UI | `browser-use` library + Playwright |
| Vector DB | Chroma (local-first; swap to Weaviate later) |
| Integration | MCP client for Jira |
| Sandbox | Docker SDK for Python |
| Eval | Custom evalset + LLM-as-judge script |

---

## Repo Layout

```
qa-engine/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + SSE endpoints
│   │   ├── config.py               # pydantic-settings, model routing config
│   │   ├── graph/
│   │   │   ├── state.py            # QAState TypedDict
│   │   │   ├── builder.py          # LangGraph wiring + conditional edges
│   │   │   └── nodes/
│   │   │       ├── context.py
│   │   │       ├── guardrail.py
│   │   │       ├── invest.py
│   │   │       ├── analyst.py
│   │   │       ├── ui_mapper.py
│   │   │       ├── scaffolder.py
│   │   │       ├── critic.py
│   │   │       ├── execution.py
│   │   │       ├── self_heal.py
│   │   │       └── synthesis.py
│   │   ├── llm/
│   │   │   ├── client.py           # Anthropic wrapper + model router
│   │   │   └── prompts/            # one file per agent
│   │   ├── tools/
│   │   │   ├── mcp_jira.py
│   │   │   ├── browser.py          # browser-use wrapper
│   │   │   └── sandbox.py          # Docker run wrapper
│   │   ├── memory/
│   │   │   └── vector_store.py     # Chroma RAG
│   │   ├── streaming/
│   │   │   └── events.py           # trajectory event schema + emitter
│   │   ├── hitl/
│   │   │   └── store.py            # pause/resume checkpoint store
│   │   └── observability/
│   │       └── audit.py            # structured audit log
│   ├── tests/
│   ├── evals/
│   │   ├── evalset.jsonl
│   │   └── judge.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml          # backend + chroma + sandbox runner
└── frontend/
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   │   ├── InputPanel.tsx
    │   │   ├── TrajectoryPanel.tsx # SSE terminal/timeline
    │   │   └── OutputPanel.tsx     # split: code | report
    │   ├── hooks/useTrajectoryStream.ts
    │   └── lib/api.ts
    ├── package.json
    └── vite.config.ts
```

---

## The Contract Everything Depends On — define these FIRST

Tell Claude Code these two schemas are the spine; nothing else gets written until
they exist and are agreed.

**`QAState` (backend/app/graph/state.py)**
```python
from typing import TypedDict, Literal, Optional
from typing_extensions import Annotated

class QAState(TypedDict, total=False):
    run_id: str
    raw_input: str                 # user story or Jira ID
    jira_context: dict             # epics, history from MCP
    sanitized_story: str
    invest_verdict: dict           # {passed: bool, scores: {...}, gaps: [...]}
    hitl_request: Optional[dict]   # populated when paused for a human
    hitl_response: Optional[str]
    test_plan: list[dict]          # step-by-step logical plan
    locators: dict                 # step_id -> {css, xpath, confidence}
    script: str                    # Playwright code
    critic_feedback: list[dict]
    critic_approved: bool
    reflection_count: int
    execution_result: dict         # {passed, logs, error, screenshots}
    heal_attempts: int
    report: dict                   # synthesis output
    status: Literal["running","paused_hitl","done","failed"]
```

**Trajectory event (backend/app/streaming/events.py)** — the single shape the SSE
stream emits and the frontend renders:
```python
{
  "run_id": str,
  "ts": float,
  "agent": str,            # "INVEST Reviewer"
  "phase": int,            # 1..5
  "type": "thought" | "action" | "tool_call" | "tool_result"
          | "decision" | "error" | "hitl_request" | "complete",
  "message": str,          # human-readable line for the terminal
  "data": dict             # optional structured payload
}
```

Every node emits these as it runs (via `astream_events` custom events or a shared
async emitter). The frontend just appends `message` lines and styles by `type`.

---

# STAGE A — Scaffold + State + SSE plumbing

> **Paste to Claude Code:**
>
> Set up a monorepo named `qa-engine` per the layout below. Create `backend/`
> (Python 3.11, FastAPI) and `frontend/` (React + Vite + TS + Tailwind + shadcn).
>
> In the backend:
> 1. Write `app/graph/state.py` with the `QAState` TypedDict exactly as specified.
> 2. Write `app/streaming/events.py`: a `TrajectoryEvent` pydantic model matching
>    the schema, plus an async `EventEmitter` that pushes events onto an
>    `asyncio.Queue` keyed by `run_id`.
> 3. Write `app/graph/nodes/` with **stub** functions for all ten nodes. Each stub
>    emits one `thought` event ("X agent starting…") and returns the state
>    unchanged for now.
> 4. Write `app/graph/builder.py` that wires the nodes into a LangGraph
>    `StateGraph` with the conditional edges shown in the architecture (INVEST→HITL,
>    Critic loop, Execution→self-heal). Use `MemorySaver` checkpointer.
> 5. Write `app/main.py`: a `POST /runs` that starts a graph run in the background
>    and a `GET /runs/{run_id}/stream` SSE endpoint (use `sse-starlette`) that
>    drains the emitter queue for that run_id.
> 6. `requirements.txt`, `config.py` (pydantic-settings, reads `ANTHROPIC_API_KEY`,
>    model-routing map), `Dockerfile`, `docker-compose.yml` (backend + chroma).
>
> Add type hints, docstrings, structured logging, and a pytest that runs the stub
> graph and asserts events were emitted. Do not implement real agent logic yet.
> Confirm `curl`-ing the stream prints stub events before stopping.

**Done when:** `POST /runs` → `GET .../stream` prints ten stub events in order.

---

# STAGE B — Phase 1 working slice

> **Paste to Claude Code:**
>
> Implement real logic for Phase 1 nodes only: `context`, `guardrail`, `invest`.
>
> - `context.py`: MCP client (`tools/mcp_jira.py`). If input is a Jira ID, fetch
>   story + epics + recent related tickets; if it's raw text, pass through. Stub
>   the MCP server connection behind an interface so it works with a fake Jira
>   fixture when no real server is configured.
> - `guardrail.py`: sanitize the story against prompt injection (strip/he-flag
>   instruction-like content, delimiter-wrap untrusted text). Emit a `decision`
>   event noting what was flagged.
> - `invest.py`: call **Sonnet** with the INVEST prompt; return per-principle
>   scores + gaps. If `passed=False`, set `status="paused_hitl"`, populate
>   `hitl_request`, emit an `hitl_request` event, and interrupt the graph.
> - Add `POST /runs/{run_id}/resume` that injects `hitl_response` and resumes from
>   the checkpoint.
> - Real Anthropic calls through `llm/client.py` with the model router.
>
> Add unit tests with a vague story (must trigger HITL) and a good story (must
> pass through to the next stub).

**Done when:** a vague story pauses and waits; resuming with clarification
continues; a good story flows straight through. All visible as live events.

---

# STAGE C — Frontend dashboard

> **Paste to Claude Code:**
>
> Build the three-panel dashboard. Use shadcn/ui, Tailwind, a dark terminal
> aesthetic for the trajectory panel.
>
> - `InputPanel`: textarea for user story / Jira ID + a Run button → `POST /runs`.
> - `useTrajectoryStream.ts`: opens `EventSource` on `/runs/{id}/stream`, parses
>   `TrajectoryEvent`s into state.
> - `TrajectoryPanel`: live timeline/console. Color + icon per event `type`
>   (thought = dim, tool_call = blue, decision = amber, error = red,
>   hitl_request = highlighted with an inline "Answer" box that calls `/resume`,
>   complete = green). Auto-scroll, monospace, typing-cadence feel.
> - `OutputPanel`: split view — left a code block (syntax-highlighted, empty until
>   Phase 3), right the report (empty until Phase 5).
> - Graceful reconnect on SSE drop.
>
> Make it genuinely demo-polished: header, status pill (running/paused/done),
> subtle animations on new events.

**Done when:** you type a story, hit Run, and watch Phase 1 think in real time,
answer a HITL prompt inline, and see it resume.

---

# STAGE D — Phase 2 (Analyst + UI Mapper)

> **Paste to Claude Code:**
>
> Implement `analyst.py` and `ui_mapper.py`.
>
> - `analyst.py`: prompt-chain the validated story into a structured `test_plan`
>   (list of `{step_id, intent, action, expected}`). Use Sonnet.
> - `ui_mapper.py`: wrap the `browser-use` library in `tools/browser.py`. For each
>   step, navigate the target app, read the DOM, and produce `{css, xpath,
>   confidence}` locators. Use **Haiku** for the bulk DOM/extraction calls. Emit a
>   `tool_call`/`tool_result` event per navigation. Accept a `target_url` on the
>   run request.
>
> Tests: feed a known test page (e.g. a simple local login form fixture) and
> assert locators resolve.

**Done when:** a story produces a step plan and real locators for a live target URL,
streaming each DOM scan.

---

# STAGE E — Phase 3 (Scaffolder ↔ Critic reflection loop)

> **Paste to Claude Code:**
>
> Implement `scaffolder.py` (Producer) and `critic.py` (Critic) as a reflection
> loop targeting **Playwright** (Python). Scaffolder turns plan + locators into a
> runnable Playwright script; Critic (Sonnet) reviews for missing assertions,
> logic flaws, syntax. Loop via conditional edge until `critic_approved` or
> `reflection_count` hits a cap (e.g. 3), then proceed with best effort + a warning
> event. Emit each critique as `decision` events so the loop is visible in the UI.
> Populate the OutputPanel code view as the script evolves.

**Done when:** the loop visibly iterates and emits a final approved Playwright script.

---

# STAGE F — Phase 4 (Execution + self-healing)

> **Paste to Claude Code:**
>
> Implement `execution.py` and `self_heal.py` with `tools/sandbox.py`.
>
> - `sandbox.py`: run the generated Playwright script inside a Docker container
>   (mount script read-only, no host network beyond target, capture stdout/stderr,
>   screenshots, exit code). Enforce least privilege + timeout.
> - `execution.py`: run, parse pass/fail, capture artifacts.
> - `self_heal.py`: on `ElementNotFound`/locator failure, re-scan DOM via UI Mapper,
>   update the failing locator, patch the script, retry. Cap `heal_attempts`
>   (e.g. 3) then escalate to HITL. Stream every heal attempt.
>
> Tests: a script with a deliberately stale locator must self-heal and pass.

**Done when:** breaking a locator triggers a visible self-heal that recovers, and
exceeding the cap escalates to a human.

---

# STAGE G — Phase 5 (Synthesis + Memory/RAG)

> **Paste to Claude Code:**
>
> Implement `synthesis.py` and `memory/vector_store.py` (Chroma).
>
> - `synthesis.py`: analyze execution logs, classify failure-vs-real-bug, produce a
>   structured stakeholder report (`report` in state). Render it in OutputPanel.
> - `vector_store.py`: embed and persist successful scripts, locators, and scenario
>   embeddings. Wire RAG retrieval into Context/Analyst/UI-Mapper so prior runs
>   inform new ones.
>
> Tests: a second run on a similar story retrieves and reuses prior locators.

**Done when:** runs produce a clean report and the second similar run is faster /
reuses memory.

---

# STAGE H — Governance & Eval

> **Paste to Claude Code:**
>
> 1. `observability/audit.py`: structured audit log of every inter-agent message
>    and tool invocation (run_id, agent, tool, args hash, timestamp).
> 2. Enforce least-privilege per node (each node only gets the tools it needs).
> 3. `evals/evalset.jsonl`: labeled stories (vague/good, expected INVEST verdict,
>    expected HITL trigger). `evals/judge.py`: LLM-as-judge scoring trajectory
>    accuracy + latency against the evalset; print a scoreboard.
> 4. Confirm the model router sends INVEST + Critic to Sonnet and DOM/extraction to
>    Haiku, and log token/cost per run.

**Done when:** `python evals/judge.py` prints accuracy + latency, and the audit log
captures a full run.

---

## Working Tips for Claude Code

- Start each stage with: *"Read the existing repo first, then implement Stage X
  only. Do not modify earlier stages' contracts."*
- Keep `QAState` and the trajectory event schema **frozen** once Stage A lands —
  changing them ripples through everything.
- After each stage, ask it to run the tests and the `curl`/UI smoke check before
  moving on.
- Commit per stage (`git`), so a broken stage is easy to roll back.
- Provide a `.env.example` with `ANTHROPIC_API_KEY`, `JIRA_MCP_URL`, `TARGET_URL`.

## Known Risk Notes

- **`browser-use` + Docker**: headless Chromium in the sandbox needs the right base
  image (`mcr.microsoft.com/playwright`); flag this to Claude Code in Stage F.
- **SSE through proxies**: disable response buffering (`X-Accel-Buffering: no`) or
  the stream looks frozen.
- **LangGraph interrupts**: HITL relies on the checkpointer; use a persistent
  checkpointer (SQLite/Postgres) before any real deployment, not `MemorySaver`.
- **MCP/Jira**: treat all fetched ticket text as untrusted — it goes through the
  guardrail before any model sees it as instructions.
