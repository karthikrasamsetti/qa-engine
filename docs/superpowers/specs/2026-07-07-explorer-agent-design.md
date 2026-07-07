# Explorer Agent Design

**Date:** 2026-07-07  
**Status:** Approved  
**Approach:** Standalone service (Approach 1)

---

## 1. Purpose

Add an Explorer agent that autonomously crawls a target web application and produces an `AppMap` — a structured record of all discovered pages, interactive elements, detected forms, and inferred user flows. The map serves as the foundation for exploration-driven test generation: discovered flows can be forwarded to the existing Test Case Designer pipeline via a single API call.

---

## 2. Frozen Contracts (do not modify)

| Contract | File | Rule |
|----------|------|------|
| `QAState` | `app/graph/state.py` | Extend additively only |
| `TrajectoryEvent` / `EventEmitter` | `app/streaming/events.py` | Extend additively only |
| `LLMClient` / `llm_client` | `app/llm/client.py` | Extend additively only |

The Explorer is a standalone module; it does not modify the LangGraph graph.

---

## 3. Data Shapes

### 3.1 AppMap (stored to disk, returned by API)

```json
{
  "schema_version": 1,
  "explore_id": "<uuid hex>",
  "target_url": "https://example.com",
  "target_origin": "https://example.com",
  "target_origin_slug": "example_com",
  "explored_at": "2026-07-07T10:00:00Z",
  "status": "exploring" | "complete" | "failed",
  "depth_cap": 2,
  "page_cap": 10,
  "pages": [
    {
      "url": "https://example.com/login",
      "title": "Sign In",
      "elements": [ ...from _EXTRACT_JS... ],
      "forms": [
        {
          "action": "/login",
          "method": "post",
          "fields": ["email", "password"],
          "destructive": false
        }
      ]
    }
  ],
  "flows": [
    {
      "name": "login flow",
      "pages_involved": ["https://example.com/login", "https://example.com/dashboard"],
      "description": "User enters credentials on /login, is redirected to /dashboard."
    }
  ],
  "error": null
}
```

Fields `error` and `flows` may be absent or null on a partial/failed map.

### 3.2 ExploreCredentials (in-memory only — never serialised)

```python
@dataclass
class ExploreCredentials:
    username: str
    password: str
    username_selector: str = ""   # CSS hint; auto-detected if empty
    password_selector: str = ""
```

**Credential contract:** Credentials are instantiated from the HTTP request body, held only as a Python dataclass attribute on the `ExploreAgent` instance during the Playwright session, and discarded when `run()` returns. They are **never** placed in `AppMap`, `TrajectoryEvent.data`, any dict that flows through the audit hook, or any log statement.

### 3.3 Storage Layout

```
backend/data/app_maps/
  {target_origin_slug}/          # e.g. "example_com", "localhost_8080"
    {explore_id}.json
```

`target_origin_slug` = origin with protocol stripped, non-alphanumeric chars replaced by `_`. Keyed by target identity to make future per-app map reuse structurally possible (reuse/caching logic deferred).

---

## 4. ExploreAgent Class (`app/tools/explorer.py`)

```
ExploreAgent(
    target_url: str,
    explore_id: str,
    credentials: ExploreCredentials | None = None,
    depth_cap: int = 2,
    page_cap: int = 10,
    app_maps_dir: Path | None = None,   # injectable for tests
)
```

### 4.1 `async def run() -> AppMap`

1. Builds initial `AppMap` with `status="exploring"`, persists to disk immediately.
2. Emits `action`: `"Starting exploration of {target_url}"`.
3. Opens `BrowserSession` (existing `tools/browser.py` class, reused for its Playwright lifecycle).
4. Calls `_login()` if credentials provided; adds post-login URL to crawl frontier.
5. Calls `_crawl()` — BFS over discovered same-origin URLs.
6. Calls `_infer_flows()` — LLM (reasoning tier) infers flows from page data.
7. Updates `AppMap` to `status="complete"`, persists, emits `complete`.
8. Returns `AppMap`.

**Top-level error handling:** any unhandled exception is caught; the partial map (pages already written incrementally) is flushed with `status="failed"` and `error=str(exc)`. The exception is re-raised so the endpoint emits an `error` event.

### 4.2 `_crawl()` — BFS with caps

- Frontier queue starts with `target_url` (and post-login URL if different).
- Per URL: navigate, run `_EXTRACT_JS` (imported directly from `tools/browser.py`), extract `<a href>` links.
- Same-origin filter: `urlparse(href).netloc == urlparse(target_url).netloc`.
- Depth tracked per URL in a `depth_map` dict; links at `depth > depth_cap` not enqueued.
- Stops when `pages_visited >= page_cap` or frontier empty.
- Emits `tool_call`: `"Visiting page {n}/{page_cap}: {url}"` per page.
- After each page, incrementally persists the partial map to disk so partial results survive a crash.

**Known limitation (documented in module docstring):** `_crawl()` follows `<a href>` links only. Single-page applications that navigate exclusively via JavaScript click handlers or programmatic `history.pushState` calls will not be discovered. This is acceptable for v1. Click-exploration is deferred.

### 4.3 `_detect_forms(page) -> list[FormSnapshot]`

Scans page elements for form-like structures. Tags `destructive=True` on any form whose `action`, button text, or field names contain: `delete`, `remove`, `cancel`, `pay`, `checkout`, `purchase`, `unsubscribe`. Destructive forms are **recorded** in the map but **never submitted**.

### 4.4 `_login(page) -> str | None`

Detects login form (username/password fields). Fills using `ExploreCredentials.username` / `.password` — values read inline, never placed in any dict or log. Submits and waits for navigation. Emits `tool_call`: `"Performing login"` (no credential data in message or `data={}` field). Returns post-login URL on success, `None` on failure. Emits `decision`: `"Login complete — landed on {url}"` or `error`: `"Login failed: {reason}"`.

### 4.5 `_infer_flows(pages) -> list[FlowSnapshot]`

Builds a prompt containing: URL, title, form names, and **form field names** (e.g. `email`, `password` — these are structural identifiers, not secrets) for each page. No element detail, no credential values. Calls `llm_client.complete(model_tier="reasoning")`. Parses returned JSON array. Falls back to `[]` on parse failure.

---

## 5. API Endpoints (additive changes to `app/main.py`)

### `POST /explore`
```
Body:  ExploreRequest { target_url, credentials?, depth_cap=2, page_cap=10 }
       credentials: { username, password, username_selector="", password_selector="" }
Response: ExploreResponse { explore_id }
```
Creates `explore_id`, starts `_execute_explore(...)` as background task. Credentials passed as `ExploreCredentials` dataclass; never stored.

### `GET /explore/{explore_id}/stream`
SSE endpoint. Same generator pattern as `/runs/{run_id}/stream`. Drains the shared `emitter` queue keyed by `explore_id`.

### `GET /explore/{explore_id}`
Returns the stored `AppMap` JSON from disk. 404 if not found.

### `POST /explore/{explore_id}/runs`
```
Body:  ExploreRunRequest { flow_names: list[str] }
Response: ExploreRunResponse { runs: [{ flow_name, run_id }] }
```
Reads stored `AppMap`. For each `flow_name` in `flow_names` (case-insensitive match), frames it as `"Test the {flow.name}: {flow.description}"` and calls `_execute_run(run_id, raw_input=..., target_url=map.target_url)`. Returns all spawned `run_id`s. Each run streams via existing `/runs/{run_id}/stream`.

> `ExploreRunRequest` does **not** carry a `target_url` field — the forwarding endpoint always uses `map.target_url` from the stored map. This removes ambiguity about which URL to test.

---

## 6. Safety Rails

| Rail | Implementation |
|------|----------------|
| Same-origin only | `urlparse` compare `scheme+netloc` on every discovered href |
| Depth cap | BFS `depth_map` dict; URLs beyond `depth_cap` not enqueued |
| Page cap | Hard counter; crawl exits when `pages_visited >= page_cap` |
| Destructive form guard | Forms tagged `destructive=True`; never submitted |
| Credential isolation | `ExploreCredentials` used only during login; never in `TrajectoryEvent.data`, `AppMap`, any log statement |
| Audit arg-hashing | All `TrajectoryEvent.data` passes through `_args_hash()` via the existing audit hook; credentials never enter `data`, so the hash never sees them |
| NODE_TOOL_REGISTRY | New entry: `"explorer agent": frozenset({"browser:navigate", "llm:reasoning"})` in `audit.py` |

---

## 7. Tests (`tests/test_stage_j.py`)

All tests use `asyncio_mode=auto` (existing pytest config). LLM calls are monkeypatched. A minimal HTTP fixture (two-page or N-page) serves static HTML via Python's `http.server` in a background thread.

### T1 — Explorer maps multi-page fixture, discovers login flow
- Fixture: `GET /` links to `GET /login`; `/login` has a username+password form.
- LLM mock returns one flow: `{"name": "login flow", ...}`.
- Asserts: both URLs in `app_map.pages`, `app_map.flows[0].name == "login flow"`, `app_map.status == "complete"`.

### T2 — Caps are enforced
- Fixture: 15 pages in a chain (`/page/0` → `/page/1` → … → `/page/14`).
- `page_cap=5`, `depth_cap=1`.
- Asserts: `len(app_map.pages) <= 5`; no page at depth > 1 in results.

### T3 — Credentials never appear in events or persisted map
- Fixture: login form. Credentials: `{username: "alice", password: "s3cr3t"}`.
- Collects all `TrajectoryEvent`s via the emitter.
- Asserts: `"alice"` and `"s3cr3t"` absent from all `event.message` and `json.dumps(event.data)`.
- Asserts: `"alice"` and `"s3cr3t"` absent from the persisted JSON file on disk.

### T4 — Partial map persisted on failure
- Fixture: 5-page chain. Mock navigation raises an exception on page 3.
- Asserts: persisted JSON has `status="failed"`, `error` field is non-empty string, and `pages` contains the 2 pages completed before the crash.

---

## 8. Files Changed / Created

| File | Change |
|------|--------|
| `app/tools/explorer.py` | **New** — `ExploreAgent`, `ExploreCredentials`, `AppMap`, storage helpers |
| `app/llm/prompts/explorer.py` | **New** — `EXPLORER_FLOW_SYSTEM`, `build_explorer_flow_user()` |
| `app/main.py` | **Additive** — 4 new endpoints, 2 new request/response models, `_execute_explore()` |
| `app/observability/audit.py` | **Additive** — `"explorer agent"` entry in `NODE_TOOL_REGISTRY` |
| `tests/test_stage_j.py` | **New** — T1–T4 |
