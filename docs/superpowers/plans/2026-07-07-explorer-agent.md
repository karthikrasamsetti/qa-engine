# Explorer Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone ExploreAgent that BFS-crawls a target web app, persists an AppMap (with partial-map safety), and exposes four additive `/explore` endpoints that forward discovered flows to the existing QA pipeline.

**Architecture:** `ExploreAgent` is a plain async class in `app/tools/explorer.py` with no LangGraph dependency. It drives Playwright directly, emits `TrajectoryEvent`s via the existing `emitter` singleton, and persists `AppMap` JSON to `data/app_maps/{slug}/{explore_id}.json`. Four new endpoints are added to `main.py`; the LangGraph graph, `QAState`, and `TrajectoryEvent` are untouched. Credential values are held only in an `ExploreCredentials` dataclass instance and never propagate to any event, map, or log.

**Tech Stack:** Python 3.13, Playwright async API, FastAPI, Pydantic v2, stdlib `http.server` + `threading` (test fixtures), pytest `asyncio_mode=auto`.

**Consistency note from spec review:** same-origin check uses `scheme+netloc` (e.g. `https://example.com` vs `http://example.com` are different origins). Implement as `f"{parsed.scheme}://{parsed.netloc}"` throughout — never netloc-only.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `app/tools/explorer.py` | **Create** | `ExploreCredentials`, `AppMap`/`PageSnapshot`/`FormSnapshot`/`FlowSnapshot` models, `save_map`/`load_map`/`_origin_slug` helpers, `ExploreAgent` class with `run()`, `_crawl()`, `_login()`, `_detect_forms()`, `_infer_flows()` |
| `app/llm/prompts/explorer.py` | **Create** | `EXPLORER_FLOW_SYSTEM`, `build_explorer_flow_user()` |
| `app/observability/audit.py` | **Modify** (additive) | Add `"explorer agent"` entry to `NODE_TOOL_REGISTRY` |
| `app/main.py` | **Modify** (additive) | `CredentialsInput`, `ExploreRequest/Response`, `ExploreRunRequest/Response`, `_execute_explore()`, four new endpoints |
| `tests/test_stage_j.py` | **Create** | Fixture server helper, T1–T4, three endpoint smoke tests |

---

## Task 1: Data shapes + storage helpers

**Files:**
- Create: `app/tools/explorer.py` (models + storage only — no Playwright yet)
- Create: `tests/test_stage_j.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_stage_j.py`:

```python
"""Stage J tests: Explorer Agent."""
from __future__ import annotations

import asyncio
import json
import threading
import http.server
from pathlib import Path
from urllib.parse import urlparse as _urlparse

import pytest


# ---------------------------------------------------------------------------
# Task 1: slug + storage helpers
# ---------------------------------------------------------------------------

def test_origin_slug_strips_protocol_and_normalises():
    from app.tools.explorer import _origin_slug
    assert _origin_slug("https://example.com") == "example_com"
    assert _origin_slug("http://app.example.com:8080") == "app_example_com_8080"
    assert _origin_slug("https://localhost:3000") == "localhost_3000"


def test_app_map_round_trips_to_json(tmp_path):
    from app.tools.explorer import AppMap, save_map, load_map
    m = AppMap(
        explore_id="abc123",
        target_url="https://example.com",
        target_origin="https://example.com",
        target_origin_slug="example_com",
        status="exploring",
        depth_cap=2,
        page_cap=10,
    )
    save_map(m, app_maps_dir=tmp_path)
    loaded = load_map("abc123", "example_com", app_maps_dir=tmp_path)
    assert loaded is not None
    assert loaded.explore_id == "abc123"
    assert loaded.status == "exploring"
    assert loaded.schema_version == 1


def test_load_map_returns_none_for_missing(tmp_path):
    from app.tools.explorer import load_map
    assert load_map("no-such-id", "example_com", app_maps_dir=tmp_path) is None
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && python -m pytest tests/test_stage_j.py::test_origin_slug_strips_protocol_and_normalises tests/test_stage_j.py::test_app_map_round_trips_to_json tests/test_stage_j.py::test_load_map_returns_none_for_missing -v
```

Expected: `ImportError` — `app.tools.explorer` does not exist yet.

- [ ] **Step 3: Create `app/tools/explorer.py` with models + storage**

```python
"""Explorer agent: autonomous web-app crawler for AppMap generation.

Drives a Playwright browser session to BFS-crawl a target web application,
recording pages, interactive elements, forms, and inferred user flows.

Known limitation — href-only crawling:
    _crawl() follows <a href> links only. Single-page applications that
    navigate exclusively via JavaScript click handlers or programmatic
    history.pushState calls will not be discovered. Pages reachable only
    through JS-driven navigation are silently absent from the resulting
    AppMap. This is a known v1 limitation; click-based exploration is
    deferred to a future stage.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_APP_MAPS_DIR = Path(__file__).parent.parent.parent / "data" / "app_maps"


# ---------------------------------------------------------------------------
# Credentials — in-memory only, NEVER serialised
# ---------------------------------------------------------------------------

@dataclass
class ExploreCredentials:
    """Login credentials used only during the Playwright session.

    Never placed in AppMap, TrajectoryEvent.data, or any log statement.
    Discarded when ExploreAgent.run() returns.
    """
    username: str
    password: str
    username_selector: str = ""
    password_selector: str = ""


# ---------------------------------------------------------------------------
# AppMap components
# ---------------------------------------------------------------------------

class FormSnapshot(BaseModel):
    action: str = ""
    method: str = "get"
    fields: list[str] = Field(default_factory=list)
    destructive: bool = False


class PageSnapshot(BaseModel):
    url: str
    title: str = ""
    elements: list[dict[str, Any]] = Field(default_factory=list)
    forms: list[FormSnapshot] = Field(default_factory=list)


class FlowSnapshot(BaseModel):
    name: str
    pages_involved: list[str] = Field(default_factory=list)
    description: str = ""


class AppMap(BaseModel):
    schema_version: int = 1
    explore_id: str
    target_url: str
    target_origin: str
    target_origin_slug: str
    explored_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "exploring"   # "exploring" | "complete" | "failed"
    depth_cap: int = 2
    page_cap: int = 10
    pages: list[PageSnapshot] = Field(default_factory=list)
    flows: list[FlowSnapshot] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _origin_slug(origin: str) -> str:
    """Convert an origin URL to a filesystem-safe slug.

    Strips protocol, replaces non-alphanumeric chars with underscores.
    """
    parsed = urlparse(origin)
    raw = parsed.netloc or parsed.path
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_")


def _map_path(explore_id: str, slug: str, base: Path) -> Path:
    return base / slug / f"{explore_id}.json"


def save_map(app_map: AppMap, app_maps_dir: Path | None = None) -> None:
    """Persist AppMap to disk. All errors are caught and logged."""
    try:
        base = app_maps_dir or _DEFAULT_APP_MAPS_DIR
        path = _map_path(app_map.explore_id, app_map.target_origin_slug, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(app_map.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("save_map failed for %s: %s", app_map.explore_id, exc)


def load_map(
    explore_id: str,
    slug: str,
    app_maps_dir: Path | None = None,
) -> AppMap | None:
    """Load AppMap from disk. Returns None when file not found."""
    try:
        base = app_maps_dir or _DEFAULT_APP_MAPS_DIR
        path = _map_path(explore_id, slug, base)
        if not path.exists():
            return None
        return AppMap.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("load_map failed for %s: %s", explore_id, exc)
        return None
```

- [ ] **Step 4: Run to confirm pass**

```
cd backend && python -m pytest tests/test_stage_j.py::test_origin_slug_strips_protocol_and_normalises tests/test_stage_j.py::test_app_map_round_trips_to_json tests/test_stage_j.py::test_load_map_returns_none_for_missing -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```
git add app/tools/explorer.py tests/test_stage_j.py && git commit -m "feat(explorer): data shapes, storage helpers — Stage J Task 1"
```

---

## Task 2: LLM prompt for flow inference

**Files:**
- Create: `app/llm/prompts/explorer.py`
- Modify: `tests/test_stage_j.py` (append)

- [ ] **Step 1: Append failing test**

```python
# ---------------------------------------------------------------------------
# Task 2: flow inference prompt
# ---------------------------------------------------------------------------

def test_build_explorer_flow_user_includes_urls_titles_and_form_fields():
    from app.llm.prompts.explorer import build_explorer_flow_user
    pages = [
        {
            "url": "https://example.com/login",
            "title": "Sign In",
            "forms": [{"action": "/auth", "method": "post", "fields": ["email", "password"]}],
        },
        {
            "url": "https://example.com/dashboard",
            "title": "Dashboard",
            "forms": [],
        },
    ]
    prompt = build_explorer_flow_user(pages)
    assert "https://example.com/login" in prompt
    assert "Sign In" in prompt
    assert "email" in prompt
    assert "password" in prompt
    assert "Dashboard" in prompt
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && python -m pytest tests/test_stage_j.py::test_build_explorer_flow_user_includes_urls_titles_and_form_fields -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `app/llm/prompts/explorer.py`**

```python
"""LLM prompts for the Explorer agent flow-inference step."""
from __future__ import annotations

EXPLORER_FLOW_SYSTEM = """\
You are a web-application analyst. You receive a list of pages discovered by an \
automated crawler: each page's URL, title, and any HTML forms with their field names.

Your task: identify user-facing *flows* — named sequences of pages that represent \
a coherent user journey (e.g. "login flow", "signup flow", "checkout flow", \
"password reset flow").

Return a JSON array of flow objects. Each object must have exactly these keys:
  "name"           — short lowercase slug (e.g. "login flow")
  "pages_involved" — list of URLs that are part of this flow, in order
  "description"    — one sentence describing what the user accomplishes

Return ONLY the JSON array with no prose, no markdown fences."""


def build_explorer_flow_user(pages: list[dict]) -> str:
    """Build the user-turn prompt for flow inference.

    Includes URL, title, and form field names (structural identifiers, not secrets).
    Excludes raw element details and any credential values.
    """
    lines: list[str] = ["Pages discovered:\n"]
    for p in pages:
        lines.append(f"URL: {p['url']}")
        lines.append(f"  Title: {p.get('title', '')}")
        for form in p.get("forms", []):
            fields = ", ".join(form.get("fields", []))
            action = form.get("action", "")
            lines.append(f"  Form → action={action!r} fields=[{fields}]")
        lines.append("")
    lines.append(
        "Return a JSON array of flow objects with keys: "
        "name, pages_involved, description."
    )
    return "\n".join(lines)
```

- [ ] **Step 4: Run to confirm pass**

```
cd backend && python -m pytest tests/test_stage_j.py::test_build_explorer_flow_user_includes_urls_titles_and_form_fields -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add app/llm/prompts/explorer.py tests/test_stage_j.py && git commit -m "feat(explorer): flow inference LLM prompt — Stage J Task 2"
```

---

## Task 3: Audit registry entry

**Files:**
- Modify: `app/observability/audit.py` (additive — one dict entry)
- Modify: `tests/test_stage_j.py` (append)

- [ ] **Step 1: Append failing test**

```python
# ---------------------------------------------------------------------------
# Task 3: audit registry
# ---------------------------------------------------------------------------

def test_explorer_agent_in_node_tool_registry():
    from app.observability.audit import NODE_TOOL_REGISTRY
    assert "explorer agent" in NODE_TOOL_REGISTRY
    allowed = NODE_TOOL_REGISTRY["explorer agent"]
    assert "browser:navigate" in allowed
    assert "llm:reasoning" in allowed
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && python -m pytest tests/test_stage_j.py::test_explorer_agent_in_node_tool_registry -v
```

Expected: `AssertionError` — key absent.

- [ ] **Step 3: Add the entry in `app/observability/audit.py`**

After the `"system"` line (currently line 71), add one entry so the dict ends:

```python
    "system":                   frozenset({"llm:reasoning", "llm:fast"}),
    "explorer agent":           frozenset({"browser:navigate", "llm:reasoning"}),
}
```

- [ ] **Step 4: Run to confirm pass**

```
cd backend && python -m pytest tests/test_stage_j.py::test_explorer_agent_in_node_tool_registry -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```
git add app/observability/audit.py tests/test_stage_j.py && git commit -m "feat(explorer): add explorer agent to NODE_TOOL_REGISTRY — Stage J Task 3"
```

---

## Task 4: `_detect_forms()`, fixture server helper, `ExploreAgent` + BFS crawl, T2

**Files:**
- Modify: `app/tools/explorer.py` (append `_detect_forms`, `ExploreAgent` class)
- Modify: `tests/test_stage_j.py` (append fixture helper + T2)

- [ ] **Step 1: Append fixture server helper + T2 failing test**

```python
# ---------------------------------------------------------------------------
# Shared fixture server (used by T1–T4)
# ---------------------------------------------------------------------------

class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    pages: dict[str, str] = {}

    def do_GET(self):
        body = self.pages.get(self.path, "<html><body>Not found</body></html>")
        body_bytes = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def log_message(self, *_):
        pass


def _start_fixture_server(pages: dict[str, str]) -> tuple[str, http.server.HTTPServer]:
    """Serve pages dict on a random local port. Returns (base_url, server)."""
    handler_cls = type("H", (_FixtureHandler,), {"pages": pages})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{port}", server


# ---------------------------------------------------------------------------
# T2 — caps are enforced
# ---------------------------------------------------------------------------

async def test_caps_are_enforced(tmp_path, monkeypatch):
    """page_cap=5, depth_cap=1: at most 5 pages visited, none past depth 1."""
    from app.tools.explorer import ExploreAgent
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _llm_stub(messages, model_tier, **kw):
        return LLMResponse(text="[]", input_tokens=1, output_tokens=1, model="mock", cost_usd=0.0)
    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_stub)

    # 15-page chain: / → /p/1 → /p/2 → … → /p/14
    def _chain_page(n: int) -> str:
        nxt = f"/p/{n+1}" if n < 14 else ""
        link = f"<a href='{nxt}'>next</a>" if nxt else ""
        return f"<html><head><title>Page {n}</title></head><body>{link}</body></html>"

    pages = {"/": _chain_page(0)}
    for i in range(1, 15):
        pages[f"/p/{i}"] = _chain_page(i)

    base_url, server = _start_fixture_server(pages)

    agent = ExploreAgent(
        target_url=base_url,
        explore_id="t2-caps",
        depth_cap=1,
        page_cap=5,
        app_maps_dir=tmp_path,
    )
    app_map = await agent.run()
    server.shutdown()

    assert len(app_map.pages) <= 5, f"Expected ≤5 pages, got {len(app_map.pages)}"

    visited_paths = {_urlparse(p.url).path for p in app_map.pages}
    # depth_cap=1: /p/2 and beyond are at depth 2 from / — must not appear
    for i in range(2, 15):
        assert f"/p/{i}" not in visited_paths, f"depth-2 page /p/{i} must not be visited"
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && python -m pytest tests/test_stage_j.py::test_caps_are_enforced -v
```

Expected: `ImportError` — `ExploreAgent` not defined.

- [ ] **Step 3: Append `_detect_forms` + `ExploreAgent` to `app/tools/explorer.py`**

Add the following imports at the top of the file (after existing imports):

```python
# (already imported above: re, Any, urlparse)
```

Then append to the end of `app/tools/explorer.py`:

```python
# ---------------------------------------------------------------------------
# DOM probe — reused from browser.py to avoid duplication
# ---------------------------------------------------------------------------

from app.tools.browser import _EXTRACT_JS  # noqa: E402

# ---------------------------------------------------------------------------
# Destructive form keywords
# ---------------------------------------------------------------------------

_DESTRUCTIVE_KEYWORDS = frozenset({
    "delete", "remove", "cancel", "pay", "checkout",
    "purchase", "unsubscribe",
})


def _detect_forms(elements: list[dict[str, Any]]) -> list[FormSnapshot]:
    """Infer form snapshots from the _EXTRACT_JS element list.

    Groups input fields + submit triggers into one logical form per page
    (v1 simplification). Tags destructive=True when any element text
    matches a destructive keyword.
    """
    inputs = [e for e in elements if e.get("tag") in ("input", "textarea", "select")]
    submits = [e for e in elements if e.get("type") in ("submit",) or e.get("tag") == "button"]

    if not inputs and not submits:
        return []

    fields = [
        e.get("name") or e.get("id") or e.get("placeholder") or e.get("type") or ""
        for e in inputs
    ]
    fields = [f for f in fields if f]

    all_text = " ".join(
        " ".join([
            str(e.get("text") or ""),
            str(e.get("name") or ""),
            str(e.get("aria_label") or ""),
            str(e.get("placeholder") or ""),
        ]).lower()
        for e in elements
    )
    destructive = any(kw in all_text for kw in _DESTRUCTIVE_KEYWORDS)
    has_password = any(e.get("type") == "password" for e in inputs)

    return [FormSnapshot(
        action="",
        method="post" if has_password else "get",
        fields=fields,
        destructive=destructive,
    )]


# ---------------------------------------------------------------------------
# ExploreAgent
# ---------------------------------------------------------------------------

class ExploreAgent:
    """Autonomous web-app crawler.

    BFS-crawls a target web application up to depth_cap and page_cap limits.
    Records URL, title, interactive elements, and forms for each page.
    Infers user-facing flows via the reasoning-tier LLM after crawling.

    Credential handling: ExploreCredentials values are used only during
    _login() and are never placed in TrajectoryEvent.data, AppMap fields,
    or any log statement.

    See module docstring for the known href-only crawling limitation.
    """

    AGENT = "Explorer Agent"
    PHASE = 0

    def __init__(
        self,
        target_url: str,
        explore_id: str,
        credentials: ExploreCredentials | None = None,
        depth_cap: int = 2,
        page_cap: int = 10,
        app_maps_dir: Path | None = None,
    ) -> None:
        self._target_url = target_url
        self._explore_id = explore_id
        self._credentials = credentials
        self._depth_cap = depth_cap
        self._page_cap = page_cap
        self._app_maps_dir = app_maps_dir or _DEFAULT_APP_MAPS_DIR

        parsed = urlparse(target_url)
        self._target_origin = f"{parsed.scheme}://{parsed.netloc}"
        self._slug = _origin_slug(target_url)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> AppMap:
        from app.streaming.events import emitter

        app_map = AppMap(
            explore_id=self._explore_id,
            target_url=self._target_url,
            target_origin=self._target_origin,
            target_origin_slug=self._slug,
            status="exploring",
            depth_cap=self._depth_cap,
            page_cap=self._page_cap,
        )
        save_map(app_map, self._app_maps_dir)

        await emitter.emit(
            self._explore_id, self.AGENT, self.PHASE, "action",
            f"Starting exploration of {self._target_url} "
            f"(depth_cap={self._depth_cap}, page_cap={self._page_cap})",
        )

        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()

                seed_urls = [self._target_url]
                if self._credentials is not None:
                    post_login_url = await self._login(page)
                    if post_login_url and post_login_url != self._target_url:
                        seed_urls.insert(0, post_login_url)

                await self._crawl(page, app_map, seed_urls)
                await browser.close()

            app_map.flows = await self._infer_flows(app_map.pages)
            app_map.status = "complete"
            save_map(app_map, self._app_maps_dir)

            await emitter.emit(
                self._explore_id, self.AGENT, self.PHASE, "complete",
                f"Exploration complete: {len(app_map.pages)} page(s), "
                f"{len(app_map.flows)} flow(s) inferred.",
                data={"pages": len(app_map.pages), "flows": len(app_map.flows)},
            )

        except Exception as exc:
            logger.error("ExploreAgent failed for %s: %s", self._explore_id, exc)
            app_map.status = "failed"
            app_map.error = str(exc)
            save_map(app_map, self._app_maps_dir)
            await emitter.emit(
                self._explore_id, self.AGENT, self.PHASE, "error",
                f"Exploration failed: {exc}",
            )
            raise

        return app_map

    # ------------------------------------------------------------------
    # Same-origin check — scheme + netloc (both must match)
    # ------------------------------------------------------------------

    def _is_same_origin(self, url: str) -> bool:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}" == self._target_origin

    # ------------------------------------------------------------------
    # BFS crawl
    # ------------------------------------------------------------------

    async def _crawl(
        self,
        page,
        app_map: AppMap,
        seed_urls: list[str],
    ) -> None:
        from playwright.async_api import TimeoutError as PWTimeout
        from app.streaming.events import emitter

        visited: set[str] = set()
        frontier: list[tuple[str, int]] = [(u, 0) for u in seed_urls]
        pages_visited = 0

        while frontier and pages_visited < self._page_cap:
            url, depth = frontier.pop(0)
            if url in visited:
                continue
            visited.add(url)
            pages_visited += 1

            await emitter.emit(
                self._explore_id, self.AGENT, self.PHASE, "tool_call",
                f"Visiting page {pages_visited}/{self._page_cap}: {url}",
                data={"url": url, "depth": depth},
            )

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            except (PWTimeout, Exception) as exc:
                logger.warning("Navigation failed for %s: %s", url, exc)
                continue

            title = await page.title()
            try:
                elements: list[dict] = await page.evaluate(_EXTRACT_JS)
            except Exception as exc:
                logger.warning("DOM extraction failed for %s: %s", url, exc)
                elements = []

            forms = _detect_forms(elements)
            snapshot = PageSnapshot(url=url, title=title, elements=elements, forms=forms)
            app_map.pages.append(snapshot)
            # Incremental persistence — partial results survive a crash
            save_map(app_map, self._app_maps_dir)

            if depth < self._depth_cap:
                try:
                    hrefs: list[str] = await page.evaluate(
                        "() => Array.from(document.querySelectorAll('a[href]'))"
                        ".map(a => a.href)"
                    )
                except Exception:
                    hrefs = []
                for href in hrefs:
                    href = href.split("#")[0].rstrip("/") or href
                    if href and href not in visited and self._is_same_origin(href):
                        frontier.append((href, depth + 1))

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def _login(self, page) -> str | None:
        """Fill and submit the login form.

        Credential values are accessed inline and never placed in any dict,
        event data field, or log statement.
        """
        from playwright.async_api import TimeoutError as PWTimeout
        from app.streaming.events import emitter

        # Intentionally empty data= dict — no credentials ever here
        await emitter.emit(
            self._explore_id, self.AGENT, self.PHASE, "tool_call",
            "Performing login",
            data={},
        )

        try:
            await page.goto(self._target_url, wait_until="domcontentloaded", timeout=15_000)

            u_sel = self._credentials.username_selector
            if not u_sel:
                for candidate in (
                    "input[type='email']", "input[name='email']",
                    "input[name='username']", "input[type='text']",
                ):
                    if await page.locator(candidate).count() > 0:
                        u_sel = candidate
                        break

            p_sel = self._credentials.password_selector or "input[type='password']"

            if not u_sel or await page.locator(u_sel).count() == 0:
                logger.warning("Login: username field not found")
                await emitter.emit(self._explore_id, self.AGENT, self.PHASE,
                                   "error", "Login failed: username field not found")
                return None

            if await page.locator(p_sel).count() == 0:
                logger.warning("Login: password field not found")
                await emitter.emit(self._explore_id, self.AGENT, self.PHASE,
                                   "error", "Login failed: password field not found")
                return None

            await page.fill(u_sel, self._credentials.username)
            await page.fill(p_sel, self._credentials.password)

            submit_sel = (
                "button[type='submit'], input[type='submit'], "
                "button:has-text('Login'), button:has-text('Sign in'), "
                "button:has-text('Sign In')"
            )
            if await page.locator(submit_sel).first.count() > 0:
                await page.locator(submit_sel).first.click()
            else:
                await page.locator(p_sel).press("Enter")

            await page.wait_for_load_state("domcontentloaded", timeout=10_000)
            post_login_url = page.url

            await emitter.emit(
                self._explore_id, self.AGENT, self.PHASE, "decision",
                f"Login complete — landed on {post_login_url}",
                data={"post_login_url": post_login_url},
            )
            return post_login_url

        except (PWTimeout, Exception) as exc:
            logger.warning("Login failed: %s", exc)
            await emitter.emit(self._explore_id, self.AGENT, self.PHASE,
                               "error", f"Login failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Flow inference
    # ------------------------------------------------------------------

    async def _infer_flows(self, pages: list[PageSnapshot]) -> list[FlowSnapshot]:
        """Ask the reasoning LLM to infer user-facing flows from page data."""
        import json as _json
        import re as _re
        from app.llm.client import llm_client
        from app.llm.prompts.explorer import EXPLORER_FLOW_SYSTEM, build_explorer_flow_user

        page_dicts = [
            {
                "url": p.url,
                "title": p.title,
                "forms": [
                    {"action": f.action, "method": f.method, "fields": f.fields}
                    for f in p.forms
                ],
            }
            for p in pages
        ]

        try:
            resp = await llm_client.complete(
                messages=[
                    {"role": "system", "content": EXPLORER_FLOW_SYSTEM},
                    {"role": "user",   "content": build_explorer_flow_user(page_dicts)},
                ],
                model_tier="reasoning",
                run_id=self._explore_id,
            )
            match = _re.search(r"\[.*\]", resp.text, _re.DOTALL)
            if not match:
                return []
            raw = _json.loads(match.group())
            if not isinstance(raw, list):
                return []
            flows = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                flows.append(FlowSnapshot(
                    name=str(item.get("name", "unnamed flow")),
                    pages_involved=list(item.get("pages_involved", [])),
                    description=str(item.get("description", "")),
                ))
            return flows
        except Exception as exc:
            logger.error("_infer_flows failed: %s", exc)
            return []
```

- [ ] **Step 4: Run T2 to confirm pass**

```
cd backend && python -m pytest tests/test_stage_j.py::test_caps_are_enforced -v
```

Expected: PASS. If navigation errors occur ensure Playwright is installed: `playwright install chromium`.

- [ ] **Step 5: Run all tasks so far to confirm no regressions**

```
cd backend && python -m pytest tests/test_stage_j.py -v -k "not t1 and not t2_caps and not creds and not partial and not explore_id and not 404 and not flows"
```

Or just run the three unit tests + T2:

```
cd backend && python -m pytest tests/test_stage_j.py::test_origin_slug_strips_protocol_and_normalises tests/test_stage_j.py::test_app_map_round_trips_to_json tests/test_stage_j.py::test_load_map_returns_none_for_missing tests/test_stage_j.py::test_build_explorer_flow_user_includes_urls_titles_and_form_fields tests/test_stage_j.py::test_explorer_agent_in_node_tool_registry tests/test_stage_j.py::test_caps_are_enforced -v
```

Expected: all 6 PASS.

- [ ] **Step 6: Commit**

```
git add app/tools/explorer.py tests/test_stage_j.py && git commit -m "feat(explorer): _detect_forms, ExploreAgent, BFS crawl, T2 green — Stage J Task 4"
```

---

## Task 5: T1 — multi-page fixture, login flow discovered

**Files:**
- Modify: `tests/test_stage_j.py` (append T1)

- [ ] **Step 1: Append T1 failing test**

```python
# ---------------------------------------------------------------------------
# T1 — Explorer maps multi-page fixture, discovers login flow
# ---------------------------------------------------------------------------

async def test_explorer_discovers_login_flow(tmp_path, monkeypatch):
    """Visits / and /login; LLM mock returns 'login flow'; map is complete."""
    from app.tools.explorer import ExploreAgent
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    _FLOW_JSON = json.dumps([{
        "name": "login flow",
        "pages_involved": [],
        "description": "User logs in via the login page.",
    }])

    async def _llm_stub(messages, model_tier, **kw):
        return LLMResponse(text=_FLOW_JSON, input_tokens=10, output_tokens=50,
                           model="mock", cost_usd=0.0)
    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_stub)

    pages = {
        "/": (
            "<html><head><title>Home</title></head>"
            "<body><a href='/login'>Login</a></body></html>"
        ),
        "/login": (
            "<html><head><title>Sign In</title></head><body>"
            "<form method='post' action='/auth'>"
            "<input type='email' name='email' placeholder='Email'/>"
            "<input type='password' name='password'/>"
            "<button type='submit'>Login</button>"
            "</form></body></html>"
        ),
    }
    base_url, server = _start_fixture_server(pages)

    agent = ExploreAgent(
        target_url=base_url,
        explore_id="t1-login",
        depth_cap=2,
        page_cap=10,
        app_maps_dir=tmp_path,
    )
    app_map = await agent.run()
    server.shutdown()

    visited_urls = {p.url.rstrip("/") for p in app_map.pages}
    assert base_url.rstrip("/") in visited_urls, \
        f"Home page missing; visited: {visited_urls}"
    assert base_url.rstrip("/") + "/login" in visited_urls, \
        f"/login missing; visited: {visited_urls}"

    assert app_map.status == "complete", f"Expected complete, got {app_map.status}"
    assert len(app_map.flows) == 1, f"Expected 1 flow, got {app_map.flows}"
    assert app_map.flows[0].name == "login flow"
```

- [ ] **Step 2: Run to confirm failure (or pass — Playwright integration may already work)**

```
cd backend && python -m pytest tests/test_stage_j.py::test_explorer_discovers_login_flow -v
```

Expected: PASS (real Playwright navigates the fixture). If it fails with a Playwright error run `playwright install chromium` first.

- [ ] **Step 3: Run T1 + T2 together**

```
cd backend && python -m pytest tests/test_stage_j.py::test_explorer_discovers_login_flow tests/test_stage_j.py::test_caps_are_enforced -v
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```
git add tests/test_stage_j.py && git commit -m "test(explorer): T1 login flow discovery green — Stage J Task 5"
```

---

## Task 6: T3 — credential isolation + T4 — partial map on failure

**Files:**
- Modify: `tests/test_stage_j.py` (append T3 and T4)

- [ ] **Step 1: Append T3 and T4 failing tests**

```python
# ---------------------------------------------------------------------------
# T3 — credentials never appear in events or persisted map
# ---------------------------------------------------------------------------

async def test_credentials_never_in_events_or_map(tmp_path, monkeypatch):
    """'alice' and 's3cr3t' must be absent from all events and the JSON file."""
    from app.tools.explorer import ExploreAgent, ExploreCredentials
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse
    from app.streaming.events import emitter

    async def _llm_stub(messages, model_tier, **kw):
        return LLMResponse(text="[]", input_tokens=1, output_tokens=1,
                           model="mock", cost_usd=0.0)
    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_stub)

    pages = {
        "/": (
            "<html><head><title>Login</title></head><body>"
            "<form method='post' action='/auth'>"
            "<input type='email' name='email'/>"
            "<input type='password' name='password'/>"
            "<button type='submit'>Login</button>"
            "</form></body></html>"
        ),
    }
    base_url, server = _start_fixture_server(pages)
    explore_id = "t3-creds"

    agent = ExploreAgent(
        target_url=base_url,
        explore_id=explore_id,
        credentials=ExploreCredentials(username="alice", password="s3cr3t"),
        depth_cap=1,
        page_cap=3,
        app_maps_dir=tmp_path,
    )
    await agent.run()
    server.shutdown()

    # Drain all events emitted for this run from the asyncio queue
    collected = []
    q = emitter._queues.get(explore_id)
    if q is not None:
        while not q.empty():
            try:
                item = q.get_nowait()
                if item is not None:
                    collected.append(item)
            except asyncio.QueueEmpty:
                break

    SECRET_TERMS = ("alice", "s3cr3t")
    for ev in collected:
        for term in SECRET_TERMS:
            assert term not in ev.message, \
                f"{term!r} found in event message: {ev.message!r}"
            data_str = json.dumps(ev.data)
            assert term not in data_str, \
                f"{term!r} found in event data: {data_str!r}"

    # Check the persisted JSON file
    files = list(tmp_path.rglob(f"{explore_id}.json"))
    assert files, "AppMap file not found on disk"
    content = files[0].read_text(encoding="utf-8")
    for term in SECRET_TERMS:
        assert term not in content, \
            f"{term!r} found in persisted AppMap JSON"


# ---------------------------------------------------------------------------
# T4 — partial map persisted on failure
# ---------------------------------------------------------------------------

async def test_partial_map_persisted_on_failure(tmp_path, monkeypatch):
    """When _crawl raises after 2 pages, map has status=failed, error, and 2 pages."""
    from app.tools.explorer import ExploreAgent, PageSnapshot, save_map
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _llm_stub(messages, model_tier, **kw):
        return LLMResponse(text="[]", input_tokens=1, output_tokens=1,
                           model="mock", cost_usd=0.0)
    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_stub)

    # Patch _crawl to save 2 fake pages then raise — tests run() error handling
    async def _patched_crawl(self, page, app_map, seed_urls):
        for i in range(2):
            snap = PageSnapshot(url=f"{self._target_url}/p/{i}", title=f"Page {i}")
            app_map.pages.append(snap)
            save_map(app_map, self._app_maps_dir)
        raise RuntimeError("Simulated navigation failure on page 3")

    monkeypatch.setattr(ExploreAgent, "_crawl", _patched_crawl)

    # Minimal fixture (browser still opens but _crawl is patched before navigation)
    pages = {"/": "<html><head><title>Home</title></head><body></body></html>"}
    base_url, server = _start_fixture_server(pages)
    explore_id = "t4-partial"

    agent = ExploreAgent(
        target_url=base_url,
        explore_id=explore_id,
        depth_cap=2,
        page_cap=10,
        app_maps_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="Simulated navigation failure"):
        await agent.run()
    server.shutdown()

    files = list(tmp_path.rglob(f"{explore_id}.json"))
    assert files, "Partial AppMap was not persisted to disk"
    raw = json.loads(files[0].read_text(encoding="utf-8"))

    assert raw["status"] == "failed", f"Expected status=failed, got {raw['status']}"
    assert raw["error"], "Expected non-empty error string"
    assert len(raw["pages"]) == 2, \
        f"Expected 2 pages before crash, got {len(raw['pages'])}"
```

- [ ] **Step 2: Run to confirm both fail (before implementation — should fail on assertion or import)**

```
cd backend && python -m pytest tests/test_stage_j.py::test_credentials_never_in_events_or_map tests/test_stage_j.py::test_partial_map_persisted_on_failure -v
```

Expected: T3 should PASS immediately (the implementation already ensures credentials stay out of events/data). T4 should PASS immediately (the `run()` error handler already saves the partial map). If either fails, debug before proceeding.

- [ ] **Step 3: Run complete T1–T4 battery**

```
cd backend && python -m pytest tests/test_stage_j.py::test_explorer_discovers_login_flow tests/test_stage_j.py::test_caps_are_enforced tests/test_stage_j.py::test_credentials_never_in_events_or_map tests/test_stage_j.py::test_partial_map_persisted_on_failure -v
```

Expected: all 4 PASS.

- [ ] **Step 4: Commit**

```
git add tests/test_stage_j.py && git commit -m "test(explorer): T3 credential isolation + T4 partial map green — Stage J Task 6"
```

---

## Task 7: API endpoints

**Files:**
- Modify: `app/main.py` (additive — models + `_execute_explore` + 4 endpoints)
- Modify: `tests/test_stage_j.py` (append 3 endpoint smoke tests)

- [ ] **Step 1: Append endpoint smoke tests (sync, use TestClient)**

```python
# ---------------------------------------------------------------------------
# Task 7: endpoint smoke tests (sync — TestClient avoids async/loop conflicts)
# ---------------------------------------------------------------------------

def test_post_explore_returns_explore_id(monkeypatch):
    """POST /explore returns {explore_id} immediately without running Playwright."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.tools import explorer as explorer_mod

    async def _noop_run(self):
        return explorer_mod.AppMap(
            explore_id=self._explore_id,
            target_url=self._target_url,
            target_origin=self._target_origin,
            target_origin_slug=self._slug,
            status="complete",
        )
    monkeypatch.setattr(explorer_mod.ExploreAgent, "run", _noop_run)

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/explore", json={"target_url": "https://example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert "explore_id" in body and len(body["explore_id"]) > 0


def test_get_explore_returns_404_for_unknown():
    """GET /explore/{id} returns 404 when no matching map exists."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/explore/definitely-does-not-exist-xyz")
    assert resp.status_code == 404


def test_explore_runs_forwards_flows(tmp_path, monkeypatch):
    """POST /explore/{id}/runs spawns a run_id per requested flow_name."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.tools.explorer import AppMap, FlowSnapshot, save_map
    import app.tools.explorer as explorer_mod
    import app.main as main_mod

    slug = "example_com"
    explore_id = "fwd-test-001"
    app_map = AppMap(
        explore_id=explore_id,
        target_url="https://example.com",
        target_origin="https://example.com",
        target_origin_slug=slug,
        status="complete",
        flows=[
            FlowSnapshot(
                name="login flow",
                pages_involved=["https://example.com/login"],
                description="User logs in via the login page.",
            )
        ],
    )
    monkeypatch.setattr(explorer_mod, "_DEFAULT_APP_MAPS_DIR", tmp_path)
    save_map(app_map, tmp_path)

    spawned: list[dict] = []

    async def _fake_execute_run(run_id, raw_input, target_url, review_scenarios=False):
        spawned.append({"run_id": run_id, "raw_input": raw_input})

    monkeypatch.setattr(main_mod, "_execute_run", _fake_execute_run)

    with TestClient(app) as client:
        resp = client.post(
            f"/explore/{explore_id}/runs",
            json={"flow_names": ["login flow"]},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["runs"]) == 1
    assert body["runs"][0]["flow_name"] == "login flow"
    assert "run_id" in body["runs"][0]
```

- [ ] **Step 2: Run to confirm failure**

```
cd backend && python -m pytest tests/test_stage_j.py::test_post_explore_returns_explore_id tests/test_stage_j.py::test_get_explore_returns_404_for_unknown tests/test_stage_j.py::test_explore_runs_forwards_flows -v
```

Expected: FAIL — `/explore` routes not defined.

- [ ] **Step 3: Add models + `_execute_explore` + 4 endpoints to `app/main.py`**

After the existing `ResumeRequest` model (after line 59), insert:

```python
class CredentialsInput(BaseModel):
    username: str
    password: str
    username_selector: str = ""
    password_selector: str = ""


class ExploreRequest(BaseModel):
    target_url: str
    credentials: CredentialsInput | None = None
    depth_cap: int = 2
    page_cap: int = 10


class ExploreResponse(BaseModel):
    explore_id: str


class ExploreRunRequest(BaseModel):
    # target_url is intentionally absent: the forwarding endpoint always
    # uses map.target_url from the stored AppMap to avoid ambiguity.
    flow_names: list[str]


class ExploreRunItem(BaseModel):
    flow_name: str
    run_id: str


class ExploreRunResponse(BaseModel):
    runs: list[ExploreRunItem]
```

After `_execute_resume`, add:

```python
async def _execute_explore(
    explore_id: str,
    target_url: str,
    credentials,   # ExploreCredentials | None
    depth_cap: int,
    page_cap: int,
) -> None:
    from app.tools.explorer import ExploreAgent
    agent = ExploreAgent(
        target_url=target_url,
        explore_id=explore_id,
        credentials=credentials,
        depth_cap=depth_cap,
        page_cap=page_cap,
    )
    try:
        await agent.run()
    except Exception:
        pass  # run() persists the failed map and emits an error event
    await emitter.close(explore_id)
```

After the `/runs/{run_id}/cost` endpoint, add:

```python
@app.post("/explore", response_model=ExploreResponse)
async def start_explore(req: ExploreRequest) -> ExploreResponse:
    from app.tools.explorer import ExploreCredentials
    explore_id = uuid.uuid4().hex
    creds = None
    if req.credentials is not None:
        creds = ExploreCredentials(
            username=req.credentials.username,
            password=req.credentials.password,
            username_selector=req.credentials.username_selector,
            password_selector=req.credentials.password_selector,
        )
    asyncio.create_task(
        _execute_explore(explore_id, req.target_url, creds, req.depth_cap, req.page_cap)
    )
    return ExploreResponse(explore_id=explore_id)


@app.get("/explore/{explore_id}/stream")
async def stream_explore(explore_id: str) -> EventSourceResponse:
    async def event_generator():
        async for event in emitter.subscribe(explore_id):
            yield {"event": event.type, "data": event.model_dump_json()}
    return EventSourceResponse(
        event_generator(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/explore/{explore_id}")
async def get_explore(explore_id: str):
    from fastapi import HTTPException
    from app.tools.explorer import AppMap, _DEFAULT_APP_MAPS_DIR
    matches = list(Path(_DEFAULT_APP_MAPS_DIR).rglob(f"{explore_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="AppMap not found")
    return AppMap.model_validate_json(matches[0].read_text(encoding="utf-8"))


@app.post("/explore/{explore_id}/runs", response_model=ExploreRunResponse)
async def forward_explore_flows(explore_id: str, req: ExploreRunRequest) -> ExploreRunResponse:
    from fastapi import HTTPException
    from app.tools.explorer import AppMap, _DEFAULT_APP_MAPS_DIR
    matches = list(Path(_DEFAULT_APP_MAPS_DIR).rglob(f"{explore_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail="AppMap not found")
    app_map = AppMap.model_validate_json(matches[0].read_text(encoding="utf-8"))
    flow_name_lower = {n.lower() for n in req.flow_names}
    runs: list[ExploreRunItem] = []
    for flow in app_map.flows:
        if flow.name.lower() not in flow_name_lower:
            continue
        story = f"Test the {flow.name}: {flow.description}"
        run_id = uuid.uuid4().hex
        asyncio.create_task(_execute_run(run_id, story, app_map.target_url))
        runs.append(ExploreRunItem(flow_name=flow.name, run_id=run_id))
    return ExploreRunResponse(runs=runs)
```

Also add `from pathlib import Path` at the top of `main.py` if not already present (check — it is not in the current imports; add it).

- [ ] **Step 4: Run endpoint tests to confirm pass**

```
cd backend && python -m pytest tests/test_stage_j.py::test_post_explore_returns_explore_id tests/test_stage_j.py::test_get_explore_returns_404_for_unknown tests/test_stage_j.py::test_explore_runs_forwards_flows -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```
git add app/main.py tests/test_stage_j.py && git commit -m "feat(explorer): /explore endpoints — Stage J Task 7"
```

---

## Task 8: Final integration — full suite green

**Files:** none changed

- [ ] **Step 1: Run the complete test suite**

```
cd backend && python -m pytest tests/ -v
```

Expected: all tests PASS, including all prior stage tests (B, D, E, F, G, H, I) and all Stage J tests.

- [ ] **Step 2: Verify import health**

```
cd backend && python -c "from app.main import app; from app.tools.explorer import ExploreAgent, AppMap, ExploreCredentials; from app.llm.prompts.explorer import EXPLORER_FLOW_SYSTEM; print('all imports ok')"
```

Expected: `all imports ok`

- [ ] **Step 3: Final commit**

```
git add -A && git commit -m "feat: Stage J Explorer Agent complete — BFS crawl, credential isolation, partial persistence, /explore endpoints, T1-T4 green"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task that implements it |
|-----------------|------------------------|
| `ExploreAgent` with target_url + optional credentials | Task 4 (`ExploreAgent.__init__`) |
| Login if credentials provided | Task 4 (`_login()`) |
| BFS discovery of same-origin pages | Task 4 (`_crawl()`) |
| Depth cap + page cap | Task 4 (`_crawl()` — `depth < self._depth_cap`, `pages_visited < self._page_cap`) |
| DOM probe reuse from `browser.py` | Task 4 (`from app.tools.browser import _EXTRACT_JS`) |
| Form detection | Task 4 (`_detect_forms()`) |
| Flow inference via reasoning LLM | Task 4 (`_infer_flows()`) |
| `AppMap` schema with `schema_version`, `target_url`, `explored_at` | Task 1 (`AppMap` model) |
| Storage keyed by `target_origin_slug` | Task 1 (`_origin_slug`, `_map_path`) |
| `status: exploring|complete|failed` | Task 1 (`AppMap.status` field) |
| Partial map on failure | Task 4 (`run()` except block + `save_map` after each page) |
| T4 partial map test | Task 6 |
| Post-login URL seeded to frontier | Task 4 (`run()` — `seed_urls.insert(0, post_login_url)`) |
| Credentials never in events/map/disk | Task 4 (`_login()` — `data={}`, no cred vars in any dict) |
| T3 credential isolation test | Task 6 |
| scheme+netloc same-origin check | Task 4 (`_is_same_origin`) |
| Destructive form guard | Task 4 (`_detect_forms`, `_DESTRUCTIVE_KEYWORDS`) |
| `NODE_TOOL_REGISTRY` entry | Task 3 |
| `POST /explore` | Task 7 |
| `GET /explore/{id}/stream` | Task 7 |
| `GET /explore/{id}` | Task 7 |
| `POST /explore/{id}/runs` with list of flow_names | Task 7 |
| T1 multi-page + login flow | Task 5 |
| T2 caps enforced | Task 4 |
| JS-only limitation documented | Task 4 (module docstring) |
| `ExploreRunRequest` no `target_url` field | Task 7 (model definition with comment) |
| Form field names in `_infer_flows` prompt | Task 2 (`build_explorer_flow_user` includes `fields`) |

All spec requirements covered. No gaps found.

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete.

**Type consistency:**
- `ExploreCredentials` defined Task 1, used Task 4 — matches.
- `AppMap`, `PageSnapshot`, `FormSnapshot`, `FlowSnapshot` defined Task 1, used throughout — match.
- `save_map(app_map, app_maps_dir)` signature consistent across all call sites.
- `_DEFAULT_APP_MAPS_DIR` defined Task 1, patched in Task 7 endpoint test — correct.
- `_execute_run` referenced in Task 7 endpoint — already defined in `main.py`.
