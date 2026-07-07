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


# ---------------------------------------------------------------------------
# Task 3: audit registry
# ---------------------------------------------------------------------------

def test_explorer_agent_in_node_tool_registry():
    from app.observability.audit import NODE_TOOL_REGISTRY
    assert "explorer agent" in NODE_TOOL_REGISTRY
    allowed = NODE_TOOL_REGISTRY["explorer agent"]
    assert "browser:navigate" in allowed
    assert "llm:reasoning" in allowed


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


def _start_fixture_server(
    pages: dict[str, str],
) -> tuple[str, http.server.HTTPServer]:
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
        return LLMResponse(
            text="[]", input_tokens=1, output_tokens=1, model="mock", cost_usd=0.0
        )
    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_stub)

    # 15-page chain: / → /p/1 → /p/2 → … → /p/14
    def _chain_page(n: int) -> str:
        nxt = f"/p/{n + 1}" if n < 14 else ""
        link = f"<a href='{nxt}'>next</a>" if nxt else ""
        return (
            f"<html><head><title>Page {n}</title></head>"
            f"<body>{link}</body></html>"
        )

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
    # At depth_cap=1, /p/2 and beyond are depth-2 links and must not appear
    for i in range(2, 15):
        assert f"/p/{i}" not in visited_paths, (
            f"depth-2 page /p/{i} must not be visited"
        )


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
        return LLMResponse(
            text=_FLOW_JSON, input_tokens=10, output_tokens=50,
            model="mock", cost_usd=0.0,
        )
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
    assert base_url.rstrip("/") in visited_urls, (
        f"Home page missing; visited: {visited_urls}"
    )
    assert base_url.rstrip("/") + "/login" in visited_urls, (
        f"/login missing; visited: {visited_urls}"
    )
    assert app_map.status == "complete", f"Expected complete, got {app_map.status}"
    assert len(app_map.flows) == 1, f"Expected 1 flow, got {app_map.flows}"
    assert app_map.flows[0].name == "login flow"


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
        return LLMResponse(
            text="[]", input_tokens=1, output_tokens=1, model="mock", cost_usd=0.0
        )
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

    # Drain all events for this run from the asyncio queue
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
            assert term not in ev.message, (
                f"{term!r} found in event message: {ev.message!r}"
            )
            data_str = json.dumps(ev.data)
            assert term not in data_str, (
                f"{term!r} found in event data: {data_str!r}"
            )

    # Check the persisted JSON file
    files = list(tmp_path.rglob(f"{explore_id}.json"))
    assert files, "AppMap file not found on disk"
    content = files[0].read_text(encoding="utf-8")
    for term in SECRET_TERMS:
        assert term not in content, (
            f"{term!r} found in persisted AppMap JSON"
        )


# ---------------------------------------------------------------------------
# T4 — partial map persisted on failure
# ---------------------------------------------------------------------------

async def test_partial_map_persisted_on_failure(tmp_path, monkeypatch):
    """When _crawl raises after 2 pages, map has status=failed, error, and 2 pages."""
    from app.tools.explorer import ExploreAgent, PageSnapshot, save_map
    from app.llm import client as llm_mod
    from app.llm.client import LLMResponse

    async def _llm_stub(messages, model_tier, **kw):
        return LLMResponse(
            text="[]", input_tokens=1, output_tokens=1, model="mock", cost_usd=0.0
        )
    monkeypatch.setattr(llm_mod.llm_client, "complete", _llm_stub)

    # Patch _crawl to save 2 fake pages then raise — tests run() error handling
    async def _patched_crawl(self, page, app_map, seed_urls):
        for i in range(2):
            snap = PageSnapshot(url=f"{self._target_url}/p/{i}", title=f"Page {i}")
            app_map.pages.append(snap)
            save_map(app_map, self._app_maps_dir)
        raise RuntimeError("Simulated navigation failure on page 3")

    monkeypatch.setattr(ExploreAgent, "_crawl", _patched_crawl)

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
    assert len(raw["pages"]) == 2, (
        f"Expected 2 pages before crash, got {len(raw['pages'])}"
    )


# ---------------------------------------------------------------------------
# Task 7: endpoint smoke tests (sync — avoids async/TestClient loop conflicts)
# ---------------------------------------------------------------------------

def test_post_explore_returns_explore_id(monkeypatch):
    """POST /explore returns {explore_id} without running Playwright."""
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
    """GET /explore/{id} returns 404 when no map exists."""
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
