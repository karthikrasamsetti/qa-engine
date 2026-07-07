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
