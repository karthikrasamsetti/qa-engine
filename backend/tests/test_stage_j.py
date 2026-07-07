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
