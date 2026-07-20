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

from app.tools.browser import _EXTRACT_JS

logger = logging.getLogger(__name__)

_DEFAULT_APP_MAPS_DIR = Path(__file__).parent.parent.parent / "data" / "app_maps"

_DESTRUCTIVE_KEYWORDS = frozenset({
    "delete", "remove", "cancel", "pay", "checkout",
    "purchase", "unsubscribe",
})


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


# ---------------------------------------------------------------------------
# Form detection
# ---------------------------------------------------------------------------

def _detect_forms(elements: list[dict[str, Any]]) -> list[FormSnapshot]:
    """Infer form snapshots from the _EXTRACT_JS element list.

    Groups input fields + submit triggers into one logical form per page
    (v1 simplification). Tags destructive=True when any element text/name
    contains a destructive keyword.
    """
    inputs = [e for e in elements if e.get("tag") in ("input", "textarea", "select")]
    submits = [
        e for e in elements
        if e.get("type") in ("submit",) or e.get("tag") == "button"
    ]

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
                try:
                    page = await browser.new_page()

                    seed_urls = [self._target_url]
                    if self._credentials is not None:
                        post_login_url = await self._login(page)
                        if post_login_url and post_login_url != self._target_url:
                            seed_urls.insert(0, post_login_url)

                    await self._crawl(page, app_map, seed_urls)
                finally:
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
        # frontier: (url, depth)
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

            # SPA timing fix: wait for render before extracting elements.
            # Best-effort — some apps have long-polling so we cap at 5s.
            try:
                await page.wait_for_load_state("networkidle", timeout=5_000)
            except (PWTimeout, Exception):
                pass

            # Use the actual landed URL (resolves redirects)
            landed_url = page.url
            title = await page.title()

            try:
                elements: list[dict] = await page.evaluate(_EXTRACT_JS)
            except Exception as exc:
                logger.warning("DOM extraction failed for %s: %s", landed_url, exc)
                elements = []

            forms = _detect_forms(elements)
            snapshot = PageSnapshot(
                url=landed_url, title=title, elements=elements, forms=forms
            )
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
                    # Strip fragments; normalise trailing slash
                    href = href.split("#")[0]
                    if href.endswith("/") and href != self._target_origin + "/":
                        href = href.rstrip("/")
                    if href and href not in visited and self._is_same_origin(href):
                        frontier.append((href, depth + 1))

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def _login(self, page) -> str | None:
        """Fill and submit the login form.

        Waits for the login form to appear before probing selectors so that
        SPAs that render forms asynchronously (after domcontentloaded) are
        handled correctly.

        Credential values are accessed inline and never placed in any dict,
        event data field, or log statement.
        """
        from playwright.async_api import TimeoutError as PWTimeout
        from app.streaming.events import emitter

        # data= intentionally empty — credentials must never appear here
        await emitter.emit(
            self._explore_id, self.AGENT, self.PHASE, "tool_call",
            "Performing login",
            data={},
        )

        try:
            await page.goto(self._target_url, wait_until="domcontentloaded", timeout=15_000)

            # SPA timing fix: wait for any auth field before probing selectors.
            # JS-rendered forms appear after domcontentloaded; this blocks until
            # at least one auth-related input is in the DOM (or 10s elapses).
            try:
                await page.wait_for_selector(
                    "input[name='username'], input[name*='user' i], "
                    "input[type='email'], input[name*='email' i], input[type='password']",
                    timeout=10_000,
                )
            except PWTimeout:
                pass  # Handled by the "field not found" checks below

            tried: list[str] = []
            u_sel = self._credentials.username_selector
            if not u_sel:
                for candidate in (
                    "input[name='username']",
                    "input[name*='user' i]",
                    "input[type='email']",
                    "input[name*='email' i]",
                    "form:has(input[type='password']) input[type='text']",
                ):
                    tried.append(candidate)
                    if await page.locator(f"{candidate}:visible").count() > 0:
                        u_sel = f"{candidate}:visible"
                        break

            p_sel = self._credentials.password_selector or "input[type='password']:visible"

            if not u_sel or await page.locator(u_sel).count() == 0:
                logger.warning("Login: username field not found (tried: %s)", tried)
                await emitter.emit(
                    self._explore_id, self.AGENT, self.PHASE,
                    "error", "Login failed: username field not found",
                    data={"tried_selectors": tried},
                )
                return None

            if await page.locator(p_sel).count() == 0:
                logger.warning("Login: password field not found")
                await emitter.emit(
                    self._explore_id, self.AGENT, self.PHASE,
                    "error", "Login failed: password field not found",
                    data={"tried_selectors": [p_sel]},
                )
                return None

            # Fill — values read inline, never stored in a loggable variable
            await page.fill(u_sel, self._credentials.username)
            await page.fill(p_sel, self._credentials.password)

            # Click submit, fall back to Enter
            submit_sel = (
                "button[type='submit'], input[type='submit'], "
                "button:has-text('Login'), button:has-text('Sign in'), "
                "button:has-text('Sign In')"
            )
            if await page.locator(submit_sel).count() > 0:
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
            raw = json.loads(match.group())
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
