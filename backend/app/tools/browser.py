"""Playwright-based DOM locator tool.

Opens a headless Chromium page, extracts interactive elements, calls our LLM
(fast tier) to identify the element matching a step intent, then verifies the
suggested CSS selector actually exists on the page before returning.

Typical use: one BrowserSession per ui_mapper run (reuses the browser across
all steps on the same URL, avoids spawning a new process per step).
"""
from __future__ import annotations

import json
import logging
import re

from playwright.async_api import (
    Browser,
    Page,
    async_playwright,
)
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from app.llm.client import llm_client
from app.llm.prompts.browser import BROWSER_SYSTEM, build_browser_user

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)

# JS function injected into the page to extract interactive elements.
# Avoids template f-strings to keep the JS literal free of escaping noise.
_EXTRACT_JS = """
() => {
    const sel = [
        "input:not([type='hidden'])",
        "button",
        "a[href]",
        "select",
        "textarea",
        "[role='button']",
        "[role='link']"
    ].join(", ");
    const out = [];
    document.querySelectorAll(sel).forEach((el) => {
        out.push({
            tag:         el.tagName.toLowerCase(),
            id:          el.id || null,
            name:        el.getAttribute('name') || null,
            type:        el.getAttribute('type') || null,
            placeholder: el.getAttribute('placeholder') || null,
            aria_label:  el.getAttribute('aria-label') || null,
            text:        (el.innerText || el.value || '').substring(0, 60).trim() || null,
            href:        el.tagName === 'A' ? el.getAttribute('href') : null,
        });
    });
    return out;
}
"""

_EMPTY_LOCATOR: dict = {"css": "", "xpath": "", "confidence": 0.0}


def _parse_locator(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        return dict(_EMPTY_LOCATOR)
    try:
        d = json.loads(m.group())
        return {
            "css":        str(d.get("css", "")),
            "xpath":      str(d.get("xpath", "")),
            "confidence": float(d.get("confidence", 0.0)),
        }
    except (json.JSONDecodeError, ValueError):
        return dict(_EMPTY_LOCATOR)


class BrowserSession:
    """Reusable headless browser session.

    Keeps a single Chromium instance open across multiple `find_locators` calls.
    Caches the extracted elements as long as the URL stays the same — avoids
    re-running the DOM extraction script for every step of the same page.

    Usage::

        async with BrowserSession() as session:
            for step in test_plan:
                locator = await session.find_locators(url, step["intent"], run_id=run_id)
    """

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._current_url: str | None = None
        self._cached_elements: list[dict] | None = None

    async def __aenter__(self) -> "BrowserSession":
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)
        self._page = await self._browser.new_page()
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def find_locators(
        self,
        url: str,
        intent: str,
        run_id: str | None = None,
    ) -> dict:
        """Navigate to *url* (cached per session), extract DOM, ask LLM for locator.

        Also verifies the suggested CSS selector resolves on the page; lowers
        confidence to 0.1 when it doesn't, rather than crashing.

        Returns ``{"css": str, "xpath": str, "confidence": float}``.
        """
        assert self._page is not None, "BrowserSession not entered"

        # Navigate only if URL has changed since last call.
        if url != self._current_url:
            try:
                await self._page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                self._current_url = url
                self._cached_elements = None
            except PlaywrightTimeoutError:
                logger.warning("Page load timeout: %s", url)
                return dict(_EMPTY_LOCATOR)
            except Exception as exc:
                logger.warning("Navigation error for %s: %s", url, exc)
                return dict(_EMPTY_LOCATOR)

        # Extract interactive elements (cached per URL load).
        if self._cached_elements is None:
            try:
                self._cached_elements = await self._page.evaluate(_EXTRACT_JS)
            except Exception as exc:
                logger.warning("DOM extraction failed: %s", exc)
                return dict(_EMPTY_LOCATOR)

        elements = self._cached_elements or []
        if not elements:
            logger.warning("No interactive elements found at %s", url)
            return dict(_EMPTY_LOCATOR)

        # Ask LLM (fast tier) to pick the matching element.
        try:
            resp = await llm_client.complete(
                messages=[
                    {"role": "system", "content": BROWSER_SYSTEM},
                    {"role": "user",   "content": build_browser_user(intent, elements)},
                ],
                model_tier="fast",
                run_id=run_id,
            )
            locator = _parse_locator(resp.text)
        except Exception as exc:
            logger.error("LLM call failed in browser tool: %s", exc)
            return dict(_EMPTY_LOCATOR)

        # Verify the CSS selector against the live page.
        # High-confidence only when it resolves to exactly one element.
        css = locator.get("css", "")
        if css:
            try:
                count = await self._page.locator(css).count()
                if count != 1:
                    logger.warning(
                        "LLM selector %r matched %d element(s) (need exactly 1); lowering confidence",
                        css, count,
                    )
                    locator["confidence"] = 0.1
            except Exception as exc:
                logger.warning("Selector verification failed for %r: %s", css, exc)
                locator["confidence"] = 0.1

        # Verify the XPath the same way.
        xpath = locator.get("xpath", "")
        if xpath:
            try:
                count = await self._page.locator("xpath=" + xpath).count()
                if count != 1:
                    logger.warning(
                        "LLM xpath %r matched %d element(s) (need exactly 1); lowering confidence",
                        xpath, count,
                    )
                    locator["confidence"] = 0.1
            except Exception as exc:
                logger.warning("XPath verification failed for %r: %s", xpath, exc)
                locator["confidence"] = 0.1

        return locator


async def find_locators(
    url: str,
    intent: str,
    run_id: str | None = None,
) -> dict:
    """Single-call convenience wrapper: opens a fresh BrowserSession per lookup."""
    async with BrowserSession() as session:
        return await session.find_locators(url, intent, run_id=run_id)
