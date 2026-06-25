"""Browser / DOM locator prompt templates."""
from __future__ import annotations

import json

BROWSER_SYSTEM = """\
You are a DOM analysis expert. Given a list of interactive elements from a web page
and a test step intent, identify the single best-matching element and return its locators.

Return ONLY valid JSON — no markdown, no prose, no explanation:
{"css": "<css-selector>", "xpath": "<xpath-expression>", "confidence": 0.95}

Selector quality rules (highest to lowest preference):
  1. id attribute   → #email, #submit
  2. name attribute → input[name='email']
  3. type + placeholder → input[type='email'][placeholder='Email address']
  4. text content   → button:has-text('Sign in')  (Playwright syntax is fine)
  5. aria-label     → [aria-label='Password']

- confidence: 0.0 (no match) to 1.0 (exact unique id match)
- If no element matches the intent, return {"css": "", "xpath": "", "confidence": 0.0}
- Do NOT return multiple candidates
"""


def build_browser_user(intent: str, elements: list[dict]) -> str:
    return (
        f"Test step intent: {intent}\n\n"
        f"Page elements:\n{json.dumps(elements, indent=2)}"
    )
