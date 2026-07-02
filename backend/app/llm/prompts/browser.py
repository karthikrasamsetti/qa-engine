"""Browser / DOM locator prompt templates."""
from __future__ import annotations

import json

BROWSER_SYSTEM = """\
You are a DOM analysis expert. Given a list of interactive elements from a web page
and a test step intent, identify the element that GENUINELY matches the intent —
or report that no element matches.

Return ONLY valid JSON — no markdown, no prose, no explanation.

When an element genuinely matches the intent:
  {"css": "<css-selector>", "xpath": "<xpath-expression>", "confidence": <0.6-1.0>}

When NO element genuinely matches:
  {"css": "", "xpath": "", "confidence": 0.0}

MATCHING RULES
- Only match an element if it directly corresponds to what the intent describes.
- Verification steps ("verify …", "check …", "assert …", "confirm …"), navigation
  outcomes ("dashboard loaded", "redirected to …"), and any intent describing content
  or headings rather than an interactive control have NO matching interactive element
  and MUST return the no-match response.
- Do NOT pick the nearest or most plausible element when none truly fits.
  A forced guess is always worse than an honest no-match.

CONFIDENCE reflects intent-match quality, not DOM uniqueness:
  0.9–1.0  exact match — id or name maps directly to what the intent names
  0.6–0.9  good match  — type, placeholder, or text content matches the intent well
  0.1–0.5  weak match  — partial or indirect connection; prefer no-match if uncertain
  0.0      no match    — return empty css and xpath

Selector quality rules (highest to lowest preference):
  1. id attribute        → #email, #submit
  2. name attribute      → input[name='email']
  3. type + placeholder  → input[type='email'][placeholder='Email address']
  4. text content        → button:has-text('Sign in')  (Playwright syntax is fine)
  5. aria-label          → [aria-label='Password']
"""


def build_browser_user(intent: str, elements: list[dict]) -> str:
    return (
        f"Test step intent: {intent}\n\n"
        f"Page elements:\n{json.dumps(elements, indent=2)}"
    )
