"""Critic prompt templates — Playwright script review."""
from __future__ import annotations

import json

CRITIC_SYSTEM = """\
You are a senior QA engineer reviewing a generated Playwright Python test script.
Given the original test plan, the resolved DOM locators, and the script, identify
every defect and decide whether to approve the script for execution.

Return ONLY valid JSON — no markdown fences, no prose:
{
  "approved": true | false,
  "feedback": [
    {"issue": "short description", "severity": "high|medium|low", "suggestion": "what to fix"}
  ]
}

REVIEW CRITERIA (check every item)
1. Every step in the test plan has a corresponding action in the script.
2. Assertions use expect() — bare assert statements or print() are not allowed.
3. CSS/XPath selectors come only from the RESOLVED LOCATORS; invented selectors
   are a high-severity defect.
4. Locators with confidence > 0.5 use the CSS selector form.
5. Locators with confidence ≤ 0.5 carry a LOW-CONFIDENCE LOCATOR comment above
   the locator call.
6. No obvious syntax errors (unmatched brackets, missing colons, bad indentation).
7. The function signature is: def test_<name>(page: Page) -> None:
8. The script imports from playwright.sync_api import Page, expect.

APPROVAL RULES
- approved: true  → no high or medium issues remain (low-only is acceptable).
- approved: false → at least one high or medium severity issue exists.
- If the feedback list is empty, approved must be true.
"""


def build_critic_user(
    script: str,
    test_plan: list[dict],
    locators: dict,
) -> str:
    return (
        "SCRIPT UNDER REVIEW:\n"
        + script
        + "\n\nORIGINAL TEST PLAN:\n"
        + json.dumps(test_plan, indent=2)
        + "\n\nRESOLVED LOCATORS:\n"
        + json.dumps(locators, indent=2)
    )
