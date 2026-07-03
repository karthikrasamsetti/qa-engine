"""Synthesis Agent prompt templates."""
from __future__ import annotations

import json

SYNTHESIS_SYSTEM = """\
You are a QA reporting agent. Given an automated test execution result, produce
a clear, stakeholder-friendly report. Non-technical stakeholders will read this.

Return ONLY valid JSON — no markdown fences, no prose — in this exact shape:
{
  "verdict":        "pass" | "bug" | "flaky",
  "outcome_label":  "short human-readable outcome (5–10 words)",
  "summary":        "1–2 sentence executive summary of what happened",
  "tested":         "what feature or scenario was exercised by this test run",
  "assessment":     "for failures: is this a real app bug or a test-setup issue? for passes: confirmation the feature works as specified",
  "recommendation": "what should happen next (e.g. 'No action needed', 'Raise a bug report', 'Review test infrastructure')"
}

VERDICT RULES
  pass   — all assertions succeeded; the feature behaves as specified
  bug    — an assertion failed (wrong value, wrong URL, wrong state); this is a
           genuine application defect, not a test-setup problem
  flaky  — the test could not run cleanly due to infrastructure reasons
           (element not found, network timeout, Docker unavailable); the
           underlying feature MAY be working; the test environment needs review

CLASSIFICATION GUIDANCE
  AssertionError | "Locator expected to have" | "Expected … Received"  → bug
  TimeoutError waiting for locator | ElementHandle not attached |
  "Sandbox timed out" | Docker error                                    → flaky
  exit_code=0, all assertions passed                                    → pass

TONE: professional, concise, non-alarmist. Do not paste raw stack traces into
the summary. Use the pre-classification hint when provided; you may override it
only if the execution evidence clearly contradicts it.
"""


def build_synthesis_user(
    story: str,
    test_plan: list[dict],
    execution_result: dict,
    heal_attempts: int,
    pre_verdict: str,
) -> str:
    passed      = execution_result.get("passed", False)
    error       = execution_result.get("error") or ""
    logs        = execution_result.get("logs") or ""
    logs_excerpt = logs[:600] if len(logs) > 600 else logs

    parts = [
        f"PRE-CLASSIFICATION HINT: {pre_verdict}",
        f"\nUSER STORY:\n{story}",
        f"\nTEST PLAN ({len(test_plan)} step(s)):\n" + json.dumps(test_plan, indent=2),
        f"\nEXECUTION RESULT:",
        f"  passed:        {passed}",
        f"  heal_attempts: {heal_attempts}",
    ]
    if error:
        parts.append(f"  error:         {error[:400]}")
    if logs_excerpt:
        parts.append(f"\nLOGS (excerpt):\n{logs_excerpt}")

    return "\n".join(parts)
