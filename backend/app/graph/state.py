"""Shared LangGraph state for the QA Engine.

This TypedDict is the spine of the whole system. Every node reads from and writes
to it. FREEZE this schema after Stage A — changing it ripples through every node
and the frontend event payloads.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict


class QAState(TypedDict, total=False):
    """State carried through every node of the QA workflow graph.

    `total=False` so nodes can populate fields incrementally; early nodes won't
    have set the later fields yet.
    """

    # --- identity / lifecycle ---
    run_id: str
    status: Literal["running", "paused_hitl", "done", "failed"]

    # --- Phase 1: ingestion, context, validation ---
    raw_input: str                  # user story text OR a Jira ticket ID
    target_url: str                 # web app under test (set on the run request)
    jira_context: dict              # {story, epics, related_tickets} from MCP
    sanitized_story: str            # guardrail output, safe to feed a model
    invest_verdict: dict            # {passed: bool, scores: {...}, gaps: [...]}

    # --- Human-in-the-loop ---
    hitl_request: Optional[dict]    # {reason, question, context} when paused
    hitl_response: Optional[str]    # clarification injected by the human on resume

    # --- Phase 1.5: test case design ---
    scenarios: list[dict]          # [{scenario_id, title, type, preconditions, description, expected_outcome}]
    review_scenarios: bool         # when True, pause for HITL after scenario generation

    # --- Phase 2: scenario generation & UI mapping ---
    test_plan: list[dict]           # [{step_id, intent, action, expected}, ...]
    locators: dict                  # {step_id: {css, xpath, confidence}}

    # --- Phase 3: script generation (reflection loop) ---
    script: str                     # generated Playwright code
    critic_feedback: list[dict]     # [{issue, severity, suggestion}, ...]
    critic_approved: bool
    reflection_count: int           # iterations of scaffolder<->critic

    # --- Phase 4: execution & self-healing ---
    execution_result: dict          # {passed, logs, error, screenshots, exit_code}
    heal_attempts: int              # locator self-heal retries

    # --- Phase 5: reporting & memory ---
    report: dict                    # synthesis output for stakeholders
