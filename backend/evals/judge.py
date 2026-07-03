"""LLM-as-judge evaluation script for the QA Engine INVEST gate.

Run from the backend/ directory:
    python -m evals.judge            # full eval with LLM quality scores
    python -m evals.judge --no-judge # skip quality scorer (faster, no LLM cost)
    python evals/judge.py

Loads evals/evalset.jsonl, runs each story through guardrail → invest,
compares actual vs expected verdicts, uses the reasoning LLM to score
verdict quality, and prints a formatted scoreboard.

The HITL-interrupt inside invest_node is patched to return a placeholder
clarification so evaluation is non-blocking.  Whether HITL was triggered is
recorded via mock.called rather than from the interrupted execution path.

Output fields per story:
  id                  — eval entry id
  story               — first 80 chars of story text
  expected_invest_pass — true/false from evalset label
  expected_hitl        — true/false from evalset label
  actual_invest_pass   — observed outcome
  actual_hitl          — observed outcome
  invest_correct       — actual_invest_pass matches expected
  hitl_correct         — actual_hitl matches expected
  correct              — both invest_correct and hitl_correct
  invest_verdict       — initial verdict dict (pre-clarification for HITL stories)
  latency_ms           — wall-clock time for guardrail+invest
  judge_score          — 0.0–1.0 quality score from reasoning LLM (None if skipped)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Ensure backend/ is on the import path when run directly as a script.
_HERE = Path(__file__).parent
_BACKEND = _HERE.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.graph.nodes.guardrail import guardrail_node
from app.graph.nodes.invest import invest_node
from app.llm.client import llm_client
from app.streaming.events import emitter

logger = logging.getLogger(__name__)

_EVALSET = _HERE / "evalset.jsonl"

# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are evaluating the accuracy of an AI system's INVEST analysis of agile user stories.
Given a user story, the expected verdict (pass or fail INVEST), and the AI's actual verdict,
decide whether the AI's classification is correct and score its quality.

Return ONLY valid JSON — no markdown fences, no prose:
{"score": <0.0 to 1.0>, "reasoning": "<one concise sentence>"}

score=1.0 — verdict is clearly correct and well-reasoned
score=0.5 — verdict is borderline or partially justified
score=0.0 — verdict is clearly wrong

Consider:
- Does the actual verdict match the expected one?
- Are the scores proportionate to the story's actual specificity?
- Are the identified gaps accurate?
"""


def _build_judge_user(story: str, verdict: dict, expected_pass: bool) -> str:
    return (
        f"USER STORY:\n{story}\n\n"
        f"EXPECTED: {'PASS (story is specific and testable)' if expected_pass else 'FAIL (story is vague)'}\n\n"
        f"AI VERDICT:\n{json.dumps(verdict, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Single-story evaluation
# ---------------------------------------------------------------------------

async def _run_invest_for_eval(story: str, run_id: str) -> tuple[dict, bool]:
    """Run story through guardrail → invest; return (initial_verdict, hitl_triggered).

    Patches interrupt() so HITL does not block — we record whether it was
    called (hitl_triggered=True means the initial INVEST verdict was failed).
    """
    state: dict = {
        "run_id": run_id,
        "raw_input": story,
        "status": "running",
    }

    # Guardrail has no LLM, no interrupt — safe to call directly.
    guardrail_result = await guardrail_node(state)
    state.update(guardrail_result)

    with patch("app.graph.nodes.invest.interrupt", return_value="[eval-clarification]") as mock_int:
        invest_result = await invest_node(state)
        hitl_triggered: bool = mock_int.called

    # Close emitter queue to avoid resource leaks across eval stories.
    await emitter.close(run_id)

    if hitl_triggered:
        # HITL was triggered → the initial verdict failed INVEST.
        # Reconstruct the pre-clarification verdict from hitl_request context.
        hitl_req = invest_result.get("hitl_request", {})
        ctx = hitl_req.get("context", {})
        initial_verdict: dict = {
            "passed": False,
            "scores": ctx.get("scores", {}),
            "gaps":   ctx.get("gaps", []),
            "overall_assessment": "Initial INVEST verdict: FAIL (HITL triggered).",
        }
    else:
        initial_verdict = invest_result.get("invest_verdict", {"passed": True})

    return initial_verdict, hitl_triggered


async def _judge_quality(story: str, verdict: dict, expected_pass: bool) -> float:
    """Ask the reasoning LLM to score verdict quality (0.0–1.0)."""
    try:
        resp = await llm_client.complete(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": _build_judge_user(story, verdict, expected_pass)},
            ],
            model_tier="reasoning",
            max_tokens=256,
        )
        m = re.search(r'\{.*\}', resp.text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
    except Exception as exc:
        logger.warning("judge_quality failed — %s", exc)
    return 0.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

async def run_evalset(
    evalset_path: Path | None = None,
    use_judge: bool = True,
) -> dict:
    """Evaluate every story in the evalset and return a scoreboard.

    Args:
        evalset_path: JSONL evalset to load. Defaults to evals/evalset.jsonl.
        use_judge:    When True, each story gets an LLM-as-judge quality score.

    Returns:
        {
          "total":    int,
          "correct":  int,
          "accuracy": float,
          "stories":  [
            {
              "id": str,
              "story": str,                  # truncated to 80 chars
              "expected_invest_pass": bool,
              "expected_hitl": bool,
              "actual_invest_pass": bool,
              "actual_hitl": bool,
              "invest_correct": bool,
              "hitl_correct": bool,
              "correct": bool,
              "invest_verdict": dict,
              "latency_ms": float,
              "judge_score": float | None,
            }
          ],
        }
    """
    path = evalset_path or _EVALSET
    entries: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    stories_out: list[dict[str, Any]] = []
    total_correct = 0

    for entry in entries:
        run_id = f"eval-{entry['id']}"
        story_text: str = entry["story"]
        expected_pass: bool = bool(entry.get("expected_invest_pass", True))
        expected_hitl: bool = bool(entry.get("expected_hitl", False))

        t0 = time.perf_counter()
        verdict, hitl_triggered = await _run_invest_for_eval(story_text, run_id)
        latency_ms = (time.perf_counter() - t0) * 1000

        actual_invest_pass: bool = not hitl_triggered
        actual_hitl: bool = hitl_triggered

        invest_correct = actual_invest_pass == expected_pass
        hitl_correct   = actual_hitl == expected_hitl
        correct        = invest_correct and hitl_correct
        if correct:
            total_correct += 1

        judge_score: float | None = None
        if use_judge:
            judge_score = await _judge_quality(story_text, verdict, expected_pass)

        stories_out.append({
            "id":                   entry["id"],
            "story":                story_text[:80] + ("…" if len(story_text) > 80 else ""),
            "expected_invest_pass": expected_pass,
            "expected_hitl":        expected_hitl,
            "actual_invest_pass":   actual_invest_pass,
            "actual_hitl":          actual_hitl,
            "invest_correct":       invest_correct,
            "hitl_correct":         hitl_correct,
            "correct":              correct,
            "invest_verdict":       verdict,
            "latency_ms":           round(latency_ms, 1),
            "judge_score":          judge_score,
        })

    total = len(entries)
    accuracy = total_correct / total if total > 0 else 0.0

    return {
        "total":    total,
        "correct":  total_correct,
        "accuracy": accuracy,
        "stories":  stories_out,
    }


# ---------------------------------------------------------------------------
# Scoreboard printer
# ---------------------------------------------------------------------------

def _print_scoreboard(scoreboard: dict) -> None:
    stories = scoreboard["stories"]
    w = 70
    print("\n" + "=" * w)
    print(f"  QA Engine INVEST Evaluation  —  {scoreboard['total']} stories")
    print("=" * w)
    print(f"  {'ID':12s}  {'Exp':4s} {'Act':4s}  {'Lat':>6s}  {'Judge':>6s}  Status")
    print("-" * w)
    for s in stories:
        mark = "✓" if s["correct"] else "✗"
        exp = "PASS" if s["expected_invest_pass"] else "FAIL"
        act = "PASS" if s["actual_invest_pass"] else "FAIL"
        lat = f"{s['latency_ms']:6.0f}ms"
        jsc = f"{s['judge_score']:5.2f}" if s["judge_score"] is not None else "  n/a"
        print(f"  {mark} {s['id']:12s}  {exp:4s} {act:4s}  {lat}  {jsc}  "
              f"{'OK' if s['correct'] else 'WRONG'}")
    print("=" * w)
    print(
        f"  Accuracy : {scoreboard['accuracy']:.1%}  "
        f"({scoreboard['correct']}/{scoreboard['total']} correct)"
    )
    if any(s["judge_score"] is not None for s in stories):
        scores = [s["judge_score"] for s in stories if s["judge_score"] is not None]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  Avg judge: {avg:.2f}  (LLM-as-judge quality score, 0-1)")
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    use_judge = "--no-judge" not in sys.argv
    scoreboard = await run_evalset(use_judge=use_judge)
    _print_scoreboard(scoreboard)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_main())
