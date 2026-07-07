"""LLM prompts for the Explorer agent flow-inference step."""
from __future__ import annotations

EXPLORER_FLOW_SYSTEM = """\
You are a web-application analyst. You receive a list of pages discovered by an \
automated crawler: each page's URL, title, and any HTML forms with their field names.

Your task: identify user-facing *flows* — named sequences of pages that represent \
a coherent user journey (e.g. "login flow", "signup flow", "checkout flow", \
"password reset flow").

Return a JSON array of flow objects. Each object must have exactly these keys:
  "name"           — short lowercase slug (e.g. "login flow")
  "pages_involved" — list of URLs that are part of this flow, in order
  "description"    — one sentence describing what the user accomplishes

Return ONLY the JSON array with no prose, no markdown fences."""


def build_explorer_flow_user(pages: list[dict]) -> str:
    """Build the user-turn prompt for flow inference.

    Includes URL, title, and form field names (structural identifiers, not secrets).
    Excludes raw element details and any credential values.
    """
    lines: list[str] = ["Pages discovered:\n"]
    for p in pages:
        lines.append(f"URL: {p['url']}")
        lines.append(f"  Title: {p.get('title', '')}")
        for form in p.get("forms", []):
            fields = ", ".join(form.get("fields", []))
            action = form.get("action", "")
            lines.append(f"  Form → action={action!r} fields=[{fields}]")
        lines.append("")
    lines.append(
        "Return a JSON array of flow objects with keys: "
        "name, pages_involved, description."
    )
    return "\n".join(lines)
