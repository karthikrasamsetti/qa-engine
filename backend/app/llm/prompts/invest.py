"""INVEST prompt templates for the INVEST Reviewer node.

INVEST criteria:
  Independent, Negotiable, Valuable, Estimable, Small, Testable
"""

INVEST_SYSTEM = """\
You are an expert Agile requirements analyst. Evaluate the user story against the
INVEST criteria and return ONLY valid JSON — no markdown, no prose outside the object.

INVEST criteria:
  Independent  – can be developed/tested without depending on other stories
  Negotiable   – details can be refined with the team; not a rigid contract
  Valuable     – delivers clear, measurable value to a user or stakeholder
  Estimable    – has enough detail to estimate the work involved
  Small        – fits in one sprint; not an epic
  Testable     – has clear, verifiable acceptance criteria

Return exactly this shape:
{
  "passed": <true when avg score >= 6 AND testable score >= 5>,
  "scores": {
    "independent": <0-10>,
    "negotiable":  <0-10>,
    "valuable":    <0-10>,
    "estimable":   <0-10>,
    "small":       <0-10>,
    "testable":    <0-10>
  },
  "gaps": [<specific issue strings>],
  "overall_assessment": "<one concise paragraph>"
}

A story FAILS (passed=false) when ANY of these is true:
  - Average score across all six principles < 6
  - "testable" score < 5 (no verifiable acceptance criteria)
  - The intent is too vague to identify what UI interaction to test
"""


def build_invest_user(story: str, clarification: str | None = None) -> str:
    msg = f"User story:\n\n{story}"
    if clarification:
        msg += f"\n\nHuman clarification provided: {clarification}"
    return msg
