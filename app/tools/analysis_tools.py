"""
Analysis, verification and export tools.

Three tools with three different permission shapes, each chosen for a reason §13 asks us to
state:

- ``calculate`` — Analyst only. Arithmetic on figures pulled from evidence, so a cost comparison
  is computed rather than estimated by a language model.
- ``validate_citations`` — Fact-Checker and Critic. **Fully deterministic**: set membership
  against the evidence store, no model involved. This is the concrete line between deterministic
  logic and agentic judgement that the excellence criteria ask for — the tool decides whether a
  citation *exists*; only the agent decides whether it *supports*.
- ``export_report`` — Writer only, and the only write tool outside research.
"""
from __future__ import annotations

import ast
import operator as op

from pydantic import BaseModel, Field

from app.schemas.common import AgentId
from app.tools.registry import ToolContext, ToolError, register_tool

# --------------------------------------------------------------------------- #
# 5. calculate
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an arithmetic AST. No names, calls, attributes or subscripts.

    ``eval`` is not used at any point. The expression reaching this tool originates from a model
    that has just read untrusted corpus text, so it is treated as hostile input: the grammar is
    an allow-list of numeric operations and nothing else parses.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("Only numeric literals are allowed.")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ToolError("Unsupported expression. Use numbers and + - * / % ** // only.")


class CalculateInput(BaseModel):
    expression: str = Field(
        min_length=1, max_length=200,
        description="Arithmetic only, e.g. '(96.60 - 45.00) / 96.60 * 100'",
    )
    explain: str = Field(default="", description="What this figure represents")


class CalculateOutput(BaseModel):
    expression: str
    result: float
    explain: str = ""


@register_tool(
    name="calculate",
    description=(
        "Evaluate an arithmetic expression over figures taken from evidence. Use for cost "
        "differences, percentages and totals rather than computing them mentally."
    ),
    input_model=CalculateInput,
    output_model=CalculateOutput,
    allowed_agents={AgentId.ANALYST},
    rationale=(
        "The Analyst is the only role that derives new quantities from evidence. Researchers "
        "record what a source says and must not compute figures a source never stated; the "
        "Writer reports numbers the Analyst already derived and verified."
    ),
)
def calculate(args: CalculateInput, ctx: ToolContext) -> CalculateOutput:
    try:
        tree = ast.parse(args.expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"Could not parse expression: {args.expression!r}") from e
    try:
        value = _safe_eval(tree)
    except ZeroDivisionError as e:
        raise ToolError("Division by zero.") from e
    if value != value or value in (float("inf"), float("-inf")):
        raise ToolError("Expression produced a non-finite result.")
    return CalculateOutput(expression=args.expression, result=round(value, 6),
                           explain=args.explain)


# --------------------------------------------------------------------------- #
# 6. validate_citations
# --------------------------------------------------------------------------- #
class ValidateInput(BaseModel):
    evidence_ids: list[str] = Field(
        default_factory=list, description="All evidence ids cited by the text under review"
    )


class ValidateOutput(BaseModel):
    checked: int = 0
    valid_ids: list[str] = Field(default_factory=list)
    fabricated_ids: list[str] = Field(default_factory=list)
    all_valid: bool = True
    verdict: str = ""


@register_tool(
    name="validate_citations",
    description=(
        "Check whether cited evidence ids exist in the evidence store. Purely mechanical: it "
        "reports existence, never whether the evidence supports the claim."
    ),
    input_model=ValidateInput,
    output_model=ValidateOutput,
    allowed_agents={AgentId.FACT_CHECKER, AgentId.CRITIC},
    rationale=(
        "Both reviewing roles need it and neither can be trusted to do it by eye — spotting that "
        "'E9' was never stored is exactly the check a language model performs unreliably and a "
        "set lookup performs perfectly. Withheld from the Analyst and Writer on purpose: an "
        "author checking its own citations is not verification."
    ),
)
def validate_citations(args: ValidateInput, ctx: ToolContext) -> ValidateOutput:
    known = {e.evidence_id for e in ctx.evidence}
    valid = [i for i in args.evidence_ids if i in known]
    fabricated = [i for i in args.evidence_ids if i not in known]
    all_valid = not fabricated
    if not args.evidence_ids:
        verdict = "No citations were supplied to check."
        all_valid = False   # nothing verified is not the same as nothing wrong
    elif all_valid:
        verdict = f"All {len(valid)} cited ids exist in the evidence store."
    else:
        verdict = (
            f"{len(fabricated)} cited id(s) do not exist: {', '.join(fabricated)}. "
            f"Any claim resting on them is unsupported."
        )
    return ValidateOutput(checked=len(args.evidence_ids), valid_ids=valid,
                          fabricated_ids=fabricated, all_valid=all_valid, verdict=verdict)


# --------------------------------------------------------------------------- #
# 7. export_report
# --------------------------------------------------------------------------- #
class ExportInput(BaseModel):
    title: str = Field(min_length=1)
    markdown: str = Field(min_length=1)


class ExportOutput(BaseModel):
    exported: bool
    persisted: bool
    characters: int
    note: str = ""


@register_tool(
    name="export_report",
    description="Persist the final report as Markdown against this run.",
    input_model=ExportInput,
    output_model=ExportOutput,
    allowed_agents={AgentId.WRITER},
    is_write=True,
    rationale=(
        "The Writer produces the deliverable, so it alone may emit one. Restricting export also "
        "keeps a single, auditable moment at which a run's output becomes final — useful for the "
        "human review checkpoint that immediately follows it."
    ),
)
def export_report(args: ExportInput, ctx: ToolContext) -> ExportOutput:
    persisted = False
    note = ""
    if ctx.store is not None and getattr(ctx.store, "enabled", False) and ctx.run_id:
        try:
            ctx.store.save_report(ctx.run_id, args.title, args.markdown)
            persisted = True
        except Exception as e:  # noqa: BLE001
            note = f"Report generated but not persisted: {type(e).__name__}"
    else:
        note = "Report generated; persistence is disabled for this run."
    return ExportOutput(exported=True, persisted=persisted,
                        characters=len(args.markdown), note=note)
