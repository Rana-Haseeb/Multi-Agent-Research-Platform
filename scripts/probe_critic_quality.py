"""
Model-selection probe for the hardest role: the Critic.

The Critic exists to catch weak reasoning, so it must not be the weakest reasoner available.
This measures that directly by handing each model an analysis with **six deliberately planted
defects** and scoring how many it finds — no LLM judge, no subjective read.

The planted defects mirror the review criteria in §18:
  D1 unsupported   — a major claim with no citation at all
  D2 fabricated    — cites E9, which does not exist in the evidence list
  D3 contradiction — two statements that cannot both be true
  D4 overgeneral   — a sweeping conclusion drawn from a single data point
  D5 irrelevant    — cites real evidence that does not support the claim made
  D6 missing       — the recommendation rests on a criterion never researched

This doubles as the seed of the **Critic Detection Rate** metric (§29). Because the defects are
known, the metric can genuinely fail — a model that finds nothing scores 0/6, which is the
property the Week 3 post-mortem (§7.3) says every metric must have.

    python scripts/probe_critic_quality.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

load_dotenv(ROOT / ".env")

from app.config import PROVIDERS  # noqa: E402

OUT = ROOT / "scripts" / "probe_results.json"

EVIDENCE = """
E1 | "LangGraph exposes an explicit StateGraph with typed channels." | src: langgraph-docs
E2 | "CrewAI organises work as role-playing agents with a task list." | src: crewai-docs
E3 | "LangGraph supports interrupt() for human-in-the-loop pauses."  | src: langgraph-docs
E4 | "In one 2024 benchmark on a single support-ticket task, CrewAI completed in 41s." | src: blog-post
E5 | "The OpenAI Agents SDK was released in March 2025."             | src: openai-blog
"""

ANALYSIS = """
CONCLUSIONS

1. LangGraph is the most production-ready option available today and is used by the majority
   of enterprise teams running agents at scale.

2. LangGraph provides explicit state management through a typed StateGraph [E1], and supports
   human-in-the-loop pauses via interrupt() [E3].

3. CrewAI is the fastest framework available, completing tasks in 41 seconds [E4].

4. CrewAI has the strongest documentation and community support of the three frameworks [E9].

5. The OpenAI Agents SDK is the newest option, released in March 2025 [E5]. Because it is newest,
   it is therefore the least mature and should be avoided by all teams.

6. LangGraph offers no mechanism for pausing a workflow for human approval, which is a
   significant drawback for regulated environments.

RECOMMENDATION
Adopt LangGraph. It offers the best total cost of ownership and the lowest operational
support burden of the three.
"""

DEFECTS = {
    "D1_unsupported": ["1", "majority", "enterprise", "most production-ready", "unsupported",
                       "no evidence", "no citation"],
    "D2_fabricated": ["e9", "does not exist", "nonexistent", "non-existent", "fabricat",
                      "not in the evidence", "no such"],
    "D3_contradiction": ["contradict", "interrupt", "e3", "conflict", "inconsist",
                         "both", "2 and 6", "6 and 2"],
    "D4_overgeneral": ["e4", "single", "one benchmark", "one task", "generali",
                       "fastest", "sample"],
    "D5_irrelevant": ["e5", "newest", "least mature", "does not follow", "non sequitur",
                      "not support", "conclusion 5", "5"],
    "D6_missing": ["total cost", "ownership", "tco", "support burden", "recommendation",
                   "not researched", "no evidence", "criteri"],
}


class Problem(BaseModel):
    location: str = Field(description="Which conclusion or section the problem is in")
    issue: str = Field(description="What is wrong with it")
    severity: str = Field(description="major or minor")


class Review(BaseModel):
    approved: bool = Field(description="True only if the analysis is sound enough to publish")
    problems: list[Problem] = Field(description="Every problem found")
    missing_evidence: list[str] = Field(default_factory=list)


SYSTEM = """You are the Critic in a multi-agent research system. Review the analysis against
the evidence list. Evaluate for: unsupported claims, citations to evidence that does not exist,
internal contradictions, overgeneralisation from thin data, citations that do not support the
claim they are attached to, and conclusions resting on criteria that were never researched.

Do NOT rewrite the analysis. Report problems only. Be specific about which conclusion is wrong
and why. Approve only if you find no major problems."""


def detected(review: Review) -> dict[str, bool]:
    """A defect counts as found if the Critic's prose mentions >=2 of its marker terms."""
    blob = " ".join(
        [p.location + " " + p.issue for p in review.problems] + review.missing_evidence
    ).lower()
    return {d: sum(m in blob for m in markers) >= 2 for d, markers in DEFECTS.items()}


def run(make, model: str) -> dict:
    t0 = time.perf_counter()
    try:
        review = make(model).with_structured_output(Review, method="function_calling").invoke(
            [("system", SYSTEM), ("user", f"EVIDENCE:\n{EVIDENCE}\n\nANALYSIS:\n{ANALYSIS}")]
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:120], "found": 0,
                "seconds": round(time.perf_counter() - t0, 2)}

    hits = detected(review)
    return {
        "ok": True,
        "seconds": round(time.perf_counter() - t0, 2),
        "approved": review.approved,          # approving this analysis is itself a failure
        "n_problems": len(review.problems),
        "hits": hits,
        "found": sum(hits.values()),
        "rejected_correctly": not review.approved,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="groq")
    args = ap.parse_args()
    cfg = PROVIDERS[args.provider]
    key = os.getenv(cfg.api_key_env)
    if not key:
        print(f"No {cfg.api_key_env} set."); return 1

    from langchain_openai import ChatOpenAI

    def make(model: str):
        return ChatOpenAI(model=model, api_key=key, base_url=cfg.base_url,
                          temperature=0, max_tokens=2048, timeout=90, max_retries=0)

    models = cfg.probes()
    runs = 2  # temperature=0 is not fully deterministic on hosted inference

    print(f"Six planted defects. {runs} runs per model.\n")
    print(f"{'model':<26} {'run':<5} {'rejected':<9} {'defects found':<40} {'time':>6}")
    print("-" * 92)

    results = []
    for model in models:
        for i in range(runs):
            r = run(make, model, )
            if not r["ok"]:
                print(f"{model:<26} {i+1:<5} {'FAILED':<9} {r['error'][:40]:<40} {r['seconds']:>5.2f}s")
                results.append({"model": model, "run": i + 1, **r})
                continue
            flags = " ".join(
                f"{d.split('_')[0]}{'+' if hit else '-'}" for d, hit in r["hits"].items()
            )
            print(f"{model:<26} {i+1:<5} {'yes' if r['rejected_correctly'] else 'NO ':<9} "
                  f"{flags} {r['found']}/6{'':<6} {r['seconds']:>5.2f}s")
            results.append({"model": model, "run": i + 1, **r})

    print("\nAverages:")
    for model in models:
        rows = [r for r in results if r["model"] == model and r["ok"]]
        if not rows:
            print(f"  {model:<26} all runs failed")
            continue
        avg = sum(r["found"] for r in rows) / len(rows)
        rej = sum(r["rejected_correctly"] for r in rows)
        sec = sum(r["seconds"] for r in rows) / len(rows)
        print(f"  {model:<26} {avg:.1f}/6 defects   rejected {rej}/{len(rows)}   {sec:.2f}s avg")

    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[f"{args.provider}_critic_quality"] = results
    OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
