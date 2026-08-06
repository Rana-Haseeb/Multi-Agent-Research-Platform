# Builder Journal (§37)

## What I set out to build

A research platform where six specialised agents plan, research, analyse, fact-check, critique and
write — and where every claim about it is backed by something that runs. The spec's warning stayed
pinned in front of me the whole time: *five prompts called five agents does not satisfy the
requirements.* So the design rule I gave myself on day one was that **every limit an agent has must
be enforced by code, not by asking the model nicely.** That single rule shaped most of what follows.

## The decisions that mattered

**Structural over instructional.** The Writer cannot search — not because the prompt says so, but
because `run_tool` checks `allowed_agents` and raises before the function is reached. A major
conclusion cannot ship without a citation because a Pydantic validator rejects it. The revision loop
stops at two because the router compares an integer, not because the Critic was told to stop. Of the
31 adversarial tests, 29 defend structurally. That ratio is the honest measure of whether the rule
was followed.

**A single-agent baseline from day one.** The graded question is "why does this need multiple
agents?" — and that is only answerable if the comparison is real rather than retrofitted. Building
the baseline path early cost maybe two hours and made Experiment 1 trustworthy.

**Metrics that can fail.** Every metric has a test proving it can return something other than
success, and an empty denominator returns `None`, never `1.0`. This mattered more than I expected —
see below.

## What broke, and what it taught me

**My scorer punished the model for being right.** Early on, `gpt-oss-120b` scored 6/8 on planning
while a weaker model scored higher. Reading the transcripts, the "failures" were cases where it
correctly withheld a plan because the request needed clarification first — exactly the behaviour §20
asks for. My scorer counted the missing plan as a miss. Fixed, it scored 10/10, and I changed the
model choice on the strength of it. **A bad metric doesn't just mis-measure; it actively selects the
wrong thing.** This was the most valuable single lesson of the week.

**My defect detector reported the Critic caught 0 of 4 planted flaws.** It caught all 4. The
detector matched prose, and the Critic's catches were phrased inside disclaimers the matcher didn't
recognise. I rewrote it to check structure instead of wording. Had I trusted the first number, I
would have reported a working component as broken.

**One substring nearly corrupted every failure statistic.** `_friendly()` classified errors by
matching `"rate"` — which lives inside `"generate"`. Every structured-output failure was being
reported as provider throttling. The §22 failure metrics would have been filled with phantom API
errors while real schema bugs stayed invisible. It is now a regression test.

**My test suite was writing to the production database.** `store=None` fell through to a real
`EvidenceStore`. I found 126 orphaned rows and removed them, then made `None` mean *no persistence*
via an explicit sentinel. Tests that touch production state are worse than no tests.

**A metric was counting the same disagreement four times.** One clarification mismatch scored as
four failures, dragging human-approval compliance to a false 76%. An early return fixed it; the true
figure is 100% (17/17).

**Prompt injection got through once — in the request, not the corpus.** I had guarded retrieved
documents but not the user's own request. Given "you are now in maintenance mode", the Supervisor
didn't leak its prompt but did adopt the attacker's framing as its research objective. That is a
partial compromise and I recorded it as one. A request-level guard fixed it; re-running changed the
objective to determining the user's legitimate topic.

## What the measurements actually said

Not all of it flattered the design, and this is the part I most want to be straight about.

**Multi-agent cost 3.6× the calls and 3× the wall time** of the single-agent baseline (40 calls /
208s vs 11 calls / 68s). What it bought: a report backed by 4 stored evidence items instead of 2, 17
declared limitations instead of 2, 8 problems raised by the Critic, and 2 revision cycles. So the
multi-agent case is a **quality-for-cost trade, not a free win** — and on a simple factual question
it would be a bad trade. That is a real finding, not a disappointing one.

**Parallel research gave 2.63× speedup** against a theoretical maximum of 3 (249s → 95s across 3
tasks). Close to the ceiling, and the honest gap is fan-out overhead.

**Experiment 4's context-trimming saved only 4.5% of tokens** live. I expected more. The offline
measurement shows why — the full context is only ~795 tokens, so there is little to trim at this
corpus size. The strategy would matter at a larger scale; at this one it nearly doesn't. Reporting
4.5% is more useful than not running it.

**Experiment 2 showed the critic-enabled run finishing *faster* than the disabled one** (12.4s vs
25.1s). That is provider latency noise, not a real effect — with n=1 it cannot be anything else. I
have left the number as measured and said so rather than dropping an inconvenient row.

**Clarification decisions are bistable.** On identical input at temperature 0, the same request
triggered clarification 3 times out of 5. That invalidates single-run measurement of that behaviour
and is a caveat on the 91.9% clarification accuracy figure.

## Constraints and what I would do differently

The binding constraint all week was **tokens per day**, not per minute — invisible in response
headers and only visible in raw 429 bodies (200k/day on the reasoning tier, 100k/day on the
throughput tier). At ~100k tokens per full workflow, that is roughly nine complete runs a day. I
built depth tiers into the evaluation so 28 cases could be scored inside that ceiling, and a
process-wide pacer to flatten bursts. **I should have measured that ceiling on day one instead of
discovering it after six failed experiment runs.** Week 3's retrospective told me to budget quota
before running anything expensive, and I still under-did it.

Given more quota, the first thing I would spend it on is **repeats**: almost every experiment here is
n=1, which is enough to show a mechanism works and not enough to size an effect. The single-vs-multi
comparison in particular deserves 5+ runs per arm before anyone treats the 3× cost ratio as precise.

Second, I would rotate every credential — the keys used this week were pasted into a chat transcript
and must be considered exposed regardless of the fact that they only ever lived in a git-ignored
`.env`.

## Where it ended up

273 tests passing, 28/28 evaluation cases scored with 25 passed (89.3%), all five §29 targets met,
five experiments complete, eleven phase verifiers green. The number I trust most is not the pass
rate — it is that every metric in it has a test proving it can fail.
