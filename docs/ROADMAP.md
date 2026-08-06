# Future Roadmap

Ordered by what the measurements actually justify, not by what would be most fun to build. Each item
names the evidence that motivates it, so nothing here is speculative wishlisting.

## Near term — things the current results say are wrong or thin

### 1. Repeats on every experiment (highest value)

**Evidence:** almost every experiment is n=1. Experiment 2 measured the critic-enabled run finishing
*faster* than the disabled one (12.4s vs 25.1s), which is provider latency noise that a single run
cannot separate from signal. The single-vs-multi cost ratio (3.6× calls, 3× wall time) shows a
mechanism but does not size an effect precisely.

**Work:** 5+ runs per arm, report median and spread, and mark any difference inside the noise band as
inconclusive rather than as a result. This needs quota, not code — the runner already supports
repeats.

### 2. Self-consistency on the clarification decision

**Evidence:** at temperature 0, the same request triggered clarification 3 times out of 5. The
decision is bistable, which puts a caveat on the 91.9% clarification accuracy figure.

**Work:** sample the decision three times and take the majority, or move the decision to a structural
pre-check (does the request name the alternatives, the constraints and the decision criteria?) with
the model only consulted on genuine ambiguity. The second option is preferable — it converts a
probabilistic behaviour into a deterministic one, which is the same move that made the other
defences reliable.

### 3. Persistent checkpointer

**Evidence:** `MemorySaver` is per-process, so a restart drops any run paused at a human checkpoint.
On Streamlit Cloud that means an interrupted approval is lost on every redeploy.

**Work:** swap in a Postgres-backed checkpointer. The `WorkflowSession` abstraction already isolates
this, so the change is confined to construction.

## Medium term — capability the design anticipates but does not yet use

### 4. Live web search alongside the corpus

The corpus is deliberate: it makes evaluation deterministic and lets defects be planted and detected.
But it also means the system cannot answer anything current. `ENABLE_LIVE_SEARCH` and a Tavily key
are already wired; what is missing is a **reliability model for live sources**, since the corpus's
`high`/`medium`/`low` labels are hand-assigned and a live result has none. Without that, the
confidence-capping defence (a low-reliability source cannot yield high confidence) silently stops
applying — so this must not ship until source scoring exists.

### 5. Context strategy at a scale where it matters

**Evidence:** Experiment 4's trimming saved 4.5% of tokens live, because the full context is only
~795 tokens. The strategy is sound and the corpus is too small for it to pay.

**Work:** re-run at 10× corpus size before concluding anything about it. The interesting question is
where the crossover is, and that is a measurement, not an implementation.

### 6. Hybrid retrieval

BM25 is the default because it needs no key and no `torch`, and it was sufficient here. `RETRIEVAL_MODE`
already accepts `vector` and `hybrid`. Worth revisiting only alongside item 5 — on a small corpus,
lexical matching is hard to beat and the added dependency is real.

## Longer term

### 7. Multi-turn research sessions

Today each run is independent. Letting a user refine a finished report ("focus the comparison on
operational cost") would reuse stored evidence rather than re-researching, which is exactly what the
evidence store makes cheap. The blocker is state design, not model capability.

### 8. Closing the remaining instructional gaps

29 of 31 adversarial defences are structural; the injection guards are the notable instructional
ones, and prompt wording can always be argued with. The containment story is already structural — an
injected instruction cannot grant a tool — but detection is not. A classifier over retrieved chunks
that flags imperative text addressed to the reader would turn a probabilistic defence into a
measurable one.

### 9. Cost model on a paid tier

Every cost figure here is in tokens and calls, with `estimated_cost_usd` at 0.0 because the free tier
bills nothing. That makes the trade-off in Experiment 1 hard to state in the terms a real user cares
about. Running one arm on a paid tier would let the quality-for-cost trade be priced rather than
merely counted.

## Explicitly not planned

- **More agents.** Six is what the workflow needs. Adding a seventh to look more sophisticated is the
  exact failure mode the spec warns about.
- **Removing the human checkpoints.** They cost latency and are the point.
- **Raising `MAX_REVISION_CYCLES` above 2.** It would put the workflow outside the §18 requirement,
  and Experiment 5 shows the Critic still had unresolved objections at 2 — the answer to that is
  better analysis, not more loops.
