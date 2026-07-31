---
doc_id: fw-benchmark-2026
title: Independent Agent Framework Benchmark, Q1 2026
domain: frameworks
source_type: independent_benchmark
publisher: Open Agent Benchmark Consortium
published: 2026-03-30
covers: [LangGraph, CrewAI, AutoGen, OpenAI Agents SDK]
reliability: high
synthetic: true
---

# Agent Framework Benchmark - Q1 2026

Methodology: 200 runs per framework on a fixed three-step research task, identical model,
identical tool set, single region. Latency measured end to end. Figures are medians.

## Results

| Framework          | p50 latency | p95 latency | Tokens/run | Task success |
|--------------------|-------------|-------------|------------|--------------|
| LangGraph          | 2.1 s       | 4.4 s       | 5,900      | 94%          |
| OpenAI Agents SDK  | 2.4 s       | 5.1 s       | 6,300      | 92%          |
| CrewAI             | 3.8 s       | 9.2 s       | 11,400     | 87%          |
| AutoGen            | 5.6 s       | 14.7 s      | 18,200     | 81%          |

## Observations

Token consumption differs more than latency. Frameworks that re-send full conversation history to
every participant used roughly two to three times the tokens of frameworks passing structured
state.

Task success was dominated by failure to produce valid structured output rather than by reasoning
quality. LangGraph's advantage narrowed to within noise once structured-output retries were
enabled for all frameworks.

## Limitations

A single task type was measured. These numbers should not be generalised to workloads with heavy
tool use or long-running conversations.
