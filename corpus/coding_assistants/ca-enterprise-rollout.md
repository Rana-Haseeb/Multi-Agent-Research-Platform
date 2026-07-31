---
doc_id: ca-enterprise-rollout
title: Rolling Out AI Coding Assistants Across 300 Engineers
domain: coding_assistants
source_type: practitioner_blog
publisher: engineering blog
published: 2026-02-24
covers: [GitHub Copilot, Cursor]
reliability: medium
synthetic: true
---

# Rolling Out AI Coding Assistants Across 300 Engineers

We ran a six-month rollout. Notes for anyone doing the same.

## Procurement was the long pole

Legal review of data handling took eleven weeks. Start there, not with the tool evaluation.

## Adoption was uneven

Enthusiastic adoption among engineers with under three years of experience; much slower among
senior engineers, several of whom reported that reviewing generated code cost more than writing
it themselves for the systems they knew well.

## What we measured

We could not demonstrate a statistically significant change in pull requests merged per engineer
per sprint. We did see a measurable reduction in time-to-first-commit for engineers new to a
service, from a median of 4.1 days to 2.3 days.

## What we would change

We would pick one tool rather than allowing two. Supporting both split our internal documentation
and made it impossible to attribute outcomes.
