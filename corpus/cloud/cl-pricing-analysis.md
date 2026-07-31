---
doc_id: cl-pricing-analysis
title: Total Cost Analysis for an AI SaaS at Three Scales
domain: cloud
source_type: independent_benchmark
publisher: Cloud Economics Group
published: 2026-03-19
covers: [AWS, GCP, Azure, Render, Railway, Fly.io]
reliability: high
synthetic: true
---

# Total Cost Analysis for an AI SaaS

Modelled workload: 50,000 requests/month, average 1,200 input and 400 output tokens, one small
Postgres instance, 40 GB egress. Costs in USD per month, excluding model inference.

## Results

| Platform | Compute | Database | Egress | Total |
|----------|---------|----------|--------|-------|
| AWS ECS  | 62      | 31       | 3.60   | 96.60 |
| GCP Cloud Run | 28 | 27       | 4.80   | 59.80 |
| Azure Container Apps | 34 | 29 | 3.48 | 66.48 |
| Render   | 25      | 20       | 0      | 45.00 |
| Railway  | 21      | 18       | 0      | 39.00 |
| Fly.io   | 19      | 22       | 2.00   | 43.00 |

## Observations

At this scale the platform-as-a-service options are 30 to 55 percent cheaper than the major
clouds. The ordering reverses above roughly 500,000 requests per month, where committed-use
discounts on the major clouds begin to dominate.

Model inference cost was excluded because it is provider-independent for teams calling an
external API, and it typically exceeds infrastructure cost by 3 to 10 times at this scale.

## Limitations

Single workload shape. No reserved-instance or committed-use pricing was applied.
