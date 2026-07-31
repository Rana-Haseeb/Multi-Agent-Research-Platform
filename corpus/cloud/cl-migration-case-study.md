---
doc_id: cl-migration-case-study
title: Case Study: Migrating an AI SaaS from Heroku to Fly.io
domain: cloud
source_type: practitioner_blog
publisher: engineering blog
published: 2026-01-17
covers: [Fly.io, Render]
reliability: medium
synthetic: true
---

# Migrating an AI SaaS from Heroku to Fly.io

Our product serves about 30,000 inference requests a month. Heroku costs had risen to roughly
180 USD monthly and cold starts on hobby dynos were hurting demos.

## What we moved to

We evaluated Render and Fly.io. Render was simpler but had no scale-to-zero on paid plans, and
our traffic is spiky - roughly 80 percent of requests arrive in a six-hour window.

Fly.io's scale-to-zero with sub-second wake fit that shape. Monthly cost dropped to 44 USD.

## What went wrong

Region-pinned volumes were the surprise. We had assumed we could run app instances in three
regions against one database; in practice the volume lives in one region and cross-region
latency ate the benefit. We now run single-region with a read replica.

## Would we do it again

Yes, but we would model the database topology before the compute topology. The compute migration
took two days; the database rethink took three weeks.
