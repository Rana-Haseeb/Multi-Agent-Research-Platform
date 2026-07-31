---
doc_id: fw-practitioner-blog
title: Six Months Running Agent Frameworks in Production
domain: frameworks
source_type: practitioner_blog
publisher: personal blog
published: 2026-02-02
covers: [LangGraph, CrewAI, AutoGen]
reliability: medium
synthetic: true
---

# Six Months of Agent Frameworks in Production

We shipped three internal agent systems this year and rewrote one of them twice. Some notes.

## CrewAI

CrewAI got us to a demo fastest - a working crew in an afternoon. Problems started when the
product team asked for an approval step before anything was sent to a customer. CrewAI has no
native human-in-the-loop mechanism, so we wrapped the whole crew in our own queueing layer and
re-ran it after approval. That took two weeks and never felt clean.

## LangGraph

The learning curve was real; understanding reducers took most of a week. But once the graph was
declared, the approval gate was just a node and we stopped writing glue code. Our on-call load
dropped noticeably after the migration, though I cannot separate that from other changes shipped
in the same quarter.

## AutoGen

We used AutoGen for an internal code-review bot. Group chat worked well until conversations got
long, at which point costs became hard to predict.

## What I would tell you

Pick based on whether your workflow has branches. Linear pipelines are fine in anything. The
moment you need conditional routing, retries down different paths, or a human in the middle, the
graph-based tools stop being overhead and start being the reason things work.
