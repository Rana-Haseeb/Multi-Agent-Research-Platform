---
doc_id: fw-openai-agents-sdk
title: OpenAI Agents SDK - Handoffs, Guardrails and Sessions
domain: frameworks
source_type: vendor_docs
publisher: OpenAI
published: 2026-03-05
covers: [OpenAI Agents SDK]
reliability: high
synthetic: true
---

# OpenAI Agents SDK

The Agents SDK is a lightweight framework built on three primitives: agents (an LLM with
instructions and tools), handoffs (one agent transferring control to another), and guardrails
(validation running alongside agent execution).

## Handoffs

A handoff is exposed to the model as a tool call. When the model selects it, control transfers to
the target agent, which receives the conversation history. Handoffs are therefore chosen by the
model rather than by explicit routing logic.

## Guardrails

Input and output guardrails run concurrently with the agent and can halt execution when
validation fails. They are typically used for relevance checks and safety filtering.

## Sessions and tracing

Sessions maintain conversation history across runs. Built-in tracing records agent runs, tool
calls and handoffs, and integrates with the OpenAI dashboard.

## Trade-offs

The SDK is deliberately minimal, keeping the API small but leaving orchestration patterns such as
conditional loops and parallel fan-out to the developer. Tracing is tied to OpenAI's platform.
Because handoff decisions are model-driven, routing is harder to constrain than in a graph where
edges are declared explicitly.
