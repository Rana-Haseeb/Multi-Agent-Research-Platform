---
doc_id: fw-crewai-docs
title: CrewAI Documentation - Crews, Agents and Tasks
domain: frameworks
source_type: vendor_docs
publisher: CrewAI
published: 2026-01-22
covers: [CrewAI]
reliability: high
synthetic: true
---

# CrewAI: Role-Based Agent Orchestration

CrewAI organises work around agents with a role, goal and backstory, and tasks assigned to those
agents. A crew binds agents and tasks together and executes them either sequentially or through
a hierarchical process in which a manager agent delegates.

## Process models

The sequential process runs tasks in declaration order, passing each output forward as context.
The hierarchical process introduces a manager LLM that decides delegation and ordering at runtime.

## Human input

CrewAI supports a human_input flag on individual tasks. When set, execution pauses and the
operator is prompted to review the task output before the crew continues. This provides built-in
human-in-the-loop review without additional code.

## Tools

Tools are attached per agent. An agent may only call tools in its own list, which provides a
straightforward permission boundary between roles.

## Trade-offs

The abstraction is quick to start with; a working crew is roughly thirty lines. State between
tasks is passed as text context rather than a typed structure, so workflows needing conditional
routing or shared structured state can become difficult to express.
