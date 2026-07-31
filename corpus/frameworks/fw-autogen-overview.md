---
doc_id: fw-autogen-overview
title: AutoGen - Conversable Agents and Group Chat
domain: frameworks
source_type: vendor_docs
publisher: Microsoft Research
published: 2025-12-11
covers: [AutoGen]
reliability: medium
synthetic: true
---

# AutoGen: Multi-Agent Conversation

AutoGen frames multi-agent systems as conversations between conversable agents. Each agent sends
and receives messages, and a GroupChatManager selects which agent speaks next according to a
configurable speaker-selection policy.

## Speaker selection

Strategies include round-robin, manual selection, and an LLM-driven auto mode in which a manager
model chooses the next speaker from the agent roster.

## Code execution

AutoGen includes first-class code execution, either in a local subprocess or a Docker container.
A UserProxyAgent can execute code blocks produced by other agents and feed results back into the
conversation.

## Human-in-the-loop

UserProxyAgent supports three modes - ALWAYS, TERMINATE and NEVER - controlling how often a human
is asked to intervene.

## Trade-offs

The conversation metaphor is flexible but makes control flow implicit: the sequence of turns
emerges from the speaker-selection policy rather than being declared. Teams report that debugging
long group chats is difficult and that token usage grows quickly, because the shared conversation
is re-sent to each participant.
