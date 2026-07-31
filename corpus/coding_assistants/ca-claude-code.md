---
doc_id: ca-claude-code
title: Claude Code - Terminal Agent Overview
domain: coding_assistants
source_type: vendor_docs
publisher: Anthropic
published: 2026-03-02
covers: [Claude Code]
reliability: high
synthetic: true
---

# Claude Code

Claude Code is a command-line coding agent that operates directly on a local repository. It reads
and edits files, runs shell commands, and can drive git operations.

## Extensibility

Custom slash commands, hooks that intercept tool calls, and MCP server integration allow the
agent to be extended with project-specific tooling. Subagents run focused tasks in an isolated
context.

## Permissions

Tool use is gated by a permission system with configurable allow and deny lists. Hooks can block
a tool call before it executes.

## Interfaces

Available as a CLI, desktop application, web application, and IDE extensions for VS Code and
JetBrains.

## Trade-offs

Terminal-first design suits engineers comfortable in a shell and is less approachable for those
expecting an IDE-native experience. Because the agent runs commands on a real machine, teams
should review its permission configuration before enabling broad access.
