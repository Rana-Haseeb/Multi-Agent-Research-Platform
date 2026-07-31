---
doc_id: ca-cursor
title: Cursor - AI-Native Editor
domain: coding_assistants
source_type: vendor_docs
publisher: Cursor
published: 2026-02-10
covers: [Cursor]
reliability: high
synthetic: true
---

# Cursor

Cursor is a fork of VS Code with AI features integrated into the editing surface. Tab completion
predicts multi-line edits, and an agent mode performs multi-file changes.

## Codebase awareness

Cursor indexes the repository to provide retrieval over project files, so questions can reference
code the model has not been shown directly.

## Model choice

Users select among several frontier models per request. Usage is metered against a monthly
request allowance, with additional usage billed separately.

## Trade-offs

Because Cursor is an editor fork, teams standardised on another IDE must switch editors to adopt
it. Extension compatibility is good but occasionally lags upstream VS Code releases.
