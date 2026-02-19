---
name: dev-docs
description: Load project development documentation for a specific topic into context. Use when you need to understand architecture, APIs, database patterns, eval framework, testing, or any other dev topic.
argument-hint: "[topic]"
allowed-tools: Read, Glob, Grep
---

Load development documentation for topic: $ARGUMENTS

## Instructions

1. Read the index file at `docs/dev/INDEX.md`
2. If `$ARGUMENTS` is empty or "list", show the user the available topics from the index and stop
3. Otherwise, match the topic against the **Topic** column keywords in the index. Pick the best matching file(s) — usually 1, sometimes 2 if the topic spans areas
4. Read the matched file(s) using the Read tool
5. Present the content to the user. Do NOT summarize — the files are already concise. Just confirm what was loaded so the user knows it's in context
6. If no match is found, list the available topics and ask the user to pick one
