---
name: codebase-improvement
description: 'Use when asked to improve, clean up, refactor, modernize, or self-review a codebase. Prioritize lean interfaces, clear ownership, reduced coupling, and small verified changes over broad rewrites or validation-only fixes.'
argument-hint: 'Optional focus area, such as interfaces, ownership, or duplication'
---

# Lean Codebase Improvement

Improve the design incrementally. Prefer a few meaningful, low-risk refactors over a broad cleanup.

## Principles

- Keep code lean: preserve the smallest useful API; avoid wrappers, configuration, indirection, and new abstractions without a demonstrated need.
- Focus on design quality: interfaces, ownership and lifetimes, dependency direction, header coupling, cohesive responsibilities, and repeated logic.
- Do not default to input validation or error handling as the main source of improvements. Address them when they reveal a concrete correctness issue, not as a substitute for refactoring.
- Follow existing conventions and nearby patterns. Avoid repository-wide renames, formatting churn, or unrelated fixes.
- Preserve user changes. Never stage or commit unrelated dirty files.
- Keep a short task checklist while working; do not create planning documents. An inline `// TODO:` is appropriate for a specific, worthwhile follow-up that is too broad for the current change.

## Workflow

1. Read the repository guidance and inspect the worktree. Identify modified files and leave unrelated changes untouched.
2. Explore a bounded area end-to-end: public headers, implementations, callers, and relevant tests. Search for existing patterns before adding a new one.
3. Choose one concrete improvement with a small scope. State the design goal briefly before editing.
4. Make one cohesive change at a time. Prefer, where appropriate:
   - references for required, non-owning dependencies and smart pointers for ownership;
   - private members over exposed implementation state;
   - forward declarations in headers, with includes moved to implementation files;
   - a narrow interface when a high-level component depends directly on concrete subsystem classes;
   - a simpler call site or existing standard-library facility instead of a new helper.
5. Keep APIs minimal. Do not introduce an interface or abstraction for a single implementation unless it removes a real dependency or establishes a useful seam.
6. Add or update focused tests for changed behavior. Build and run the smallest relevant test set, then run the broader suite when the change crosses shared interfaces.
7. Inspect the final diff for behavior changes, unnecessary complexity, formatting, and accidental user-file inclusion.
8. Commit each independently verified improvement with a concise message when commits are requested or established as the user's preference. Never include unrelated worktree changes.
9. Repeat only while a specific, safely actionable improvement remains. Summarize completed changes, verification, commits, and any deferred follow-up.

## Avoid

- Treating every warning, TODO, or missing guard as a mandate for a rewrite.
- Introducing generic frameworks or speculative abstractions before the need is demonstrated.
- Combining unrelated cleanups in one patch.
- Claiming a broad review is exhaustive when only a bounded area was inspected.
