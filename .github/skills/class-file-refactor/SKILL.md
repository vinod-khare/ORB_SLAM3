---
name: class-file-refactor
description: 'Use when renaming, reorganizing, or cleaning up a class and its files. Trace references, follow repository naming conventions, regroup members logically, update callers and build metadata, and validate the refactor.'
---

# Class and File Refactoring

Use this workflow for a class or similar type and its related source/header files. Make the smallest complete change, preserve behavior, and avoid unrelated cleanup.

## Procedure

1. **Establish scope and conventions**
   - Read applicable repository and language instructions before editing.
   - Inspect the target declaration, implementation, neighboring types, build files, and all references.
   - Check the worktree first. Preserve unrelated user changes and do not stage them.
   - Follow established repository naming conventions. In this repository, class names use `lower_camel_case`, methods use `lower_snake_case`, and private data members begin with `_` and do not end with `_`.

2. **Plan the complete refactor**
   - Identify the class declaration, constructor/destructor, definitions, includes, instantiations, qualified names, tests, documentation, and build-system entries that need updates.
   - Distinguish similarly named settings, runtime state, and derived values before moving or consolidating members.
   - Check whether a field or method is duplicated or unused by searching all references; remove it only when its lack of purpose is established.

3. **Refactor consistently**
   - Rename the type, constructor, related file names, and all references together when requested.
   - Group data members by role (for example dependencies, metadata, lifecycle state, and command state). Sort clearly related members within each group when requested, without obscuring dependencies or changing initialization semantics.
   - Keep declaration and constructor-initializer order consistent.
   - Preserve behavior and public API semantics unless the requested refactor explicitly changes them.
   - Update associated build files and directly related documentation. Avoid compatibility aliases unless the repository needs them.

4. **Validate**
   - Search for stale identifiers and old paths.
   - Format touched files using the repository formatter and run whitespace/diff checks.
   - Run the smallest relevant test, build, or type-check. If unavailable, report the exact environmental blocker without claiming success.
   - Review the final diff to ensure it contains only the requested, coherent refactor and does not include unrelated worktree changes.

5. **Commit or push only when asked**
   - Before committing, stage only the intended files and inspect the staged diff.
   - Before pushing, verify the current branch and remote. Never include unrelated changes.
