---
name: clean-code-fixer
description: >-
  Actively refactors code to enforce guidelines. It fixes simple issues automatically but flags and explains complex violations (e.g., massive coupled functions) for manual review.
---

# Clean Code Fixer

## Overview
This skill actively applies refactoring to enforce the project's clean code guidelines. It categorizes issues as either trivial (safe to auto-fix) or complex (requires user review).

## Workflow

### 1. Identify Target Issues
Determine the violations that need to be addressed (either passed from the `clean-code-post-hook` or discovered independently).

### 2. Trivial Fixes (Automatic)
Automatically refactor and fix trivial issues without asking for permission:
- Remove leftover debug `print` statements (replace with `assert` where applicable).
- Rename poorly named variables to be descriptive.
- Delete commented-out code blocks used for logic toggling (replace with `if/else` flags if necessary).
- Enforce basic formatting rules.

### 3. Complex Fixes (Flag for Review)
For violations that carry high risk during refactoring, **do not** attempt an automated fix. Instead, flag the issue and provide a highly detailed explanation of the problem. Complex issues include:
- Functions significantly exceeding 60 lines.
- Functions with more than 6 parameters.
- Heavy coupling between modules.
- Missing dependency injection.
- Lack of test coverage or absent assertions for critical inputs.

### 4. Present Actions Taken
Output a summary of the trivial issues that were automatically fixed, and list out the detailed explanations for any complex issues that were skipped and flagged for manual review.
