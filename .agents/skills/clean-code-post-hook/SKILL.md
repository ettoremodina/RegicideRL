---
name: clean-code-post-hook
description: >-
  Runs after code modifications to verify surface-level rules (function length, parameters, commented-out code, debug prints, descriptive names) and automatically invokes the fixer for trivial issues.
---

# Clean Code Post-Hook

## Overview
This skill acts as a lightweight check to ensure code changes adhere to basic clean code guidelines, specifically looking for common mistakes and trivial anti-patterns.

## Dependencies
- clean-code-fixer

## Workflow

### 1. Perform Code Inspection
Review the recently modified code for the following surface-level rules:
- **Function Length:** Are functions limited to around 60 lines?
- **Parameters:** Do functions have 6 or fewer parameters?
- **Commented Logic:** Are there any blocks of code commented out instead of using `if/else` logic?
- **Naming:** Are variables and functions named descriptively?
- **Leftover Prints:** Are there leftover `print` statements used for debugging?

### 2. Output Quick Summary
Generate a brief summary highlighting any rules that were violated. Do not produce an extensive report.

### 3. Automatic Invocation of Fixer (If Applicable)
If the violations found are trivial and easily fixable (e.g., removing leftover debug `print` statements, simple renaming), automatically run the `clean-code-fixer` skill to resolve them.
If the violations are complex (e.g., a function is 150 lines long), flag them in the summary but do **not** run the fixer.
