---
name: clean-code-analyzer
description: >-
  Deep-dives into the codebase to explain and evaluate adherence to architectural rules (modularity, dependency injection, DRY, strict folder organization, and 'why' documentation).
---

# Clean Code Analyzer

## Overview
This skill performs an in-depth analysis of the codebase or specific components, evaluating them against the project's core architectural and organizational guidelines.

## Workflow

### 1. Architectural Review
Examine the codebase for adherence to these principles:
- **Modularity:** Are components isolated, reusable, and independently testable?
- **Dependency Injection & Encapsulation:** Is coupling minimized through dependency injection?
- **DRY (Don't Repeat Yourself):** Is there a single authoritative source for data and constants? Are duplicate code blocks refactored into general functions?
- **Testing and CI:** Are there unit tests, regression tests for fixed bugs, and minimal "smoke tests"? Are assertions used to catch bad inputs?

### 2. Organizational Review
Ensure the project structure follows strict guidelines:
- **Folders:** Is the project divided into `data` (raw datasets), `results` (generated files), `src` (source code), and `doc` (documentation)?
- **Data Integrity:** Are raw datasets treated as immutable?
- **Naming:** Are data files named descriptively without ambiguous sequential numbering (e.g., `YYYY-MM-name.csv`)?
- **Format:** Do tabular files follow "tidy data" principles (columns=variables, rows=observations)?

### 3. Documentation Review
Verify that documentation aligns with the project's philosophy:
- **Focus on the "Why":** Does the documentation explain the abstract intent and reasoning, rather than the "how" (which should be evident from the code)?
- **Structure:** Are there standard files like `README`, `CONTRIBUTING`, `CITATION`, and `LICENSE`?

### 4. Provide Detailed Report
Present a detailed, structured report explaining how well the codebase adheres to these rules and highlighting specific areas for improvement.
