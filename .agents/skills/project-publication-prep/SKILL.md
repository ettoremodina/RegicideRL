---
name: project-publication-prep
description: A standard process for transforming any completed software prototype or project into a clean, well-documented, publication-ready repository.
---

# Project Publication Prep Workflow

This skill defines a standard workflow to prepare a software project for public release or long-term maintainability. Follow these steps when asked to prepare a project for publication.

## 1. Project State Analysis
- Review the current state of the codebase.
- Identify unused files, scripts, or dead code.
- Assess the current directory structure against standard conventions.
- Identify missing documentation, logging mechanisms, and reporting tools.

## 2. Refinement & Ticket Breakdown
- Break down the work into independent, actionable tickets.
- Each ticket must have a clear scope, minimal dependencies, and a strict Definition of Done (DoD).
- Include for each ticket: 
  - Title
  - Objective
  - Context (Why it exists & problem it solves)
  - Tasks to perform
  - Definition of Done
  - Dependencies
  - Priority (High/Medium/Low)
  - Effort Estimate (Small/Medium/Large)

## 3. Prioritization
- Sequence tickets logically to minimize rework and unblock subsequent tasks.
- Typical priority sequence: 
  1. Code Refactoring & Cleanup
  2. Core Infrastructure (Logging, Metrics)
  3. Experimental Reporting
  4. Documentation & README

## 4. Code Quality & Refactoring
- Standardize code style and naming conventions.
- Extract duplicated code into shared utilities.
- Remove dead code and unused files.
- Reorganize folder structures professionally.

## 5. Testing, Logging & Reporting
- Implement a centralized logging system replacing ad-hoc print statements.
- Set up experimental reporting scripts and standardize outputs (e.g., metrics.json, plots).
- Prepare systems for saving, historicizing, and analyzing run/game data.

## 6. Documentation
- Ensure all modules, classes, and functions have standardized docstrings (e.g., Google or NumPy style).
- Setup automatic documentation generation if applicable.
- Write a comprehensive `README.md` covering: project description, setup/installation, dependencies, usage examples, and future work.

## 7. Repository Preparation
- Ensure `.gitignore` is comprehensive.
- Ensure all sensitive data or local paths are removed.
- Final review of the repository before marking it "publication-ready".
