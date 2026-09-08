# Chapter 2 Split and Chapter Renumbering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Split the current Chapter 2 into a stochastic-optimization survey and a KDE foundations chapter, then renumber all later chapters and references consistently.

**Architecture:** Preserve the current section bodies while creating explicit chapter introductions and moving the existing chapter source/figure directories so filesystem names match printed chapter numbers. Apply semantic replacements to chapter references and local figure paths, preserving external theorem numbering and stable equation/figure labels.

**Tech Stack:** LaTeX, git mv, Python/rg source scans, latexmk, Ghostscript visual inspection.

---

### Task 1: Split Chapter 2 content

**Files:**
- Modify: `kap02/kapitel02.tex`
- Create: `kap03/kapitel03.tex`
- Move: current KDE sections and figures into the new `kap03` source/figure directory.

- [x] Leave Sections 2.1--2.3 and their figures in Chapter 2 under the survey title, with a brief survey-specific introduction.
- [x] Move current KDE sections (currently 2.4--2.5) into Chapter 3 under `Chance Constraints and Kernel Density Estimation`, with a new introduction that states the chapter scope and transition to the residual-KDE method.
- [x] Renumber the moved sections as 3.1 and 3.2 and preserve all equation, figure, and subsection labels unless they encode the old chapter number.
- [x] Update chapter-local figure paths and `\graphicspath` declarations.

### Task 2: Renumber chapter files and inputs

**Files:**
- Move: `kap03`→`kap04`, `kap04`→`kap05`, …, `kap10`→`kap11`.
- Modify: `Masterarbeit.tex`.

- [x] Use reversible `git mv` operations and preserve all chapter figure assets.
- [x] Update every `\input` from `kap01`--`kap10` to `kap01`--`kap11`, inserting the new Chapter 3.
- [x] Update any root-level comments or metadata that identify the chapter count.

### Task 3: Update cross-references and local paths

**Files:**
- Modify: all submission-facing `kap01`--`kap11` `.tex` files and relevant source figure captions/scripts.

- [x] Replace printed references to the old chapters according to the mapping: old Ch. 3→4, 4→5, 5→6, 6→7, 7→8, 8→9, 9→10, 10→11.
- [x] Update paths such as `kap03/figures`, `kap04/figures`, etc. to match moved directories.
- [x] Preserve references to external source theorem numbers (e.g., Schuster Theorems 3.1–3.3) and equation/figure labels that do not encode chapter numbers.
- [x] Update the introduction, discussion, conclusion, and captions so their chapter map agrees with the new 11-chapter structure.

### Task 4: Verify structure and build

**Files:**
- Verify: all moved sources, `Masterarbeit.tex`, bibliography and figure assets.

- [x] Scan section headings, `\input` paths, chapter references, labels, and figure includes for stale references.
- [x] Run `latexmk -pdf -interaction=nonstopmode -halt-on-error` in a fresh temporary output directory.
- [x] Scan the log for undefined references, duplicate labels, citation warnings, and missing figures.
- [x] Render representative pages for Chapters 2, 3, 4, 10, and 11 and inspect headings, captions, and page flow.
- [x] Review the final diff and commit only the approved split/renumbering changes and planning documents.
