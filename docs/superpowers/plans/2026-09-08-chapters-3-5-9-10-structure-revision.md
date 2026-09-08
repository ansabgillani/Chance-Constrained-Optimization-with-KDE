# Chapters 3--5 and 9--10 Structure Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved structural, evidential, and limitation revisions in the thesis while preserving traceable numerical claims.

**Architecture:** Edit the chapter LaTeX sources in bounded chapter tasks, then correct data provenance and Chapter 8 table values. Reuse existing figures where valid, relabel or regenerate only when captions do not match their generators, and add diagrams only when they communicate an existing equation or audit gate.

**Tech Stack:** LaTeX/pdflatex or latexmk, Python figure generators, Markdown planning documents, git.

---

### Task 1: Reorganize and correct Chapters 3--5

**Files:**
- Modify: `kap03/kapitel03.tex`
- Modify: `kap04/kapitel04.tex`
- Modify: `kap05/kapitel05.tex`
- Modify: `src/figure_generators/kap03/make_figs.py`
- Modify: `src/figure_generators/kap04/make_figs.py`
- Modify: `src/figure_generators/kap05/make_figs.py`

- [ ] Add the approved subsubsection hierarchy, preserving equation labels and cross-references.
- [ ] Qualify smoothness assumptions, pointwise consistency, fixed-decision biased-KDE expectation bounds, and optimizer-selection limits.
- [ ] State the missing geometric/projection assumptions for the convergence proof architecture and correct the uniform-convergence quantifier.
- [ ] Correct false dispatch provenance in Chapters 3--5 figure captions by labelling standalone lognormal diagnostics or regenerating from dispatch data; correct stale theorem numbering in `fig_chain`.
- [ ] Remove rhetorical overstatements and repair grammar without changing verified results.
- [ ] Run a label/reference scan and the chapter figure generators; inspect generated figures before proceeding.

### Task 2: Rebuild Chapter 9 discussion

**Files:**
- Modify: `kap09/kapitel09.tex`
- Optionally create/modify: `src/figure_generators/kap09/make_figs.py` and `kap09/figures/*` for approved diagnostic diagrams.

- [ ] Replace the experiment-recap opening with explicit RQ1--RQ4 evidence/claim/boundary subsubsections.
- [ ] Split the literature comparison into Schuster and Keil replication boundaries.
- [ ] Replace duplicated `Threats to validity and repair plan` content with a four-part mechanistic synthesis.
- [ ] Make method-selection rules explicitly conditional on the stated theory and evidence assumptions.
- [ ] Preserve future work in Chapter 9, but mark proposed post-selection equations and margin decompositions as conjectural targets with defined symbols or replace unsupported probability-margin algebra with prose.
- [ ] Add only diagrams whose data sources are existing ledgers/equations and whose captions state diagnostic scope.

### Task 3: Consolidate Chapter 10 limitations

**Files:**
- Modify: `kap10/kapitel10.tex`

- [ ] Replace the combined limitations/future-work subsection with two limitation subsubsections: statistical certification/convergence and model/joint-event/numerical transfer.
- [ ] Add the qualified empirical-envelope equation using existing notation.
- [ ] Correct the lunar quadrature wording: repairing the endpoint artifact does not create physical fuel variation in the fixed-horizon invariant model.
- [ ] Remove the repeated four-item future-work roadmap and retain a single cross-reference to Chapter 9.
- [ ] Verify the conclusion still answers the research questions without introducing unsupported rankings.

### Task 4: Correct Chapter 8 evidence table and cross-chapter provenance

**Files:**
- Modify: `kap08/kapitel08.tex`
- Modify: `docs/research/paper-verification-report-2026-09-06.md` only if its statements become stale.

- [ ] Replace duplicated dispatch training values with the corrected ledger values: nominal `0.588`, scenario `0.004`, unbiased KDE `0.052`, local shifted Epanechnikov `0.012`; retain test values `0.57875`, `0.006`, `0.03975`, `0.011`.
- [ ] Add a short note that static/dispatch paper-aligned comparisons change estimator, bandwidth, and continuation branch together.
- [ ] Check all Chapter 9 and Chapter 10 references against the corrected table.

### Task 5: Build and verify the revised manuscript

**Files:**
- Verify: `Masterarbeit.tex` and all included chapter sources.

- [ ] Run `latexmk -pdf -interaction=nonstopmode -halt-on-error` in a fresh temporary output directory.
- [ ] Scan the log for undefined references, duplicate labels, citation warnings, overfull boxes in edited chapters, and figure-file failures.
- [ ] Run available Python source checks and figure generators without overwriting user artifacts.
- [ ] Review the final diff to ensure only approved files changed and no generated PDF was deleted or overwritten.
- [ ] Commit the manuscript revision separately from the planning documents.
