# Chapters 1–2 Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore evidence-backed explanatory and literature context in Chapters 1–2, align figures/captions, and verify the compiled thesis.

**Architecture:** Chapter 1 will be expanded around problem motivation, correctly ordered difficulty mechanisms, residual identity, scope, and four RQs. Chapter 2 will be expanded around classical taxonomy, cited approximation methods, KDE foundations, dimensionality, and literature gap. Existing Chapter 3–10 content remains unchanged except for downstream RQ labels needed for consistency.

**Tech Stack:** LaTeX source, BibTeX, repository figure generators/assets, latexmk, Python source-level checks.

---

### Task 1: Restore Chapter 1 context and notation

**Files:**
- Modify: `kap01/kapitel01.tex`
- Reference: `src/figure_generators/kap01/make_figs.py`

- [ ] Add a compact domain-motivation paragraph with citations and synthetic-evidence qualification.
- [ ] Define notation and correct the three-difficulty order, including moving-region integral and empirical-indicator equations.
- [ ] Add parametric misspecification, residual CDF identity, residual-space benefits, explicit scope boundaries, and four numbered RQs.
- [ ] Correct dispatch provenance wording, epsilon-to-support qualification, and editorial prose.
- [ ] Correct the mean/median annotation in `src/figure_generators/kap01/make_figs.py` and regenerate only if the local environment supports it.

### Task 2: Restore Chapter 2 survey and KDE foundations

**Files:**
- Modify: `kap02/kapitel02.tex`
- Modify: `bibliography.bib` only if a required DOI or missing key is confirmed
- Reference: `src/figure_generators/kap02/make_figs.py`

- [ ] Replace the taxonomy text so it matches `fig_taxonomy.pdf` and includes stochastic programming.
- [ ] Add cited, separate compact treatments of Gaussian, scenario, SAA, CVaR, quantile smoothing, and black-box methods.
- [ ] Add KDE assumptions, consistency, kernel choices, AMISE notation, bandwidth caveat, and multivariate rate.
- [ ] Add literature-map paragraphs and the conditional thesis gap.
- [ ] Align all retained captions with generated figures; do not restore the unsafe `fig_kernels` claim.

### Task 3: Align downstream research-question terminology

**Files:**
- Modify: `kap04/kapitel04.tex`
- Modify: `kap06/kapitel06.tex`
- Modify: `kap08/kapitel08.tex`

- [ ] Use the restored four-question numbering consistently, or remove stale RQ4 references if the final Chapter 1 uses three questions.
- [ ] Keep the local shifted-Epanechnikov and Keil Split–Bernstein distinction explicit.

### Task 4: Verify source and build

**Files:**
- Inspect: `Masterarbeit.tex`, all chapter files, bibliography, figure assets

- [ ] Run source scans for duplicate labels, undefined cross-reference targets, undefined citations, and stale RQ references.
- [ ] Run `python3 -m compileall -q src`.
- [ ] Run `latexmk -pdf -interaction=nonstopmode -halt-on-error` in a fresh temporary output directory.
- [ ] Inspect generated PDF metadata/page count and report any remaining warnings or caption/content mismatch.
