# Chapters 3--5 and 9--10 Structure Revision Design

## Goal

Reorganize Chapters 3--5 into a consistent construction-to-theory progression, make Sections 9.1--9.4 analytical rather than repetitive, and consolidate Chapter 10 around two limitation dimensions while preserving the evidence boundaries of the experiments.

## Scope

- Add subsubsection hierarchy and revise transitions in `kap03/kapitel03.tex`, `kap04/kapitel04.tex`, and `kap05/kapitel05.tex`.
- Correct safety, convergence, theorem-assumption, and provenance claims in those chapters.
- Rebuild Chapter 9 around explicit research-question answers, literature relation, mechanism synthesis, method-selection boundaries, and future work.
- Replace Chapter 10's duplicated limitations/future-work section with two limitation dimensions and a qualified empirical-envelope equation.
- Correct Chapter 8's dispatch training/test table values and any figure captions whose source does not match the generated data.
- Add only high-value explanatory diagrams if their data source and takeaway are explicit; retain existing figures where they already explain the mechanism.

## Design

Chapter 3 will move from residual-space identity through estimator construction and differentiation to statistical properties. Chapter 4 will move from the failure of symmetric smoothing through fixed-decision envelope scope, biased kernels, joint events, log-sum-exp smoothing, and continuation. Chapter 5 will state the formal problem pair and assumptions before separating the three convergence theorems and their applicability limits. Chapter 9 will answer RQ1--RQ4 in a claim/evidence/boundary matrix, then synthesize mechanisms and state conditional method-selection rules. Chapter 10 will retain only statistical-certification and model/numerical-transfer limitations; the technical roadmap remains in Chapter 9.

## Evidence rules

- Every numerical claim must match the corrected ledgers or a named figure generator.
- Pointwise biased-kernel domination is a fixed-decision expectation statement; it is not a finite-sample population certificate for a data-selected optimizer.
- Held-out frequencies are conditional measurements for an independently fitted decision; failed solver rows cannot support feasibility claims.
- Theorem statements and proof architectures must state the uniform-convergence, geometry, projection, and sharp-minimum assumptions actually needed.
- Figures generated from the standalone lognormal diagnostic must be labelled as such, or regenerated from the dispatch data.

## Visual additions

Use a residual-space reduction diagram in Chapter 3 and a Chapter 9 claim-contract diagram if they can be produced from existing equations/schema without inventing empirical results. A joint-event boundary visual and lunar invariant schematic are optional and must remain diagnostic.

## Acceptance criteria

The manuscript has no duplicated future-work roadmap, no false figure provenance, no unqualified finite-sample safety claim, no ambiguous uniform-convergence quantifier, and no Chapter 3--5 section without a meaningful subsubsection hierarchy. A fresh LaTeX build has no undefined references, duplicate labels, or citation errors.
