# Chapter 2 Split and Renumbering Design

## Goal

Split the current background chapter into two coherent chapters and renumber the remaining thesis chapters without changing the scientific content or source-theorem numbering.

## Design

The first new chapter will be `A Survey on Stochastic Optimization and Chance Constraints`. It will contain a brief introduction followed by the current Sections 2.1--2.3: optimization paradigms, classical analytical reformulations, and approximation methods/trade-offs.

The second new chapter will be `Chance Constraints and Kernel Density Estimation`. It will contain a new short introduction followed by the current KDE foundations and literature sections (currently Sections 2.4--2.5). The existing residual-KDE method chapter will become Chapter 4 and retain its content, while the current Chapters 4--10 become Chapters 5--11.

## File mapping

`kap02/kapitel02.tex` becomes the survey chapter; a new `kap03/kapitel03.tex` contains the KDE chapter; existing chapter directories/files are shifted with `git mv` so file names match printed chapter numbers. Figure directories move with their chapter sources. `Masterarbeit.tex` inputs are updated to `kap01` through `kap11`.

## Reference rules

Printed chapter references and chapter-local figure paths are updated globally. Labels remain stable unless they encode a chapter number. Theorems cited from external papers retain their source numbering (for example, Schuster Theorems 3.1--3.3). Local chapter references to the residual-KDE formulation become Chapter 4, biased KDE Chapter 5, convergence theory Chapter 6, implementation Chapter 7, experiments Chapter 9, discussion Chapter 10, and conclusion Chapter 11.

## Acceptance criteria

- The table of contents shows 11 scientific chapters with the requested titles and section numbering.
- The new Chapter 2 contains only the survey sections; the new Chapter 3 contains only KDE foundations/literature sections.
- All `\\input` paths, `\\cref` links, chapter prose, figure paths, and captions resolve after renumbering.
- External theorem numbers and equation/figure labels remain scientifically correct.
- A fresh LaTeX build has no undefined references, duplicate labels, or missing figures.
