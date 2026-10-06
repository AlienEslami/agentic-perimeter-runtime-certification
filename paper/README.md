# Submission preparation revision - 6 October 2026

The title is **Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control**. The 6 October update adopts the
author's abstract with the agreed terminal-condition qualification and
collective experimental wording.

This revision updates the 1 October sequential-handover manuscript with
consistent physical and normalized uncertainty notation, clearer system
definitions, affirmative wording, separated timing and interpretation
subsections, and author-supplied submission declarations. Numerical model
parameters, experiment records and numerical table values are preserved.
The original 1 October source archive remains unchanged.

`submission_declarations.tex` contains the funding, competing-interest,
CRediT, data/code availability and AI-assistance statements. The data/code
availability statement includes the repository link. The repository is private
during manuscript preparation and will be made public before submission:
https://github.com/AlienEslami/agentic-perimeter-runtime-certification

`highlights.txt` supplies the five submission highlights.
`SUBMISSION_ITEMS_TO_CONFIRM.txt` records the remaining author approval,
historical AI-use and journal upload checks. `REVISION_QA_2026-10-05.json`
records source consistency and PDF checks.

## Original sequential-handover revision

This package preserves the 27 September concise manuscript and adds the
sequential planning-tool experiment. The prior package and archive remain
unchanged.

- `main.tex`: Transportation Research Part B / Elsevier one-column paper.
- `main_ieee.tex`: content-matched IEEE two-column reading layout.
- `supplement.tex`: companion methods and complete sequential results.
- `FOUNDATION_REFERENCES.bib`, the cross-reference files and `figures/` are
  included for independent compilation.

Compile each document with pdfLaTeX, BibTeX, then pdfLaTeX twice, or use
`latexmk -pdf -bibtex main.tex` (substitute the desired document name).
Select the appropriate main document when uploading the folder to Overleaf.

The new article subsection reports all three preselected sequential cases
and the first case found in a complete matched-pair discovery scan. Supplement
S1.7 reports all three discovered queue-overflow cases, the six matched
schedule comparisons and the no-new-measurement proof protocol. The experiment
uses deterministic stored planner requests; the separate live-agent evidence
remains in the preceding manuscript sections. The benchmark scripts, proofs,
matched trajectories and independent adaptive witnesses are in
`benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/`
in the workspace.

The exact-activation sequential simulation and the timed-preparation study
are distinct experiments. The manuscript's claim concerns certification of
the inherited traffic state and retention of the existing proof at each
scheduled plan change. The associated traffic-performance numbers refer to
the stated benchmark cases; the experiment does not compare against a robust
MPC that reoptimizes at both later decision times.
