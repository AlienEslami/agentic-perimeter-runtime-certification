# Standalone paper — 7 October 2026

**Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control**

`main.tex` is the primary Elsevier single-column manuscript. Its abstract and
introduction are preserved exactly from the author-approved version.
The argument, mathematical proofs and essential experimental specifications
are contained in one document. Existing proof appendices A–F are followed by:

- G: experimental protocols and validation;
- H: certification comparisons and service margins;
- I: connected-network experiment;
- J: agent experiment protocol.

`main_ieee.tex` contains the same content in an alternative two-column reading
layout. Use `main.tex` for the non-IEEE paper. A separate supplement is no longer
required. The preceding supplement and historical studies are preserved in the
private repository's `archive/` directory. Full current numerical records,
certificates and recorded agent calls remain in its `benchmarks/` directories.

Repository: https://github.com/AlienEslami/agentic-perimeter-runtime-certification
The repository remains private during preparation, with public release planned
before submission.

For compilation, use pdfLaTeX, BibTeX and two further pdfLaTeX passes. The source
package contains both layouts, the bibliography and generated bibliography
files, declarations, and the figure assets used by the integrated paper.
Select `main.tex` when opening the package in Overleaf.

`STANDALONE_MATERIALS_MAP.md` maps the integrated and archived material.
`STANDALONE_QA_2026-10-07.json` records preservation, reference and PDF checks.
`submission_declarations.tex` contains the author-supplied declarations;
`highlights.txt` contains submission highlights. Remaining author and release
checks are in `SUBMISSION_ITEMS_TO_CONFIRM.txt`.

Numerical models, reported values and stored experimental evidence are
preserved. This revision reorganizes the manuscript and introduces no new
optimization runs or live model calls.
