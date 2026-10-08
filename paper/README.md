# Long and reduced manuscripts — 8 October 2026

**Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control**

- `main_reduced.tex` / `main_reduced.pdf`: the substantially reduced, standalone paper, **44 pages** including references.
- `main.tex` / `main.pdf`: the unchanged long paper, **79 pages**.
- `main_ieee.tex` / `main_ieee.pdf`: the unchanged long paper in its **33-page** IEEE reading layout.

The reduced version is 26 pages shorter than its preceding 70-page version (37.1%). All four theorems, six propositions and the corollary retain their mathematical statements and have complete proofs in the main text. Repeated derivations and explanations were consolidated. Essential model, task and experimental settings remain in the paper; detailed reproduction instructions, implementation variants and secondary comparisons are in `supporting_materials/` and the research folders identified by the guide.

The abstract is unchanged. The introduction differs only in its final roadmap sentence, explicitly approved to identify the new proof locations. The conclusion, future-work paragraph, author declarations and bibliography database are unchanged. Font size and page dimensions are unchanged. No numerical experiments or live model calls were rerun.

For Overleaf, select `main_reduced.tex` as the main document. Compile with pdfLaTeX, BibTeX and two further pdfLaTeX passes. The reduced-only source ZIP includes the bibliography, declarations and architecture figure. `PERIMETER_LONG_AND_REDUCED_2026-10-08_SOURCE.zip` additionally includes the unchanged long versions.

See `SUBSTANTIAL_REDUCTION_NOTES_2026-10-08.md` for the changes and `SUBSTANTIAL_REDUCTION_QA_2026-10-08.json` for current checks and hashes. Earlier audit files describe their own historical revisions. `STANDALONE_MATERIALS_MAP.md` records the earlier integration retained by the long paper.

The exact preceding package, including the 70-page reduced paper, is preserved at `archive/reduced_manuscript_2026_10_08_before_substantial_reduction/` in the repository. Supporting `.tex` extracts preserve the old appendix and native-regulation details; they are reference fragments, not separate compilable manuscripts.

Repository: https://github.com/AlienEslami/agentic-perimeter-runtime-certification

The repository remains private. Author and release checks are tracked separately in `SUBMISSION_ITEMS_TO_CONFIRM.txt` and the repository's `PUBLIC_RELEASE_CHECKLIST.md`.
