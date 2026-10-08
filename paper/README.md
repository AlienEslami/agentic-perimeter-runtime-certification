# Long and reduced manuscripts — 8 October 2026

**Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control**

- `main_reduced.tex` / `main_reduced.pdf`: the substantially reduced, standalone paper, **45 pages** including references.
- `main.tex` / `main.pdf`: the unchanged long paper, **79 pages**.
- `main_ieee.tex` / `main_ieee.pdf`: the unchanged long paper in its **33-page** IEEE reading layout.

The reduced version is 25 pages shorter than its preceding 70-page version (35.7%). All four theorems, six propositions and the corollary retain their mathematical statements and have complete proofs in the main text. Repeated derivations and explanations were consolidated. Essential model, task and experimental settings remain in the paper; detailed reproduction instructions, implementation variants and secondary comparisons are in `supporting_materials/` and the research folders identified by the guide.

The abstract is unchanged. The introduction differs only in its final roadmap sentence, explicitly approved to identify the new proof locations. The conclusion, future-work paragraph and bibliography database are unchanged. The preceding reduced-only revision made Jiangbo Yu corresponding author, removed the AI-assistance declaration and fixed the page-5 float gap. The latest revision explains results-table labels, units, methods and counts in the captions, and clarifies experimental terminology. Numerical values are unchanged. The long manuscripts retain their prior author details and declarations. Font size and page dimensions are unchanged. No numerical experiments or live model calls were rerun.

For Overleaf, select `main_reduced.tex` as the main document. Compile with pdfLaTeX, BibTeX and two further pdfLaTeX passes. The reduced-only source ZIP includes the bibliography, declarations and architecture figure. `PERIMETER_LONG_AND_REDUCED_2026-10-08_SOURCE.zip` additionally includes the unchanged long versions.

See `RESULTS_CAPTION_CLARIFICATIONS_2026-10-08.txt` and `RESULTS_CAPTION_QA_2026-10-08.json` for the current changes, checks and hashes. `MANUSCRIPT_LAYOUT_AND_AI_MENTIONS_2026-10-08.txt` contains updated locations; `AUTHOR_FLOAT_QA_2026-10-08.json` records the preceding 43-page revision. `SUBSTANTIAL_REDUCTION_NOTES_2026-10-08.md` and its audit describe the preceding 44-page revision. Earlier audit files describe their own historical revisions. `STANDALONE_MATERIALS_MAP.md` records the earlier integration retained by the long paper.

The exact preceding package, including the 70-page reduced paper, is preserved at `archive/reduced_manuscript_2026_10_08_before_substantial_reduction/` in the repository. Supporting `.tex` extracts preserve the old appendix and native-regulation details; they are reference fragments, not separate compilable manuscripts.

Repository: https://github.com/AlienEslami/agentic-perimeter-runtime-certification

The repository remains private. Author and release checks are tracked separately in `SUBMISSION_ITEMS_TO_CONFIRM.txt` and the repository's `PUBLIC_RELEASE_CHECKLIST.md`.
