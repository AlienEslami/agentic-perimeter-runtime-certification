# Current manuscripts — 16 October 2026

**Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control**

- `main_reduced.tex` / `main_reduced.pdf`: current reduced standalone paper, **45 pages**.
- `main.tex` / `main.pdf`: current complete paper, **79 pages**.

Both versions now share corresponding author Jiangbo Yu, consistent method names, units, repair counts and disambiguated mathematical symbols. Appendix equations use `(B.1)` while subsection references retain `Appendix B.1`. The AI manuscript-preparation declaration is omitted from both at the author's explicit final instruction. The abstract, introduction, conclusion and future work retain their previous wording. Scientific results and all eleven formal claims are preserved.

The reduced paper contains complete proofs in its main text and one essential-settings appendix. Additional experimental/reproduction material remains mapped in `supporting_materials/REPRODUCING_THE_REDUCED_PAPER.md`.

See `MANUSCRIPT_CONSISTENCY_REVIEW_2026-10-16.txt` for the assessment and symbol mapping, and `NOTATION_CONSISTENCY_QA_2026-10-16.json` for checks and hashes. Earlier dated reviews and `main_ieee.*` describe historical snapshots; the archived 8 October source bundle preserves their matching dependencies.

For Overleaf, select `main_reduced.tex` or `main.tex` as main document. Use pdfLaTeX and BibTeX. The 16 October ZIPs contain current sources, figures, bibliography, prebuilt bibliography files and this review; no old manuscript or historical IEEE layout is included. The complete bundle also includes the frozen arrays and portable generator for the Figure H.2 label correction.

No simulations or live model calls were rerun. Font sizes and page dimensions are unchanged. Both PDFs compiled cleanly and were visually reviewed.

Repository: https://github.com/AlienEslami/agentic-perimeter-runtime-certification
