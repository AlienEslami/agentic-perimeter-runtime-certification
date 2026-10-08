# Substantial reduction — 8 October 2026

The reduced paper is **44 pages**, down from **70 pages**: 26 fewer pages (37.1%). The 79-page long paper and its 33-page IEEE layout are unchanged, including their compiled PDFs.

## Mathematical material

All eleven formal results remain: four theorems, six propositions and one corollary. Their hypotheses and conclusions are preserved; only two result statements change internal proof-location references. Each result now has its complete proof beside it in Section 4. The short main-text proofs and longer derivations in former Appendices B, C and F have been combined into a single argument. The theorem and proposition numbers are unchanged. Corollary 1 follows the terminal theorem it uses.

The proof review checked nonsmooth Jacobians, initial-set coverage, strict containment, independent positivity and queue comparison, signed uncertainty propagation, inherited remainder bounds, numerical conversion error, actual installation timing, rejection expiry, network conservation and rank, pending service obligations, terminal inclusion and invariant backup conditions, and the common-domain assumption for the cost bound. Essential proofs have no external dependency.

Section 2 retains the continuous traffic model, units, MFD, admission law, physical limits, conservation relation, service requirements, Q/D objectives, uncertainty contract and robust feasible set. It now includes the demand record directly. Section 3 includes the task transitions and their temporal-consistency argument, formerly repeated in Appendix E.

The repository preserves optional implementation details: changing coordinate maps, stable numerical evaluation shortcuts, explicit constant-command residual lists, coefficient and controller-design records, regression checks, and corrections for changing allocation. Their original source appears in `supporting_materials/implementation_details.tex`.

## Experimental material

The main numerical section retains four groups of evidence:

1. Unchanged feasible economic plans, tighter traffic bounds, the binding service example and a compact connected-network demonstration.
2. The 80 deadline contracts, checked repair, fresh reserves and the matched queue/service conflict.
3. Successive plan changes, the 30 eligible matched-pair scan and the D025 storage-overflow example.
4. Computation delays and the distinct 96-call interpretation pilot and eight actual-planning traces.

Five retained numerical tables and the repair-metric equations are unchanged. A short appendix keeps the primary state, plan/repair settings, replay profiles, comparison scope, future-activation setup, ring dynamics and agent-scoring context. Full solver settings, raw calls, all outcomes and reproduction commands are mapped in `supporting_materials/REPRODUCING_THE_REDUCED_PAPER.md`.

The secondary 24-case regional-regulation comparison, eight-event compatibility reporting, extended 17-case online study, full network sweep and detailed histories are preserved on GitHub. Original text is retained in `supporting_materials/experimental_details.tex` and `regional_regulation_task.tex`; saved research records are unchanged. The guide distinguishes later MFD-parameter diagnostics from the paper's nominal-MFD sensing/arrival comparison.

## Preserved text and validation

The abstract, conclusion and future work are byte-identical to the preceding version. The introduction changes only its final roadmap sentence, as explicitly approved. Author declarations and the bibliography database are unchanged. The paper uses the same font size and page dimensions.

The built-in compiler was unavailable on this platform; the existing pdfLaTeX installation compiled the paper with resolved references and citations. The PDF was rendered for visual review. Source checks and independent scientific review confirm the preserved formal claims, internal references and long-version files. No simulations or live API calls were rerun.

The exact preceding package is archived at `archive/reduced_manuscript_2026_10_08_before_substantial_reduction/`. The current audit is `SUBSTANTIAL_REDUCTION_QA_2026-10-08.json`; earlier audits remain records of their corresponding revisions.
