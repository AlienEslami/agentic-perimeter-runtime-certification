# Reproducing the 16 November 2026 recovery manuscript

Use release `paper-2026-11-16-v2` and
`paper/reduced_2026-11-16-v2/main.tex`. The PDF includes the Section 6
consistency review and the new three-region congested-recovery study.
Earlier complete and reduced manuscripts and release tags remain unchanged.

## Build the paper

From `paper/reduced_2026-11-16-v2/`, run:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Use Elsevier's `cas-sc` class and `cas-model2-names.bst` from a current TeX
distribution (or Overleaf). All referenced figure assets and the bibliography
are included. The standalone LaTeX ZIP contains only `main.tex`, the
bibliography and referenced images. It contains no generated build files.

## Numerical evidence

| Current manuscript result | Saved evidence |
|---|---|
| Two-region congested recovery, Figure 4 and Table 2 | `benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery/`, `full_certificates/`, `queue_witnesses_w600.json` |
| Three-region congested recovery, Figure 5 | `benchmarks/three_region_recovery_2026-11-16/` |
| Matched two-region priority comparison, Tables 3 and 4 | `benchmarks/partb_traffic_presentation_2026-11-15/matched_controller_table.json`, `cbf_ensemble.json`; portable-suite contracts D027 and D063 |
| Priority and total-time trade-off, Table 5 | `benchmarks/partb_traffic_presentation_2026-11-15/repair_traffic_statistics.json` |
| Supporting comparisons, timing, replanning, agent and model-error studies | [15 November guide](REPRODUCTION_GUIDE_2026-11-15.md), with its historical figure/table numbers |

For the new study, follow its [README](../../benchmarks/three_region_recovery_2026-11-16/README.md).
It provides a command to regenerate both full-hour certificates and the
192 matched simulations into a new output directory, and a command to
redraw all three regions and queues from the archived results. The complete
saved source snapshots and inconclusive attempts are included.

To redraw Figure 4 from existing records, run the following from the
repository root, after installing the numerical requirements:

```text
python -B scripts/plot_recovery_compact_2026_11_16.py
```

The plotting scripts write generated figures under `output/`; they do not
replace the released PDF or the assets inside the dated paper directory.
The released assets are the authoritative figures used in the compiled PDF.

The new study uses 64 matched scenarios for each of three prescribed plans.
The two-region comparisons use 640 simulations per plan. These finite
simulations measure traffic performance; the successful certificates cover
the corresponding full uncertainty families. The model assumptions, distinct
uncertainty sets and experiment-specific settings are stated in the paper.
Recorded runtimes are measurements, not worst-case guarantees.

`PAPER_RELEASE_MANIFEST_2026-11-16-v2.json` lists hashes of the new release
files. Older records keep their previous manifests and immutable snapshots.
