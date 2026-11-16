# Reproducing the 15 November 2026 manuscript

Use the `paper-2026-11-15` release with `paper/reduced_2026-11-15/main.tex`.
The manuscript package contains only TeX, bibliography and ten figure assets.
The historical [reproduction guide](REPRODUCTION_GUIDE.md) documents the
portable suite, its model and optional live-agent experiments.

## Current numerical evidence

| Manuscript result | Saved sources |
|---|---|
| Congested recovery, Fig. 5 and Table 2 | `benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery/`, `full_certificates/`, `queue_witnesses_w600.json` |
| Matched 640-simulation comparison, Table 3 | `benchmarks/partb_traffic_presentation_2026-11-15/matched_controller_table.json`, `cbf_ensemble.json`; portable-suite contract D027 |
| Larger initial-error comparison, Table 4 | Portable-suite contracts D027 and D063; `output/review/results_refinement_2026_11_15/evidence_and_source_checks.json` |
| Priority trade-off, Table 5 | `benchmarks/partb_traffic_presentation_2026-11-15/repair_traffic_statistics.json`; portable-suite contract records |
| Arrival-increase comparison, Table 6 | `output/review/results_refinement_2026_11_15/evidence_and_source_checks.json` and the regulation records identified by its source hashes |
| Replanning and delays | Portable-suite `results/sequential_authority_2026-10-01/` and `results/delayed_handover/` |
| Acceptance map and coordinate comparison | `output/review/section6_revision_2026_11_15/verified_evidence.json`; fixed-portfolio records and `benchmarks/manuscript_review_2026-11-08/results/coordinate_comparison.json` |
| Model-error sensitivity | `benchmarks/reviewer_strengthening_2026-11-11/plant_sensitivity.py`, its protocol and saved JSON |
| Checking times and interval evaluation | `benchmarks/reviewer_strengthening_2026-11-11/probe.py`, `timing.py`, `timing_summary.json` and `four_box_bound_summary.json` |
| Agent traces and pilot | Portable-suite live-agent records and `benchmarks/section5_support_2026-11-11/summaries/baseline_and_pilot_summary.json` |

Saved-record checks require Python 3.12 and the numerical packages listed in
the portable suite's requirements files. From the repository root:

```
python -B scripts/audit_repository_records.py
python -B scripts/audit_current_results_2026_11_15.py
python -B scripts/reproduce_current_figures_2026_11_15.py
```

The plotting entry point reconstructs the current scenario, congestion,
replanning and coordinate-comparison figures from saved results. It writes
under `output/package/PERIMETER_PARTB_CONSOLIDATED_2026-11-12/figures/`
and leaves the released manuscript assets unchanged. Historical figure
assembly scripts are retained for provenance; some expect historical
output folders. Conceptual Figures 1-3 are reviewed explanatory illustrations
with AI-assistance disclosures in their captions, not simulation outputs.

For fresh simulations or verification, read the study README before running
its scripts. The reviewer-strengthening folder documents the 600 veh/h case,
the distinct 300 veh/h model-error case, fixed inputs, numerical refinement
and timing scope. About 0.9 GB of intermediate per-case comparison traces
are omitted from this compact publication copy; frozen inputs, aggregate
outcomes, selected full certificates and rerun scripts are supplied. The
earlier complete study archives remain assets of the November 11 release.
No new optimization sweeps or paid API experiments were run to prepare
this release. Stored timings are historical measurements, not latency bounds.
