MANUSCRIPT REVIEW EVIDENCE - 8 NOVEMBER 2026

This folder adds a matched coordinate comparison, a continuous queue-bound
figure, seven-state timing measurements, and summaries of archived scans.
PROTOCOL.txt records the choices fixed before the new runs. Existing plans,
model coefficients, numerical kernels, and uncertainty contracts are unchanged.

Place this folder at benchmarks/manuscript_review_2026-11-08/ in the
reproduction repository. It uses the existing portable benchmark at
benchmarks/reusable_perimeter_suite_2026-09-26/, including its vendor sources,
anchor.json, protocols, contract records, and sequential/agent records.
Requirements: Python 3.12, NumPy, SciPy, Matplotlib. Runtime measurement
currently reads the processor name from the Windows registry.

Commands from the repository root:

python benchmarks/manuscript_review_2026-11-08/run_checks.py sweep
python benchmarks/manuscript_review_2026-11-08/run_checks.py trace
python benchmarks/manuscript_review_2026-11-08/run_checks.py runtime
python benchmarks/manuscript_review_2026-11-08/plot_certificate.py
python benchmarks/manuscript_review_2026-11-08/extract_records.py

The sweep reuses completed coordinates_*.json files. For a fresh sweep,
use a separate copy of this folder without those cached results. Preserve
coordinate_fixture.json, which freezes the archived initial state and plan.
The trace and runtime commands recompute their results. Plotting uses the
saved trace and independently integrates ten scalar queue replays.

The supplied POSITIVE_REACHABILITY_PROTOCOL.md is the archived protocol
used by the positivity verifier's provenance check. run_checks.py adapts
only source/protocol path lookup. It does not change the numerical routines.

Interpretation:
- The five methods use the same fixed minute-0 plan, 15-second partition,
  60 veh/h residual, and d in {1,5,10,25,50,100}. Queue half-width is min(d,10).
- Destination-class and conservation boxes both pass through d=10 on
  this grid. Conservation tightens the deadline bound from 2075 to 1780
  vehicles at d=1. Signed propagation passes through d=50. This is a
  tested-grid result, not a continuous maximum or field sensor specification.
- The continuous-bound figure covers the existing D027 repaired plan
  over minutes 20-60. The 160 interval bounds apply continuously to all
  admissible measurable residuals. The ten replays only illustrate and
  numerically check containment; they do not establish the certificate.
- Runtime measurements are observed wall times on the stated machine,
  not worst-case time guarantees. Sweep runs used two workers and their
  recorded times are not the serial runtime comparison. The new timing
  experiment does not replace the delayed-handover study's recorded times.
- Archived violation ranges use each affected contract's largest observed
  excess. Unresolved certification is kept separate from observed violation.

Key outputs:
results/coordinate_comparison.json and coordinates_*.json
results/D027_repair_continuous_certificate.json
results/runtime.json
results/archived_record_checks.json
results/continuous_queue_bounds.csv and queue_bound_replays.csv
results/figure_validation.json
figures/certified_queue_enclosure.pdf and .png

The accompanying manuscript source ZIP and review report identify the
corresponding sections, figures, and tables. No API calls or new planning
optimization are required for these additional checks.
