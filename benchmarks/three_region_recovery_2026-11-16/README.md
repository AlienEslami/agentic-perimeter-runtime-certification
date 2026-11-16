# Three-region congested recovery

This study supplies Section 6.2 and Figure 5 of release
`paper-2026-11-16-v2`. It uses a three-region directed ring, with one exterior
entry queue per region. Regional counts start at 4,500 vehicles and queue
counts at 500 vehicles. A temporary arrival increase affects Queue 1.

The backup and repaired plans have full-hour certificates for an initial
error of 25 vehicles per regional total (25/3 per destination component),
10 vehicles per queue, and independent arrival errors of 300 veh/h per
queue. The repaired plan uses interpolation weight 0.45. Every lower tested
weight on the 0.05 grid has an exact queue-overflow witness. The certificates
are conditional on the stated traffic model and receiving-capacity assumptions.
Larger regional-error attempts at 50 and 75 vehicles were inconclusive and
are retained alongside the successful results.

## Files and units

- `PROTOCOL.md` and `PLAN_SELECTION.md`: model, selection rules and tested plans.
- `study.py`, `refined.py`, `vendor/`: model and verification implementation.
- `source_snapshots/`: immutable sources bound to the saved certificates.
- `results/`: successful and inconclusive proof records, matched simulation
  arrays, exact overflow witnesses and validation summaries.
- `RESULTS.md`: reported outcomes and limits.
- `evaluate.py`: 64 matched simulations per plan and independent checks.
- `plot_all_queues.py`: the six-panel Figure 5; `plot.py` is an earlier view.

Internal computations use units of 100 vehicles, hours and 100 veh/h.
Published arrays named `state_vehicles` and `regional_vehicles`, the report
and figures use vehicles; arrival bounds in the interfaces use veh/h.
In the implementation, method name `class` means destination coordinates.

## Reproduce the published checks

Use Python 3.12 and the repository's numerical requirements. From the
repository root:

```text
python -m pip install -r benchmarks/reusable_perimeter_suite_2026-09-26/requirements-lock.txt
python -B benchmarks/three_region_recovery_2026-11-16/reproduce.py --output-dir output/reruns/three_region_recovery
```

This regenerates the two accepted certificates with the tightened Jacobian
bounds (`net.refined = True`), reruns the 192 matched simulations, recomputes
every certificate interval, and checks the independent vehicle balances,
derivatives and exact overflow witnesses. It writes to a new directory and
refuses to replace existing results. Archived certificates and simulations
remain unchanged. Observed timings depend on the machine and are not latency
guarantees. No optimization sweep or paid API call is involved.

To redraw the published figure directly from the archived results:

```text
python -B benchmarks/three_region_recovery_2026-11-16/plot_all_queues.py
```

The PDF and PNG are written under
`output/review/three_region_paper_2026_11_16/`.
The two successful records are
`monotone_backup_class_d25_w300_proof.json.gz` and
`monotone_repair045_class_d25_w300_proof.json.gz`.
The generic `study.py certify` entry point uses the unrefined check; use
`reproduce.py` for the published successful configuration.
