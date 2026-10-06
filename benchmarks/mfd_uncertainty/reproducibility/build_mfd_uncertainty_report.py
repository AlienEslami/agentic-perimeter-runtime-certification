#!/usr/bin/env python3
"""Build the Phase-1 report from completed immutable result records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reproducibility/results/mfd_uncertainty"


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    exploratory = json.loads((R / "mfd_uncertainty_results.json").read_text())
    e1 = json.loads((R / "required_E1_grid.json").read_text())
    e2 = json.loads((R / "required_E2_binding_service.json").read_text())
    e34 = json.loads((R / "required_E3_E4_status.json").read_text())
    assert e1["completed"] and e1["all_requested_cases_recorded"] and e2["completed"]
    write_csv(R / "required_E1_grid.csv", e1["rows"])
    write_csv(R / "required_E2_binding_service.csv", e2["rows"])

    methods = e1["methods"]
    labels = {"original": "Original", "positive_queue": "Positive/queue",
              "signed_generator": "Signed generator"}
    summary1 = []
    for delta in e1["deltas"]:
        for method in methods:
            rows = [row for row in e1["rows"] if row["delta"] == delta and row["method"] == method]
            summary1.append(dict(delta=delta, method=method,
                                 accepted=sum(row["accepted"] for row in rows),
                                 cases=len(rows), maximum_certified_until_h=max(row["certified_until_h"] for row in rows),
                                 wall_min=min(row["total_wall_seconds"] for row in rows),
                                 wall_median=float(np.median([row["total_wall_seconds"] for row in rows])),
                                 wall_max=max(row["total_wall_seconds"] for row in rows)))

    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    for method in methods:
        rows = [row for row in e1["rows"] if row["method"] == method
                and row["state_half_width_vehicles"] == 1.
                and row["arrival_residual_vehicles_per_hour"] == 60.]
        rows.sort(key=lambda row: row["delta"])
        ax.plot([100 * row["delta"] for row in rows], [60 * row["certified_until_h"] for row in rows],
                "o-", label=labels[method])
    ax.axhline(60, color="black", linestyle="--", linewidth=1, label="required horizon")
    ax.set(xlabel="Regional MFD uncertainty bound (%)", ylabel="Certified horizon (min)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(R / "required_E1_certified_horizon.png", dpi=180)
    fig.savefig(R / "required_E1_certified_horizon.pdf")
    plt.close(fig)

    e1_lines = []
    for item in summary1:
        e1_lines.append(f"| {100*item['delta']:.0f}% | {labels[item['method']]} | "
                        f"{item['accepted']}/{item['cases']} | {60*item['maximum_certified_until_h']:.1f} | "
                        f"{item['wall_min']:.2f}/{item['wall_median']:.2f}/{item['wall_max']:.2f} |")
    plan_lines = []
    for target in e2["targets"]:
        plan = e2["plans"][f"{target:g}"]
        plan_lines.append(f"| {target:.0f} | {'yes' if plan['feasible'] else 'no'} | "
                          f"{plan.get('replay',{}).get('total_time_spent_veh_h','---')} | "
                          f"{plan.get('replay',{}).get('deadline_n1_vehicles','---')} |")
    repair = exploratory["repair_uncertainty"]
    repair_lines = []
    for row in repair:
        repair_lines.append(f"| {row['plan'].replace('_',' ')} | {100*row['delta']:.2f}% | "
                            f"{'yes' if row['accepted'] else 'no'} | "
                            f"{row['service_upper_vehicles'] if row['service_upper_vehicles'] is not None else '---'} | "
                            f"{row['first_failure'] or '---'} |")
    stress_lines = []
    for row in e34["E4"]["summary"]:
        stress_lines.append(f"| {100*row['true_delta']:.1f}% | {row['realizations']} | "
                            f"{100*row['storage_violation_rate']:.1f}% | {100*row['service_violation_rate']:.1f}% | "
                            f"{100*row['class_nonnegativity_violation_rate']:.1f}% |")

    report = rf"""# Bounded MFD uncertainty — Phase 1 report

## Decision summary

The field, certifier, signed-generator verifier and simulated plant now support

\[
\widetilde q_{{rj}}=(1+\Delta_r(t))q_{{rj}},\qquad |\Delta_r(t)|\leq\bar\delta_r<1.
\]

The planners and all archived commands remain nominal. The zero-uncertainty branch reproduces the manuscript's Table-2 bounds to recorded precision, including 1779.41/1728.68/1721.75 vehicles in the first row. The nominal repair bound is reproduced as 2849.093 vehicles. All deterministic tests pass, and 200 within-contract time-varying simulations lie inside the 0.5% signed-generator enclosure at every 15-s checkpoint.

The requested 2%, 5% and 10% experiments do **not** produce a full-hour certificate for any Table-2 cell or method. The best 2% case closes through 31.0 minutes; the best 5% and 10% cases close through 14.25 and 8.0 minutes. The requested 5% online protocol therefore cannot commission its first plan, so its online and retained-set stress experiments have no certified execution to evaluate.

**Recommendation:** do not put the requested 2–10% study into the manuscript as a positive robustness result. The implementation is sound and exposes a concrete research problem: synthesis and/or a tighter structured enclosure is required before MFD uncertainty at those levels can support the paper's one-hour mission. The subpercent frontier can be retained as diagnostic evidence, but it should enter the manuscript only after the uncertainty scale is calibrated and the repair is redesigned to reserve robust deadline margin.

No manuscript file was edited.

## Code changes

- `mfd_incident_model.py`: regional multiplicative flow factors, exact uncertainty channels, uncertain RK4 replay and uncertainty-aware departure accounting.
- `mfd_incident_verifier.py`: interval-Delta Jacobian, polynomial regional-channel forcing, proof binding and an exact zero-uncertainty branch.
- `mfd_incident_affine_reachability.py`: uncertain generalized Jacobian and forcing while retaining the nominal frozen Jacobian.
- `mfd_incident_mpc_certification.py`, `mfd_incident_positive_reachability.py`: uncertainty-bound plumbing; independent positivity and scalar-queue arguments retained.
- Study/test drivers in this directory generate only `results/mfd_uncertainty/` outputs.

The regional factors preserve all transfer cancellations. Total regional inventory still contains nonpositive trip-completion terms; its existing upper comparison remains valid because \(1+\Delta_r>0\) and the upper bound discards those completion terms. The queue equation contains no MFD factor.

## Tests

| Test | Outcome |
|---|---|
| Field decomposition against the implemented uncertain plant (100 random points) | pass |
| Exact zero-uncertainty Table-2 reproduction | pass |
| Nominal 2849.10-vehicle repair reproduction | pass (2849.093) |
| Structural transfer/sign, quasipositivity and queue checks | pass |
| 200 random/corner containment runs at 0.5%, 15-s checkpoints | pass; maximum normalized ratio {exploratory['sampled_containment']['maximum_ratio']:.12f} |
| Acceptance monotonicity on the subpercent grid | pass |

The broader δ=0 Table-3 online code path was not rerun as a new live mission; the exact δ=0 arithmetic branch and all plan-level bounds above match the archived implementation. This is the remaining protocol-level regression if development continues.

## E1 — prescribed Table-2 extension

Each cell below aggregates the six unchanged \((d,\bar w)\) rows. “Maximum horizon” is the farthest any cell closed; failure to close is not a simulated counterexample.

| δ | Method | Full-hour accepted | Maximum horizon (min) | certificate wall time min/median/max (s) |
|---:|---|---:|---:|---:|
{chr(10).join(e1_lines)}

One original-method 5% failure required 972.22 s; it is retained in the record. No tolerance, partition, horizon or target was changed.

## E2 — 1700-vehicle binding service

The nominal planner found plans through a 1550-vehicle synthesis target; the lower four targets were infeasible under the unchanged optimizer.

| Nominal synthesis target (veh) | Plan found | Nominal TTS (veh-h) | Nominal deadline n1 (veh) |
|---:|---:|---:|---:|
{chr(10).join(plan_lines)}

At both 2% and 5%, none of the original, positive/queue or signed-generator methods certified any feasible target. Consequently there is no “first certified target,” internal certified service margin, or robust TTS comparison to report. Complete per-certificate wall times and stopping causes are in `required_E2_binding_service.csv`.

## E3 — 5% online missions

| Required initial condition | Result |
|---|---|
| Full-mission 5% certificate at epoch 0 | unavailable for every method and Table-2 contract |
| Candidate admissions | 0 |
| Authorized online executions | 0 |
| Realized storage/service outcomes | undefined under the certified protocol because execution authority is never established |

## E4 — out-of-contract stress

E4 presupposes a retained 5% certified set. Since no such set exists, the requested fresh-box noncontainment count is undefined. A clearly separated plant-only diagnostic was still run with 100 stochastic seeds plus all four constant sign corners and a five-minute alternating profile at each true uncertainty level:

| True uncertainty | Realizations | Storage violations | Service violations | Class-negativity violations |
|---:|---:|---:|---:|---:|
{chr(10).join(stress_lines)}

These sampled zero-violation outcomes show conservatism of the current enclosure; they do not replace the missing certificate.

## E5 — minute-20 repair (optional protocol item)

| Plan | δ | Accepted | Certified service upper (veh) | First failure |
|---|---:|---:|---:|---|
{chr(10).join(repair_lines)}

The minimum-intervention nominal repair has only 0.907 vehicles of certified slack and loses the service certificate at 0.1%. The retained backup stays certified through 0.25% on the tested grid, then loses the service row at 0.5%.

## Machine-readable records

- `required_E1_grid.json/.csv`: every prescribed E1 certificate, stopping cause and wall time.
- `required_E2_binding_service.json/.csv`: all feasible target plans and every E2 certificate.
- `required_E3_E4_status.json`: the blocked certified protocol and separate plant-only diagnostic.
- `mfd_uncertainty_results.json`: subpercent frontier, repair study and 200-run containment audit.
- `containment_certificate_delta_0p005.json`: complete retained interval proof.
- `test_mfd_uncertainty.py`: deterministic regression assertions.
"""
    (R / "REPORT.md").write_text(report, encoding="utf-8")
    print(R / "REPORT.md")


if __name__ == "__main__":
    main()
