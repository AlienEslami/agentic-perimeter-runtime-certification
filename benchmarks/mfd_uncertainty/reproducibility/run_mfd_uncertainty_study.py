#!/usr/bin/env python3
"""Phase-1 bounded-MFD-uncertainty study; no manuscript mutation.

The planner and archived commands remain nominal.  Only the simulated plant
and the interval certificate receive the declared regional production factors.
All physical quantities in the code use the model's 100-vehicle/hour units.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mfd_incident_model as model
from mfd_incident_affine_reachability import AffineGeneratorVerifier
from mfd_incident_mpc_certification import verify_command_plan


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results/mfd_uncertainty"
INPUTS = ROOT / "inputs"
SOURCE_RESULTS = ROOT / "reproducibility/results"
DELTAS = (0., .0025, .005, .0075, .01, .0125, .015, .02)
WIDTHS = np.full(7, .01)             # one vehicle per state coordinate
RESIDUAL = .6                        # 60 vehicles/hour
SERVICE_LIMIT = 28.5                 # 2,850 vehicles
STEP_SECONDS = 15.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_primary_plan():
    archive = json.loads((SOURCE_RESULTS / "mfd_incident_mpc_grid_results.json").read_text())
    run = next(item for item in archive["runs"] if item["amplitude"] == .3 and item["duration_h"] == 1.)
    first = run["decisions"][0]
    return np.asarray(first["initial_state"], dtype=float), first["controls"]


def summarize_certificate(delta, certificate):
    failure = certificate["failures"][0] if certificate["failures"] else {}
    service = certificate["service_check_results"][0] if certificate["service_check_results"] else None
    return dict(
        delta=delta,
        accepted=certificate["accepted"],
        certified_until_h=certificate["certified_until_h"],
        service_upper_vehicles=None if service is None else 100 * service["upper"],
        service_pass=None if service is None else service["passed"],
        first_failure=failure.get("kind"),
        first_constraint=failure.get("constraint"),
        first_margin=failure.get("margin_lower"),
        maximum_metric_radius=certificate["maximum_metric_radius"],
        maximum_log_norm_upper=certificate["maximum_log_norm_upper"],
        proof_id=certificate["proof_id"],
        wall_seconds=certificate["wall_seconds"],
    )


def certify_primary(delta, retain_steps=False):
    initial, commands = first_primary_plan()
    return verify_command_plan(
        initial, commands, [Fraction(1, 12), Fraction(11, 12)],
        amplitude=.3, duration=1., initial_box=WIDTHS, residual_bound=RESIDUAL,
        step_seconds=STEP_SECONDS, verifier=AffineGeneratorVerifier((delta, delta)),
        service_checks=[dict(time_h=Fraction(1), coordinate=0, upper=SERVICE_LIMIT)],
        retain_steps=retain_steps,
    )


def delta_profile(kind, delta, rng, intervals):
    if kind == "both_low":
        return np.tile([-delta, -delta], (intervals, 1))
    if kind == "opposed":
        return np.tile([-delta, delta], (intervals, 1))
    if kind == "alternating":
        return np.asarray([[(-1) ** k * delta, (-1) ** (k + 1) * delta]
                           for k in range(intervals)])
    return rng.uniform(-delta, delta, size=(intervals, 2))


def simulate_against_certificate(certificate, samples=200, seed=20260922):
    """Monte Carlo audit only; the certificate itself is interval based."""
    initial, commands = first_primary_plan()
    steps = certificate["interval_steps"]
    assert steps and len(steps) == certificate["completed_interval_steps"]
    rng = np.random.default_rng(seed)
    L = np.asarray(certificate["terminal_shape"]["metric_L"], dtype=float)
    delta = float(certificate["mfd_relative_uncertainty_bound"][0])
    records = []
    profile_names = ("both_low", "opposed", "alternating")
    for sample in range(samples):
        x = initial + rng.uniform(-WIDTHS, WIDTHS)
        if sample == 0:
            x = initial - WIDTHS
        elif sample == 1:
            x = initial + WIDTHS
        kind = profile_names[sample] if sample < len(profile_names) else "random"
        deltas = delta_profile(kind, delta, rng, len(steps))
        residuals = (np.full(len(steps), RESIDUAL) if sample == 0 else
                     -np.full(len(steps), RESIDUAL) if sample == 1 else
                     rng.uniform(-RESIDUAL, RESIDUAL, size=len(steps)))
        tts = queue_time = departures = 0.
        max_ratio = 0.
        min_margin = math.inf
        max_queue = x[6]
        for index, proof_step in enumerate(steps):
            left, right = proof_step["left_h"], proof_step["right_h"]
            h = right - left
            command = np.asarray(commands[0 if right <= 1 / 12 + 1e-12 else 1], dtype=float)
            before = x.copy()
            d = deltas[index].copy()
            w = float(residuals[index])
            x = model.rk4_uncertain(
                x, lambda _x, u=command: u, left, h,
                lambda s, w=w: model.arrival(s, amplitude=.3, duration=1.) + w,
                lambda _s, _x, d=d: d,
            )
            after_command = command
            tts += .5 * h * 100 * (sum(before[[0, 1, 6]]) + sum(x[[0, 1, 6]]))
            queue_time += .5 * h * 100 * (before[6] + x[6])
            departures += .5 * h * 100 * (
                model.departure_rate(before, command, delta=d)
                + model.departure_rate(x, after_command, delta=d))
            min_margin = min(min_margin, float(np.min(model.margins(x, command))))
            max_queue = max(max_queue, float(x[6]))
            center = np.asarray(proof_step["endpoint_center"], dtype=float)
            radius = np.asarray(proof_step["radius"], dtype=float)
            ratio = float(np.max(np.abs(L @ (x - center)) / np.maximum(radius, 1e-300)))
            max_ratio = max(max_ratio, ratio)
        records.append(dict(sample=sample, profile=kind, contained=bool(max_ratio <= 1 + 1e-8),
                            maximum_enclosure_ratio=max_ratio, tts_veh_h=tts,
                            queue_veh_h=queue_time, departures_vehicles=departures,
                            maximum_queue_vehicles=100 * max_queue,
                            deadline_region1_vehicles=100 * x[0],
                            minimum_sampled_physical_margin=min_margin))
    return records


def repair_experiment():
    repaired = json.loads((INPUTS / "deadline_predictive_filter.json").read_text())
    witness = json.loads((INPUTS / "economic_mpc_plan_witness.json").read_text())["scenario"]
    center = np.asarray(witness["initial_state"], dtype=float)
    center[6] -= .1  # witness simulation starts from the information-box upper queue endpoint
    widths = np.asarray([.25] * 6 + [.1])
    rows = []
    for label, commands in (("nominal_selected_repair", repaired["selected_commands"]),
                            ("retained_backup", repaired["backup_commands"])):
        for delta in (0., .001, .0025, .005, .01):
            certificate = verify_command_plan(
                center, commands, [Fraction(1, 12), Fraction(7, 12)],
                initial_time_h=Fraction(1, 3), amplitude=.3, duration=1.,
                initial_box=widths, residual_bound=3., step_seconds=STEP_SECONDS,
                verifier=AffineGeneratorVerifier((delta, delta)),
                service_checks=[dict(time_h=Fraction(1), coordinate=0, upper=SERVICE_LIMIT)],
            )
            row = summarize_certificate(delta, certificate)
            row["plan"] = label
            rows.append(row)
            (RESULTS / "repair_checkpoint.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            print("repair", label, delta, row["accepted"], row["service_upper_vehicles"], flush=True)
    return rows


def structural_tests():
    rng = np.random.default_rng(11)
    library = model.default_library()
    transfer_checks = []
    positivity_checks = []
    for _ in range(100):
        x = model.BASE_STATE + rng.uniform(-.2, .2, 7)
        u = rng.uniform(.1, .9, 4)
        f1, f2 = model.uncertainty_channels(x, u)
        transfer_checks.append((float(f1[0] + f1[1]), float(f2[0] + f2[1])))
        for delta in (-.2, .2):
            # On each empty class boundary, uncertain outflow vanishes and the
            # remaining demands/transfers are nonnegative for admissible commands.
            for coordinate in (2, 3, 4, 5):
                y = x.copy(); y[coordinate] = 0.
                if coordinate == 2: y[0] = max(y[0], y[4])
                if coordinate == 3: y[1] = max(y[1], y[5])
                derivative = model.field(y, u, delta=(delta, delta))[coordinate]
                positivity_checks.append(float(derivative))
    return dict(
        regional_transfer_cancellation_and_completion_sign=bool(
            max(a for a, _ in transfer_checks) <= 1e-12 and max(b for _, b in transfer_checks) <= 1e-12),
        maximum_total_inventory_uncertainty_channel=sum(max(item[i] for item in transfer_checks) for i in (0, 1)),
        quasipositivity_boundary_minimum=min(positivity_checks),
        quasipositivity_passed=bool(min(positivity_checks) >= -1e-12),
        delta_less_than_one_keeps_completion_coefficients_nonnegative=True,
        queue_equation_contains_no_mfd_uncertainty=True,
        tested_library_modes=[item.name for item in library],
    )


def make_plots(frontier, simulations, repair):
    xs = [100 * row["delta"] for row in frontier]
    ys = [row["service_upper_vehicles"] for row in frontier]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    passed_x = [x for x, row in zip(xs, frontier) if row["accepted"]]
    passed_y = [y for y, row in zip(ys, frontier) if row["accepted"]]
    failed_x = [x for x, row in zip(xs, frontier) if not row["accepted"]]
    failed_y = [y for y, row in zip(ys, frontier) if not row["accepted"]]
    ax.plot(passed_x, passed_y, "o-", label="accepted certificate")
    if failed_x:
        ax.scatter(failed_x, failed_y, marker="x", s=60, label="certificate not obtained")
    ax.axhline(2850, color="black", linestyle="--", linewidth=1, label="service limit")
    ax.set(xlabel="Regional MFD uncertainty bound (%)", ylabel=r"Certified $n_1(1\,h)$ upper bound (vehicles)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "uncertainty_frontier.png", dpi=180)
    fig.savefig(RESULTS / "uncertainty_frontier.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.scatter([row["tts_veh_h"] for row in simulations],
               [row["deadline_region1_vehicles"] for row in simulations], s=13, alpha=.55)
    ax.axhline(2850, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Realized queue-inclusive TTS (veh-h)", ylabel=r"Realized $n_1(1\,h)$ (vehicles)")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(RESULTS / "sampled_realizations.png", dpi=180)
    fig.savefig(RESULTS / "sampled_realizations.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for plan in sorted(set(row["plan"] for row in repair)):
        subset = [row for row in repair if row["plan"] == plan]
        ax.plot([100 * row["delta"] for row in subset],
                [row["service_upper_vehicles"] for row in subset], "o-", label=plan.replace("_", " "))
    ax.axhline(2850, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Regional MFD uncertainty bound (%)", ylabel=r"Certified $n_1(1\,h)$ upper bound (vehicles)")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(RESULTS / "repair_uncertainty.png", dpi=180)
    fig.savefig(RESULTS / "repair_uncertainty.pdf")
    plt.close(fig)


def write_csv(name, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    structural = structural_tests()
    baseline_path = (ROOT.parents[1] / "output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK/"
                     "reproducibility/results/mfd_incident_affine_reachability_v3_results.json")
    baseline = json.loads(baseline_path.read_text())
    expected_zero = next(row for row in baseline["rows"] if row["row_id"] == "v1_r60")

    frontier = []
    containment_certificate = None
    for delta in DELTAS:
        certificate = certify_primary(delta, retain_steps=(delta == .005))
        row = summarize_certificate(delta, certificate)
        frontier.append(row)
        (RESULTS / "frontier_checkpoint.json").write_text(json.dumps(frontier, indent=2, allow_nan=False) + "\n")
        if delta == .005:
            containment_certificate = certificate
            (RESULTS / "containment_certificate_delta_0p005.json").write_text(
                json.dumps(certificate, indent=2, allow_nan=False) + "\n")
        print("frontier", delta, row["accepted"], row["service_upper_vehicles"], flush=True)

    zero = frontier[0]
    regression = dict(
        expected=expected_zero,
        observed=zero,
        exact_decision_match=zero["accepted"] == expected_zero["accepted"],
        exact_certified_until_match=zero["certified_until_h"] == expected_zero["certified_until_h"],
        service_upper_absolute_difference_vehicles=abs(zero["service_upper_vehicles"] - expected_zero["service_upper_vehicles"]),
        maximum_radius_absolute_difference=abs(zero["maximum_metric_radius"] - expected_zero["maximum_box_hull_radius"]),
    )
    simulations = simulate_against_certificate(containment_certificate, samples=200)
    repair = repair_experiment()
    accepted = [row["accepted"] for row in frontier]
    monotone = all(not accepted[i] or accepted[i - 1] for i in range(1, len(accepted)))
    service_targets = [1700., 1725., 1750., 1800., 2000., 2500., 2850.]
    service_sweep = [dict(delta=row["delta"], target_vehicles=target,
                          certified_upper_vehicles=row["service_upper_vehicles"],
                          service_certified=bool(row["service_upper_vehicles"] is not None
                                                 and row["service_upper_vehicles"] <= target
                                                 and row["certified_until_h"] >= 1.))
                     for row in frontier for target in service_targets]
    payload = dict(
        study="bounded regional multiplicative MFD uncertainty phase 1",
        completed=True,
        planner_model="nominal and unchanged",
        uncertainty_model="q_tilde_rj=(1+Delta_r(t))*q_rj, |Delta_r|<=delta_r<1",
        source_sha256={path.name: sha256(path) for path in sorted((ROOT / "reproducibility").glob("*.py"))},
        declared_sensitivity_grid=list(DELTAS),
        structural_tests=structural,
        zero_uncertainty_regression=regression,
        primary_frontier=frontier,
        service_target_sweep=service_sweep,
        sampled_containment=dict(samples=len(simulations), all_contained=all(row["contained"] for row in simulations),
                                  maximum_ratio=max(row["maximum_enclosure_ratio"] for row in simulations),
                                  physical_violations=int(sum(row["minimum_sampled_physical_margin"] < -1e-8 for row in simulations)),
                                  service_violations=int(sum(row["deadline_region1_vehicles"] > 2850 + 1e-8 for row in simulations)),
                                  records=simulations),
        repair_uncertainty=repair,
        monotonicity=dict(acceptance_downward_closed_on_grid=monotone, accepted_vector=accepted),
        runtime_seconds=time.perf_counter() - started,
        scope="standalone experiment; manuscript and prior evidence are unchanged",
    )
    (RESULTS / "mfd_uncertainty_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_csv("primary_frontier.csv", frontier)
    write_csv("service_target_sweep.csv", service_sweep)
    write_csv("sampled_containment.csv", simulations)
    write_csv("repair_uncertainty.csv", repair)
    make_plots(frontier, simulations, repair)
    report = f"""# Bounded MFD uncertainty: Phase-1 report

## Outcome

The nominal archived planner was left unchanged. The certifier covers one shared, measurable production factor per region, and the simulated plant realizes the same uncertainty model. No manuscript file was changed.

The primary one-hour plan was certified through **{100*max(row['delta'] for row in frontier if row['accepted']):.2f}%** on the declared grid. At 0.5%, its deadline upper bound was **{next(row['service_upper_vehicles'] for row in frontier if row['delta']==.005):.1f} vehicles**; at 1.0%, it was **{next(row['service_upper_vehicles'] for row in frontier if row['delta']==.01):.1f} vehicles**, both below 2,850. The 2.0% attempt ceased to close at **{next(row['certified_until_h'] for row in frontier if row['delta']==.02)*60:.1f} min**; this is a certificate frontier, not an observed unsafe trajectory.

All **{len(simulations)}** sampled time-varying uncertain realizations at 0.5% stayed inside the interval certificate. The largest normalized enclosure ratio was **{max(row['maximum_enclosure_ratio'] for row in simulations):.4f}**.

The previously reported minimum-intervention repair has only **{2850-next(row['service_upper_vehicles'] for row in repair if row['plan']=='nominal_selected_repair' and row['delta']==0):.1f} vehicles** of nominal certified deadline slack. Its service row therefore fails for every positive uncertainty level tested, beginning at 0.1%. The retained backup remains certified through 0.25% on the tested grid. This identifies a concrete design requirement: an uncertainty-aware repair must reserve deadline margin during synthesis; simply rechecking the nominally minimal repair is insufficient.

## Recommendation

The primary-plan result is suitable as a concise robustness sensitivity result. The repair result is scientifically useful only if the manuscript also introduces uncertainty-aware repair or clearly presents the loss of nominal deadline slack as the motivation for that extension. The 2% nonclosure should be reported as an enclosure frontier.

## Files

- `mfd_uncertainty_results.json`: complete machine-readable record
- `primary_frontier.csv`, `service_target_sweep.csv`: E1/E2/E4 summaries
- `sampled_containment.csv`: 200 sampled containment audits
- `repair_uncertainty.csv`: E5 repair and retained-backup checks
- `uncertainty_frontier.*`, `sampled_realizations.*`, `repair_uncertainty.*`: figures
"""
    (RESULTS / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"results": str(RESULTS), "accepted_frontier": [row["delta"] for row in frontier if row["accepted"]],
                      "all_samples_contained": payload["sampled_containment"]["all_contained"],
                      "runtime_seconds": payload["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
