#!/usr/bin/env python3
"""Plant-only E4 diagnostic when the required 5% certificate is unavailable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mfd_incident_model as model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_uncertainty/required_E3_E4_status.json"
STEP = 15 / 3600
RESIDUAL = .6
WIDTHS = np.full(7, .01)


def first_plan():
    data = json.loads((ROOT / "reproducibility/results/mfd_incident_mpc_grid_results.json").read_text())
    first = next(run for run in data["runs"] if run["amplitude"] == .3 and run["duration_h"] == 1.)["decisions"][0]
    return np.asarray(first["initial_state"], dtype=float), np.asarray(first["controls"], dtype=float)


def profile(name, bound, rng, count):
    if name == "--": return np.tile([-bound, -bound], (count, 1))
    if name == "-+": return np.tile([-bound, bound], (count, 1))
    if name == "+-": return np.tile([bound, -bound], (count, 1))
    if name == "++": return np.tile([bound, bound], (count, 1))
    if name == "alternating":
        return np.asarray([[(-1.) ** (k // 20) * bound, (-1.) ** (k // 20 + 1) * bound]
                           for k in range(count)])
    return rng.uniform(-bound, bound, (count, 2))


def simulate(bound, name, seed):
    rng = np.random.default_rng(seed)
    initial, commands = first_plan()
    x = initial + (rng.uniform(-WIDTHS, WIDTHS) if name == "random" else np.zeros(7))
    count = round(1 / STEP)
    deltas = profile(name, bound, rng, count)
    residuals = rng.uniform(-RESIDUAL, RESIDUAL, count) if name == "random" else np.full(count, RESIDUAL)
    min_margin = float(np.min(model.margins(x, commands[0])))
    min_class = min(x[2], x[3], x[4], x[5], x[0]-x[2]-x[4], x[1]-x[3]-x[5])
    max_queue = x[6]
    for k in range(count):
        t = k * STEP
        u = commands[0 if t < 1 / 12 - 1e-12 else 1]
        d = deltas[k]
        w = float(residuals[k])
        x = model.rk4_uncertain(x, lambda _x, u=u: u, t, STEP,
                                lambda s, w=w: model.arrival(s, amplitude=.3, duration=1.) + w,
                                lambda _s, _x, d=d: d)
        min_margin = min(min_margin, float(np.min(model.margins(x, u))))
        min_class = min(min_class, x[2], x[3], x[4], x[5],
                        x[0]-x[2]-x[4], x[1]-x[3]-x[5])
        max_queue = max(max_queue, float(x[6]))
    return dict(true_delta=bound, profile=name, seed=seed,
                storage_violation=bool(max_queue > model.QUEUE_LIMIT + 1e-9),
                service_violation=bool(x[0] > 28.5 + 1e-9),
                class_nonnegativity_violation=bool(min_class < -1e-9),
                maximum_storage_exceedance_vehicles=max(0., 100 * (max_queue-model.QUEUE_LIMIT)),
                maximum_service_exceedance_vehicles=max(0., 100 * (x[0]-28.5)),
                maximum_class_nonnegativity_exceedance_vehicles=max(0., -100 * min_class),
                deadline_region1_vehicles=100 * x[0], maximum_queue_vehicles=100 * max_queue,
                minimum_sampled_physical_margin=min_margin)


def main():
    e1 = json.loads((OUTPUT.parent / "required_E1_grid.json").read_text())
    five = [row for row in e1["rows"] if row["delta"] == .05]
    assert five and not any(row["accepted"] for row in five)
    records = []
    for bound in (.075, .10, .15):
        for seed in range(100):
            records.append(simulate(bound, "random", 40000 + round(1000 * bound) * 100 + seed))
        for index, name in enumerate(("--", "-+", "+-", "++", "alternating")):
            records.append(simulate(bound, name, 50000 + index))
    summary = []
    for bound in (.075, .10, .15):
        rows = [row for row in records if row["true_delta"] == bound]
        summary.append(dict(
            true_delta=bound, realizations=len(rows),
            storage_violation_rate=sum(row["storage_violation"] for row in rows) / len(rows),
            service_violation_rate=sum(row["service_violation"] for row in rows) / len(rows),
            class_nonnegativity_violation_rate=sum(row["class_nonnegativity_violation"] for row in rows) / len(rows),
            maximum_storage_exceedance_vehicles=max(row["maximum_storage_exceedance_vehicles"] for row in rows),
            maximum_service_exceedance_vehicles=max(row["maximum_service_exceedance_vehicles"] for row in rows),
            maximum_class_nonnegativity_exceedance_vehicles=max(row["maximum_class_nonnegativity_exceedance_vehicles"] for row in rows)))
    payload = dict(
        E3=dict(status="not executable under the requested certified protocol",
                reason="no method obtained the required full-mission 5% certificate at epoch zero",
                initial_candidate_admissions=0, authorized_executions=0,
                realized_violation_metrics="not defined because no plan receives execution authority"),
        E4=dict(status="plant-only diagnostic; requested retained-set audit unavailable",
                reason="E4 presupposes a 5% certificate, but E1 obtained none",
                fresh_box_noncontainment_epochs=None,
                fresh_box_metric_reason="no retained certified checkpoint set exists",
                summary=summary, records=records),
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(E3=payload["E3"], E4=summary), indent=2))


if __name__ == "__main__":
    main()
