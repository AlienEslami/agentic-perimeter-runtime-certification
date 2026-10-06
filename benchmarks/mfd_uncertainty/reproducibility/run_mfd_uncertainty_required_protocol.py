#!/usr/bin/env python3
"""Execute the exact E1 grid requested for bounded MFD uncertainty.

Results are append-only/resumable and separate from the exploratory subpercent
frontier.  Failed enclosures are recorded without changing numerical settings.
"""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from mfd_incident_affine_reachability import AffineGeneratorVerifier
from mfd_incident_mpc_certification import ConstantCommandVerifier, verify_command_plan
from mfd_incident_positive_reachability import PositiveVerifier


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "reproducibility/results/mfd_uncertainty"
OUTPUT = OUTDIR / "required_E1_grid.json"
GRID = ((1., 60.), (10., 60.), (50., 60.), (100., 60.), (1., 300.), (25., 300.))
DELTAS = (0., .02, .05, .10)
METHODS = ("original", "positive_queue", "signed_generator")


def first_plan():
    data = json.loads((ROOT / "reproducibility/results/mfd_incident_mpc_grid_results.json").read_text())
    first = next(run for run in data["runs"] if run["amplitude"] == .3 and run["duration_h"] == 1.)["decisions"][0]
    return first["initial_state"], first["controls"]


def verifier(name, delta):
    bound = (delta, delta)
    if name == "original":
        return ConstantCommandVerifier(delta_bound=bound)
    if name == "positive_queue":
        return PositiveVerifier("positive_queue", delta_bound=bound)
    return AffineGeneratorVerifier(delta_bound=bound)


def summarize(method, delta, width, residual, certificate, elapsed):
    failure = certificate["failures"][0] if certificate["failures"] else {}
    service = certificate["service_check_results"][0] if certificate["service_check_results"] else None
    return dict(
        method=method, delta=delta, state_half_width_vehicles=width,
        arrival_residual_vehicles_per_hour=residual,
        accepted=certificate["accepted"], certified_until_h=certificate["certified_until_h"],
        service_upper_vehicles=None if service is None else 100 * service["upper"],
        service_pass=None if service is None else service["passed"],
        first_failure=failure.get("kind"), first_constraint=failure.get("constraint"),
        first_margin=failure.get("margin_lower"),
        maximum_metric_radius=certificate["maximum_metric_radius"],
        proof_id=certificate["proof_id"], certificate_wall_seconds=certificate["wall_seconds"],
        total_wall_seconds=elapsed,
    )


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = (json.loads(OUTPUT.read_text()) if OUTPUT.exists() else
               dict(protocol="exact requested E1 Table-2 extension", grid=[list(x) for x in GRID],
                    deltas=list(DELTAS), methods=list(METHODS), settings=dict(step_seconds=15.,
                    deadline_vehicles=2850., queue_width_rule="min(d,10) vehicles"), rows=[]))
    complete = {(row["method"], row["delta"], row["state_half_width_vehicles"],
                 row["arrival_residual_vehicles_per_hour"]) for row in payload["rows"]}
    initial, commands = first_plan()
    for delta in DELTAS:
        for width, residual in GRID:
            widths = np.asarray([width / 100.] * 6 + [min(width, 10.) / 100.])
            for method in METHODS:
                key = (method, delta, width, residual)
                if key in complete:
                    continue
                started = time.perf_counter()
                certificate = verify_command_plan(
                    initial, commands, [Fraction(1, 12), Fraction(11, 12)],
                    amplitude=.3, duration=1., initial_box=widths,
                    residual_bound=residual / 100., step_seconds=15.,
                    verifier=verifier(method, delta),
                    service_checks=[dict(time_h=Fraction(1), coordinate=0, upper=28.5)],
                )
                row = summarize(method, delta, width, residual, certificate, time.perf_counter() - started)
                payload["rows"].append(row)
                OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                print(json.dumps({k: row[k] for k in ("method", "delta", "state_half_width_vehicles",
                      "arrival_residual_vehicles_per_hour", "accepted", "certified_until_h",
                      "service_upper_vehicles", "first_failure")}), flush=True)
    payload["completed"] = True
    payload["all_requested_cases_recorded"] = len(payload["rows"]) == len(GRID) * len(DELTAS) * len(METHODS)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
