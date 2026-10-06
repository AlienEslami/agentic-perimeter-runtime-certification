"""Isolate initial-box and residual contributions to one unchanged service bound.

Zeroing one uncertainty source is a diagnostic only; it does not change the
installed contract or create an admissible certificate for that contract.
"""
from __future__ import annotations

import json
import numpy as np

from common import HERE, START, DURATIONS, fixture, save
from run_falsification import records
from mfd_incident_affine_reachability import AffineGeneratorVerifier
from mfd_incident_mpc_certification import verify_command_plan


def main():
    record = next(r for _, r in records() if r["contract"]["id"] == "D027")
    c = record["contract"]
    commands = np.asarray(record["policies"]["full_scenario_mpc"]["commands"])
    pairs = [
        ("original", c["width_vehicles"]/100, c["residual_veh_h"]/100),
        ("initial_box_only", c["width_vehicles"]/100, 0),
        ("arrival_residual_only", 0, c["residual_veh_h"]/100),
        ("exact_initial_and_arrival", 0, 0),
    ]
    result = []
    for name, width, residual in pairs:
        proof = verify_command_plan(
            np.asarray(fixture()["center"]), commands, DURATIONS,
            initial_time_h=START, amplitude=c["amplitude"],
            duration=c["duration_h"], initial_box=width, residual_bound=residual,
            step_seconds=15, verifier=AffineGeneratorVerifier(),
            service_checks=[dict(time_h=1, coordinate=0, upper=c["deadline_vehicles"]/100)],
        )
        result.append(dict(name=name, diagnostic_width_vehicles=100*width,
                           diagnostic_residual_veh_h=100*residual,
                           service_upper_vehicles=100*proof["service_check_results"][0]["upper"],
                           physical_verified=proof["verified_continuous_physical"],
                           wall_seconds=proof["wall_seconds"]))
    save(HERE / "results/directional_service_2026-10-01/service_bound_ablation_D027.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
