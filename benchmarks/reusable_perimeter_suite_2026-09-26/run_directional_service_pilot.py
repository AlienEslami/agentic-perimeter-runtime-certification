"""Diagnostic terminal sensitivities for fixed, deadline-unresolved MPC plans.

Numerical derivatives here are evidence about a proof strategy, not a certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from common import HERE, fixture, save
from run_falsification import SensitivityPrediction, records

OUT = HERE / "results/directional_service_2026-10-01"


def audit_case(path, record, samples):
    contract = record["contract"]
    commands = np.asarray(record["policies"]["full_scenario_mpc"]["commands"])
    predictor = SensitivityPrediction(commands, contract, 40)
    seed = 2026100100 + int(contract["id"][1:])
    rng = np.random.default_rng(seed)
    vectors = [np.r_[np.ones(7), np.ones(40)], np.r_[-np.ones(7), -np.ones(40)]]
    vectors += [rng.uniform(-1, 1, 47) for _ in range(samples)]
    gradients = []
    terminal = []
    for vector in vectors:
        value, derivative_of_negative = predictor.evaluate(vector)
        terminal.append(-value)
        gradients.append(-derivative_of_negative)
    gradients = np.asarray(gradients)
    tol = 1e-8
    lower = gradients.min(axis=0)
    upper = gradients.max(axis=0)
    signs = np.where(lower > tol, 1, np.where(upper < -tol, -1, 0))
    sign_changes = [int(i) for i in range(47) if lower[i] < -tol and upper[i] > tol]
    variable_indices = np.where((abs(lower) > tol) | (abs(upper) > tol))[0]
    min_abs = float(min(np.min(abs(gradients[:, variable_indices])), np.inf)) if len(variable_indices) else None
    original_cert = record["policies"]["full_scenario_mpc"]["certificate"]
    original_upper = original_cert["service_check_results"][0]["upper"] * 100
    proof = original_cert["proof_descriptor"]
    split = proof.get("signed_generator_method", {})
    return {
        "id": contract["id"],
        "contract": contract,
        "commands": commands.tolist(),
        "original_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(vectors),
        "terminal_n1_min_max_vehicles": [float(min(terminal)), float(max(terminal))],
        "gradient_minimum": lower.tolist(),
        "gradient_maximum": upper.tolist(),
        "sampled_signs": signs.tolist(),
        "sampled_sign_changes": sign_changes,
        "smallest_absolute_sampled_active_derivative_vehicles": min_abs,
        "sampled_identically_zero_coordinates": [int(i) for i in range(47) if max(abs(lower[i]), abs(upper[i])) <= tol],
        "original_deadline_upper_vehicles": original_upper,
        "original_deadline_limit_vehicles": contract["deadline_vehicles"],
        "original_signed_generator_record": split,
        "numerical_only": True,
        "random_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    assert args.samples >= 1
    start = time.perf_counter()
    output = [audit_case(path, record, args.samples) for path, record in records()]
    sign_change_cases = [r["id"] for r in output if r["sampled_sign_changes"]]
    signs = np.asarray([r["sampled_signs"] for r in output])
    summary = {
        "date": "2026-10-01", "purpose": "diagnostic_sensitivity_only",
        "case_count": len(output), "interior_samples_per_case": args.samples,
        "sign_change_cases": sign_change_cases,
        "observed_sign_vectors": {str(tuple(v.tolist())): int(sum(np.all(signs == v, axis=1))) for v in np.unique(signs, axis=0)},
        "smallest_absolute_sampled_active_derivative_vehicles": min(r["smallest_absolute_sampled_active_derivative_vehicles"] for r in output),
        "original_deadline_upper_range_vehicles": [min(r["original_deadline_upper_vehicles"] for r in output), max(r["original_deadline_upper_vehicles"] for r in output)],
        "elapsed_seconds": time.perf_counter() - start,
        "certified_new_cases": 0,
        "warning": "Sampled derivatives are not a whole-family monotonicity proof.",
    }
    save(OUT / "cases.json", output)
    save(OUT / "sensitivity_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
