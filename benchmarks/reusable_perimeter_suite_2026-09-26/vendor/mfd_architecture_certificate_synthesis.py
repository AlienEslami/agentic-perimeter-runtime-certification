#!/usr/bin/env python3
"""Joint quadratic-certificate design for the corrected P/PI MFD fibers.

The current manuscript constructs P and PI Lyapunov matrices independently,
then evaluates the P-to-PI transfer.  This script instead places

    T_P,PI.T P_PI T_P,PI <= nu P_P

inside a fixed-nu SDP and bisects on nu.  Both matrices are trace-normalized,
which prevents a meaningless reduction obtained only by rescaling P_PI.  The
script also evaluates the Euclidean coefficient-sum and certificate-metric
remainder bounds for the resulting matrices.

The fixed-gain SOS search is in ``mfd_certificate_synthesis.py``.  The much
larger 12/14-state SOS extension can use the matrices exported here, but the
quadratic transition co-design and its numerical residuals remain separately
auditable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import brentq

from mfd_architecture_validation import (
    DIRECTIONAL_LIMIT,
    FLOW_RATE,
    P_TO_PI_HEADROOM,
    U_STAR,
    X_STAR,
    coefficient_bounds,
    physical_polynomials,
    state_constraint_rows,
    synthesize,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"


def import_cvxpy():
    try:
        import cvxpy as cp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "cvxpy is required. Install `requirements-synthesis.txt` first."
        ) from exc
    return cp


def select_solver(cp, requested: str) -> str:
    installed = set(cp.installed_solvers())
    if requested != "auto":
        if requested not in installed:
            raise SystemExit(f"Solver {requested} is unavailable; installed: {sorted(installed)}")
        return requested
    for candidate in ("MOSEK", "CLARABEL", "CVXOPT", "SCS"):
        if candidate in installed:
            return candidate
    raise SystemExit("Install a semidefinite-capable CVXPY solver")


def solve(problem, solver: str, verbose: bool) -> None:
    options: dict[str, Any] = {"solver": solver, "verbose": verbose}
    if solver == "SCS":
        options.update({"eps": 2.0e-7, "max_iters": 300_000})
    problem.solve(**options)


def symmetrize(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


def generalized_max(successor: np.ndarray, predecessor: np.ndarray) -> float:
    return float(np.max(eigvalsh(successor, predecessor)))


def independent_records(design: dict) -> tuple[np.ndarray, np.ndarray, float]:
    matrices = []
    for matrix in (design["AP"], design["API"]):
        shifted = matrix + 0.5 * FLOW_RATE * np.eye(matrix.shape[0])
        from scipy.linalg import solve_continuous_lyapunov

        matrices.append(solve_continuous_lyapunov(shifted.T, -np.eye(matrix.shape[0])))
    transfer = np.vstack((np.eye(12), np.zeros((2, 12))))
    factor = generalized_max(transfer.T @ matrices[1] @ transfer, matrices[0])
    return matrices[0], matrices[1], factor


def factor_problem(cp, design: dict, factor: float, solver: str, verbose: bool) -> dict:
    p_p = cp.Variable((12, 12), symmetric=True)
    p_pi = cp.Variable((14, 14), symmetric=True)
    slack = cp.Variable()
    i_p, i_pi = np.eye(12), np.eye(14)
    transfer = np.vstack((np.eye(12), np.zeros((2, 12))))
    constraints = [
        p_p >> 1.0e-7 * i_p,
        p_pi >> 1.0e-7 * i_pi,
        cp.trace(p_p) == 1.0,
        cp.trace(p_pi) == 1.0,
        design["AP"].T @ p_p + p_p @ design["AP"] + FLOW_RATE * p_p << -slack * i_p,
        design["API"].T @ p_pi + p_pi @ design["API"] + FLOW_RATE * p_pi << -slack * i_pi,
        transfer.T @ p_pi @ transfer << factor * p_p,
        slack >= 0.0,
    ]
    problem = cp.Problem(cp.Maximize(slack), constraints)
    solve(problem, solver, verbose)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or p_p.value is None or p_pi.value is None:
        return {"feasible": False, "status": problem.status}
    matrices = [symmetrize(np.asarray(p_p.value)), symmetrize(np.asarray(p_pi.value))]
    actual_factor = generalized_max(transfer.T @ matrices[1] @ transfer, matrices[0])
    flow_slacks = [
        -float(np.max(eigvalsh(a.T @ p + p @ a + FLOW_RATE * p)))
        for a, p in zip((design["AP"], design["API"]), matrices)
    ]
    min_eigenvalue = min(float(np.min(eigvalsh(p))) for p in matrices)
    feasible = (
        min_eigenvalue >= -2.0e-7
        and min(flow_slacks) > 2.0e-8
        and actual_factor <= factor + max(2.0e-6, factor * 2.0e-6)
    )
    return {
        "feasible": feasible,
        "status": problem.status,
        "objective_slack": float(slack.value),
        "actual_transition_factor": actual_factor,
        "flow_slacks": flow_slacks,
        "minimum_matrix_eigenvalue": min_eigenvalue,
        "matrices": matrices,
    }


def codesign(design: dict, solver: str, tolerance: float, verbose: bool) -> dict:
    cp = import_cvxpy()
    _baseline_p, _baseline_pi, baseline_factor = independent_records(design)
    lower, upper = 1.0, baseline_factor
    best = factor_problem(cp, design, upper, solver, verbose)
    while not best["feasible"] and upper < 1.0e3:
        upper *= 2.0
        best = factor_problem(cp, design, upper, solver, verbose)
    if not best["feasible"]:
        raise RuntimeError("Could not bracket a feasible trace-normalized architecture factor")
    history = []
    while upper - lower > tolerance:
        tested = 0.5 * (lower + upper)
        trial = factor_problem(cp, design, tested, solver, verbose)
        history.append({key: value for key, value in trial.items() if key != "matrices"} | {"factor_tested": tested})
        if trial["feasible"]:
            upper, best = tested, trial
        else:
            lower = tested
    matrices = best.pop("matrices")
    transfer = np.vstack((np.eye(12), np.zeros((2, 12))))
    factor = generalized_max(transfer.T @ matrices[1] @ transfer, matrices[0])
    return {
        "solver": solver,
        "normalization": "trace(P_P)=trace(P_PI)=1",
        "transition_factor": factor,
        "architecture_dwell_time": math.log(max(1.0, factor)) / FLOW_RATE,
        "flow_rate": FLOW_RATE,
        "flow_slacks": [
            -float(np.max(eigvalsh(a.T @ p + p @ a + FLOW_RATE * p)))
            for a, p in zip((design["AP"], design["API"]), matrices)
        ],
        "matrices": matrices,
        "baseline_independent_factor": baseline_factor,
        "final_solver_record": best,
        "bisection_history": history,
    }


def nonlinear_vectors(command_rows: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    dimension = command_rows.shape[1]
    vectors: dict[tuple[int, ...], np.ndarray] = {}
    for output, polynomial in enumerate(physical_polynomials(command_rows)):
        for exponent, coefficient in polynomial.items():
            if sum(exponent) >= 2:
                vector = vectors.setdefault(exponent, np.zeros(dimension))
                vector[output] = coefficient
                vector[6 + output] = coefficient
    return vectors


def constraint_level(p_matrix: np.ndarray, command_rows: np.ndarray) -> tuple[float, str]:
    inverse = np.linalg.inv(p_matrix)
    rows, margins = state_constraint_rows(p_matrix.shape[0])
    candidates = [
        (float(margin**2 / (row @ inverse @ row)), f"state row {index}")
        for index, (row, margin) in enumerate(zip(rows, margins))
    ]
    for index, row in enumerate(command_rows):
        candidates.append((float(0.25 / (row @ inverse @ row)), f"input row {index}"))
    candidates.extend(
        [
            (
                float(DIRECTIONAL_LIMIT**2 / ((command_rows[1] - command_rows[0]) @ inverse @ (command_rows[1] - command_rows[0]))),
                "u21-u12 directional",
            ),
            (
                float(DIRECTIONAL_LIMIT**2 / ((command_rows[3] - command_rows[2]) @ inverse @ (command_rows[3] - command_rows[2]))),
                "u20-u02 directional",
            ),
        ]
    )
    return min(candidates)


def bound_root(slack: float, first: float, second: float, third: float) -> float:
    return float(
        brentq(
            lambda radius: 2.0
            * (first * radius + second * radius**2 + third * radius**3)
            - slack,
            0.0,
            10.0,
        )
    )


def remainder_record(matrix: np.ndarray, command_rows: np.ndarray, p_matrix: np.ndarray, name: str) -> dict:
    residual = matrix.T @ p_matrix + p_matrix @ matrix + FLOW_RATE * p_matrix
    slack = -float(np.max(eigvalsh(residual)))
    c2, c3, c4 = coefficient_bounds(command_rows)
    vectors = nonlinear_vectors(command_rows)
    values, directions = np.linalg.eigh(p_matrix)
    sqrt_p = directions @ np.diag(np.sqrt(values)) @ directions.T
    metric2 = sum(np.linalg.norm(sqrt_p @ value) for key, value in vectors.items() if sum(key) == 2)
    metric3 = sum(np.linalg.norm(sqrt_p @ value) for key, value in vectors.items() if sum(key) == 3)
    metric4 = sum(np.linalg.norm(sqrt_p @ value) for key, value in vectors.items() if sum(key) == 4)
    direct_radius = bound_root(
        slack,
        np.linalg.norm(p_matrix, 2) * c2,
        np.linalg.norm(p_matrix, 2) * c3,
        np.linalg.norm(p_matrix, 2) * c4,
    )
    metric_radius = bound_root(
        slack,
        math.sqrt(float(np.max(values))) * metric2,
        math.sqrt(float(np.max(values))) * metric3,
        math.sqrt(float(np.max(values))) * metric4,
    )
    maximum_level, binding = constraint_level(p_matrix, command_rows)
    return {
        "fiber": name,
        "linear_slack": slack,
        "euclidean_coefficient_sums": [c2, c3, c4],
        "certificate_metric_coefficient_sums": [metric2, metric3, metric4],
        "direct_radius": direct_radius,
        "metric_radius": metric_radius,
        "direct_level": min(float(np.min(values)) * direct_radius**2, maximum_level),
        "metric_level": min(float(np.min(values)) * metric_radius**2, maximum_level),
        "constraint_level": maximum_level,
        "binding_constraint": binding,
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--nu-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    cp = import_cvxpy()
    solver = select_solver(cp, args.solver)
    design = synthesize()
    result = codesign(design, solver, args.nu_tolerance, args.verbose)
    command_p = np.column_stack((design["Kp"], -design["Kp"]))
    command_pi = np.column_stack((design["Kp"], -design["Kp"], design["Ki"]))
    records = [
        remainder_record(design["AP"], command_p, result["matrices"][0], "P"),
        remainder_record(design["API"], command_pi, result["matrices"][1], "PI"),
    ]
    installed_p_level = min(
        records[0]["metric_level"],
        P_TO_PI_HEADROOM * records[1]["metric_level"] / result["transition_factor"],
    )
    output = {
        "scope": "corrected 12-state P and 14-state PI MFD fibers",
        "quadratic_codesign": result,
        "remainder_certificates": records,
        "installed_P_metric_level": installed_p_level,
        "transition_headroom": P_TO_PI_HEADROOM,
        "interpretation": (
            "The trace constraints remove pure matrix-rescaling ambiguity. "
            "The optimization reduces the P-to-PI comparison factor subject to both flow inequalities; "
            "usable nonlinear levels are evaluated afterward and remain part of the design decision."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mfd_architecture_synthesis_results.json").write_text(
        json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            json_ready(
                {
                    "solver": solver,
                    "baseline_factor": result["baseline_independent_factor"],
                    "codesigned_factor": result["transition_factor"],
                    "architecture_dwell_time": result["architecture_dwell_time"],
                    "remainder_certificates": records,
                    "installed_P_metric_level": installed_p_level,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
