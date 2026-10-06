#!/usr/bin/env python3
"""Synthesize less-conservative certificates for the corrected MFD model.

This script performs three auditable steps on the *current* six-state model:

1. quasiconvex co-design of the two quadratic matrices by bisection on the
   identity-transition factor;
2. the coefficient-sum and certificate-metric nonlinear remainder bounds;
3. an SOS/Positivstellensatz search for the largest certified level inside all
   installed state and input constraints.

The SOS implication is sufficient at the selected multiplier degree.  It is
not described as a necessary fixed-degree characterization.  Every returned
SDP is followed by direct eigenvalue, coefficient-matching, and random-point
checks.  MOSEK is preferred for publication records; SCS is supported as a
reproducibility fallback but should not be used as the sole numerical evidence
without checking the reported residuals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import brentq

from mfd_fixed_gain_validation import (
    CLOSED_LOOP,
    DIRECTIONAL_LIMIT,
    GAINS,
    PROPOSAL_INTERVAL,
    X_STAR,
    closed_loop_polynomials,
    independent_certificate,
    nonlinear_coefficient_bounds,
    state_constraint_rows,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
DIMENSION = 6
ZERO = (0,) * DIMENSION
DEFAULT_RATE = 0.20

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, float]


def import_cvxpy():
    try:
        import cvxpy as cp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "cvxpy is required for synthesis. Install the synthesis extras with "
            "`python -m pip install -r requirements-synthesis.txt`."
        ) from exc
    return cp


def select_solver(cp, requested: str) -> str:
    installed = set(cp.installed_solvers())
    if requested != "auto":
        if requested not in installed:
            raise SystemExit(
                f"Requested solver {requested!r} is not installed; available: {sorted(installed)}"
            )
        return requested
    for candidate in ("MOSEK", "CLARABEL", "CVXOPT", "SCS"):
        if candidate in installed:
            return candidate
    raise SystemExit(
        "No semidefinite-capable CVXPY solver was found. Install MOSEK, "
        "Clarabel, CVXOPT, or SCS."
    )


def solve_problem(problem, solver: str, verbose: bool = False) -> None:
    options: dict[str, Any] = {"solver": solver, "verbose": verbose}
    if solver == "SCS":
        options.update({"eps": 2.0e-7, "max_iters": 300_000, "acceleration_lookback": 20})
    problem.solve(**options)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def generalized_max(successor: np.ndarray, predecessor: np.ndarray) -> float:
    return float(np.max(eigvalsh(successor, predecessor)))


def codesign_for_factor(cp, factor: float, rate: float, solver: str, verbose: bool) -> dict:
    matrices = [cp.Variable((DIMENSION, DIMENSION), symmetric=True) for _ in range(2)]
    slack = cp.Variable()
    identity = np.eye(DIMENSION)
    constraints = [
        matrices[0] >> 1.0e-7 * identity,
        matrices[1] >> 1.0e-7 * identity,
        cp.trace(matrices[0]) == 1.0,
        matrices[1] << factor * matrices[0],
        matrices[0] << factor * matrices[1],
        slack >= 0.0,
    ]
    for matrix, closed_loop in zip(matrices, CLOSED_LOOP):
        residual = closed_loop.T @ matrix + matrix @ closed_loop + rate * matrix
        constraints.append(residual << -slack * identity)
    problem = cp.Problem(cp.Maximize(slack), constraints)
    solve_problem(problem, solver, verbose)
    acceptable = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if not acceptable or slack.value is None or any(matrix.value is None for matrix in matrices):
        return {"feasible": False, "status": problem.status}
    values = [symmetrize(np.asarray(matrix.value, dtype=float)) for matrix in matrices]
    actual_slacks = [
        -float(np.max(eigvalsh(a.T @ p + p @ a + rate * p)))
        for a, p in zip(CLOSED_LOOP, values)
    ]
    actual_factor = max(
        generalized_max(values[1], values[0]),
        generalized_max(values[0], values[1]),
    )
    minimum_eigenvalue = min(float(np.min(eigvalsh(value))) for value in values)
    feasible = (
        minimum_eigenvalue >= -2.0e-7
        and min(actual_slacks) > 2.0e-8
        and actual_factor <= factor + max(2.0e-6, 2.0e-6 * factor)
    )
    return {
        "feasible": feasible,
        "status": problem.status,
        "objective_slack": float(slack.value),
        "actual_flow_slacks": actual_slacks,
        "actual_transition_factor": actual_factor,
        "minimum_matrix_eigenvalue": minimum_eigenvalue,
        "matrices": values,
    }


def codesign(rate: float, solver: str, tolerance: float, verbose: bool) -> dict:
    baseline = independent_certificate(rate)
    lower = 1.0
    upper = float(baseline["transition_factor"])
    best = codesign_for_factor(cp=import_cvxpy(), factor=upper, rate=rate, solver=solver, verbose=verbose)
    if not best["feasible"]:
        raise RuntimeError("The baseline transition factor was unexpectedly infeasible in co-design")
    cp = import_cvxpy()
    history = []
    while upper - lower > tolerance:
        midpoint = 0.5 * (lower + upper)
        trial = codesign_for_factor(cp, midpoint, rate, solver, verbose)
        history.append({key: value for key, value in trial.items() if key != "matrices"} | {"factor_tested": midpoint})
        if trial["feasible"]:
            upper = midpoint
            best = trial
        else:
            lower = midpoint
    matrices = best.pop("matrices")
    factor = max(
        generalized_max(matrices[1], matrices[0]),
        generalized_max(matrices[0], matrices[1]),
    )
    dwell = math.log(factor) / rate
    realized = math.ceil(dwell / PROPOSAL_INTERVAL - 1.0e-12) * PROPOSAL_INTERVAL
    return {
        "rate": rate,
        "solver": solver,
        "normalization": "trace(P1)=1",
        "bisection_tolerance": tolerance,
        "transition_factor": factor,
        "dwell_time": dwell,
        "realized_dwell": realized,
        "sampled_contraction": factor * math.exp(-rate * realized),
        "flow_slacks": [
            -float(np.max(eigvalsh(a.T @ p + p @ a + rate * p)))
            for a, p in zip(CLOSED_LOOP, matrices)
        ],
        "matrices": matrices,
        "final_solver_record": best,
        "bisection_history": history,
    }


def remainder_vectors(mode: int) -> dict[Exponent, np.ndarray]:
    polynomials = closed_loop_polynomials(GAINS[mode])
    vectors: dict[Exponent, np.ndarray] = {}
    for output, polynomial in enumerate(polynomials):
        for exponent, coefficient in polynomial.items():
            if sum(exponent) >= 2:
                vectors.setdefault(exponent, np.zeros(DIMENSION))[output] = coefficient
    return vectors


def exact_constraint_level(p_matrix: np.ndarray, mode: int) -> tuple[float, str]:
    inverse = np.linalg.inv(p_matrix)
    names, rows, margins = state_constraint_rows()
    candidates: list[tuple[float, str]] = []
    for name, row, margin in zip(names, rows, margins):
        candidates.append((float(margin * margin / (row @ inverse @ row)), name))
    for index, row in enumerate(GAINS[mode]):
        candidates.append((float(0.25 / (row @ inverse @ row)), f"u{index + 1} box"))
    for label, row in (
        ("u21-u12 directional", GAINS[mode][1] - GAINS[mode][0]),
        ("u20-u02 directional", GAINS[mode][3] - GAINS[mode][2]),
    ):
        candidates.append((float(DIRECTIONAL_LIMIT**2 / (row @ inverse @ row)), label))
    return min(candidates)


def scalar_root(slack: float, first: float, second: float) -> float:
    return float(
        brentq(
            lambda radius: 2.0 * (first * radius + second * radius * radius) - slack,
            0.0,
            10.0,
        )
    )


def remainder_records(matrices: list[np.ndarray], rate: float) -> list[dict]:
    records = []
    for mode, (p_matrix, closed_loop) in enumerate(zip(matrices, CLOSED_LOOP)):
        residual = closed_loop.T @ p_matrix + p_matrix @ closed_loop + rate * p_matrix
        slack = -float(np.max(eigvalsh(residual)))
        c2, c3 = nonlinear_coefficient_bounds(GAINS[mode], closed_loop)
        vectors = remainder_vectors(mode)
        values, directions = np.linalg.eigh(p_matrix)
        sqrt_p = directions @ np.diag(np.sqrt(values)) @ directions.T
        c2_metric = sum(
            np.linalg.norm(sqrt_p @ vector)
            for exponent, vector in vectors.items()
            if sum(exponent) == 2
        )
        c3_metric = sum(
            np.linalg.norm(sqrt_p @ vector)
            for exponent, vector in vectors.items()
            if sum(exponent) == 3
        )
        direct_radius = scalar_root(
            slack,
            np.linalg.norm(p_matrix, 2) * c2,
            np.linalg.norm(p_matrix, 2) * c3,
        )
        metric_radius = scalar_root(
            slack,
            math.sqrt(float(np.max(values))) * c2_metric,
            math.sqrt(float(np.max(values))) * c3_metric,
        )
        constraint_level, binding = exact_constraint_level(p_matrix, mode)
        minimum_eigenvalue = float(np.min(values))
        records.append(
            {
                "mode": mode + 1,
                "linear_slack": slack,
                "euclidean_coefficient_sums": [c2, c3],
                "certificate_metric_coefficient_sums": [c2_metric, c3_metric],
                "direct_radius": direct_radius,
                "metric_radius": metric_radius,
                "direct_level": min(minimum_eigenvalue * direct_radius**2, constraint_level),
                "metric_level": min(minimum_eigenvalue * metric_radius**2, constraint_level),
                "constraint_level": constraint_level,
                "binding_constraint": binding,
            }
        )
    return records


def pclean(poly: Polynomial, tolerance: float = 1.0e-13) -> Polynomial:
    return {key: float(value) for key, value in poly.items() if abs(value) > tolerance}


def padd(left: Polynomial, right: Polynomial) -> Polynomial:
    result = left.copy()
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, 0.0) + value
    return pclean(result)


def pscale(poly: Polynomial, scale: float) -> Polynomial:
    return pclean({key: scale * value for key, value in poly.items()})


def pmul(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exp, left_value in left.items():
        for right_exp, right_value in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            result[exponent] = result.get(exponent, 0.0) + left_value * right_value
    return pclean(result)


def quadratic_polynomial(matrix: np.ndarray) -> Polynomial:
    result: Polynomial = {}
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            exponent = [0] * DIMENSION
            exponent[row] += 1
            exponent[column] += 1
            key = tuple(exponent)
            result[key] = result.get(key, 0.0) + float(matrix[row, column])
    return pclean(result)


def linear_polynomial(row: np.ndarray) -> Polynomial:
    result: Polynomial = {}
    for index, value in enumerate(row):
        if abs(value) > 1.0e-14:
            exponent = [0] * DIMENSION
            exponent[index] = 1
            result[tuple(exponent)] = float(value)
    return result


def dissipation_polynomial(mode: int, p_matrix: np.ndarray, rate: float) -> Polynomial:
    flow = closed_loop_polynomials(GAINS[mode])
    derivative: Polynomial = {}
    for output in range(DIMENSION):
        derivative = padd(derivative, pmul(linear_polynomial(2.0 * p_matrix[output]), flow[output]))
    value = quadratic_polynomial(p_matrix)
    return pscale(padd(derivative, pscale(value, rate)), -1.0)


def monomial_basis(max_degree: int) -> list[Exponent]:
    return [
        exponent
        for exponent in itertools.product(range(max_degree + 1), repeat=DIMENSION)
        if sum(exponent) <= max_degree
    ]


def gram_coefficients(matrix, basis: list[Exponent]) -> dict[Exponent, Any]:
    coefficients: dict[Exponent, Any] = {}
    for row, left in enumerate(basis):
        for column, right in enumerate(basis):
            exponent = tuple(a + b for a, b in zip(left, right))
            coefficients[exponent] = coefficients.get(exponent, 0.0) + matrix[row, column]
    return coefficients


def expression_product(left: dict[Exponent, Any], right: Polynomial) -> dict[Exponent, Any]:
    result: dict[Exponent, Any] = {}
    for left_exp, left_value in left.items():
        for right_exp, right_value in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            result[exponent] = result.get(exponent, 0.0) + left_value * right_value
    return result


def sos_at_level(cp, mode: int, p_matrix: np.ndarray, rate: float, level: float, solver: str, verbose: bool) -> dict:
    multiplier_basis = monomial_basis(1)
    target_basis = monomial_basis(2)
    q_multiplier = cp.Variable((len(multiplier_basis), len(multiplier_basis)), symmetric=True)
    q_target = cp.Variable((len(target_basis), len(target_basis)), symmetric=True)
    sigma = gram_coefficients(q_multiplier, multiplier_basis)
    target = gram_coefficients(q_target, target_basis)
    value = quadratic_polynomial(p_matrix)
    domain = {ZERO: level}
    domain = padd(domain, pscale(value, -1.0))
    product = expression_product(sigma, domain)
    dissipation = dissipation_polynomial(mode, p_matrix, rate)
    exponents = set(dissipation) | set(product) | set(target)
    constraints = [q_multiplier >> 0.0, q_target >> 0.0]
    for exponent in exponents:
        constraints.append(
            dissipation.get(exponent, 0.0)
            - product.get(exponent, 0.0)
            == target.get(exponent, 0.0)
        )
    problem = cp.Problem(cp.Minimize(cp.trace(q_multiplier) + 1.0e-8 * cp.trace(q_target)), constraints)
    solve_problem(problem, solver, verbose)
    acceptable = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if not acceptable or q_multiplier.value is None or q_target.value is None:
        return {"feasible": False, "status": problem.status, "level": level}
    multiplier_value = symmetrize(np.asarray(q_multiplier.value, dtype=float))
    target_value = symmetrize(np.asarray(q_target.value, dtype=float))
    sigma_numeric = gram_coefficients(multiplier_value, multiplier_basis)
    target_numeric = gram_coefficients(target_value, target_basis)
    product_numeric = expression_product(sigma_numeric, domain)
    residuals = [
        float(dissipation.get(exponent, 0.0) - product_numeric.get(exponent, 0.0) - target_numeric.get(exponent, 0.0))
        for exponent in exponents
    ]
    minimum_gram_eigenvalue = min(
        float(np.min(eigvalsh(multiplier_value))),
        float(np.min(eigvalsh(target_value))),
    )
    maximum_coefficient_residual = max(abs(value) for value in residuals)
    feasible = minimum_gram_eigenvalue >= -2.0e-6 and maximum_coefficient_residual <= 2.0e-6
    return {
        "feasible": feasible,
        "status": problem.status,
        "level": level,
        "minimum_gram_eigenvalue": minimum_gram_eigenvalue,
        "maximum_coefficient_residual": maximum_coefficient_residual,
        "multiplier_gram": multiplier_value,
        "target_gram": target_value,
    }


def evaluate_polynomial(poly: Polynomial, point: np.ndarray) -> float:
    return float(
        sum(value * np.prod(point ** np.asarray(exponent)) for exponent, value in poly.items())
    )


def random_falsification(mode: int, p_matrix: np.ndarray, rate: float, level: float, samples: int = 20_000) -> dict:
    rng = np.random.default_rng(20260812 + mode)
    values, vectors = np.linalg.eigh(p_matrix)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    directions = rng.normal(size=(samples, DIMENSION))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.random(samples) ** (1.0 / DIMENSION)
    points = math.sqrt(level) * (directions * radii[:, None]) @ inverse_sqrt.T
    dissipation = dissipation_polynomial(mode, p_matrix, rate)
    evaluations = np.array([evaluate_polynomial(dissipation, point) for point in points])
    return {
        "sample_count": samples,
        "minimum_minus_dotV_minus_lambdaV": float(np.min(evaluations)),
        "violations_below_minus_1e-8": int(np.sum(evaluations < -1.0e-8)),
    }


def sos_search(cp, mode: int, p_matrix: np.ndarray, rate: float, solver: str, relative_tolerance: float, verbose: bool) -> dict:
    constraint_level, binding = exact_constraint_level(p_matrix, mode)
    lower = 0.0
    upper = constraint_level
    best = None
    history = []
    for _ in range(60):
        if upper <= 0.0 or (upper - lower) / upper <= relative_tolerance:
            break
        midpoint = 0.5 * (lower + upper)
        trial = sos_at_level(cp, mode, p_matrix, rate, midpoint, solver, verbose)
        history.append(
            {
                key: value
                for key, value in trial.items()
                if key not in {"multiplier_gram", "target_gram"}
            }
        )
        if trial["feasible"]:
            lower = midpoint
            best = trial
        else:
            upper = midpoint
    if best is None:
        raise RuntimeError(f"No positive SOS level was certified for mode {mode + 1}")
    result = {
        key: value
        for key, value in best.items()
        if key not in {"multiplier_gram", "target_gram"}
    }
    result.update(
        {
            "mode": mode + 1,
            "constraint_level": constraint_level,
            "binding_constraint": binding,
            "relative_bisection_tolerance": relative_tolerance,
            "history": history,
            "random_falsification": random_falsification(mode, p_matrix, rate, float(best["level"])),
            "multiplier_gram": best["multiplier_gram"],
            "target_gram": best["target_gram"],
        }
    )
    return result


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


def write_tex_macros(output: dict) -> None:
    codesign_record = output["quadratic_codesign"]
    remainders = output["remainder_certificates"]
    sos = output["sos_certificates"]
    direct_common = min(record["direct_level"] for record in remainders)
    metric_common = min(record["metric_level"] for record in remainders)
    sos_common = min(record["level"] for record in sos)
    contents = "\n".join(
        [
            "% Generated by reproducibility/mfd_certificate_synthesis.py",
            f"\\newcommand{{\\MFDCodesignNu}}{{{codesign_record['transition_factor']:.6g}}}",
            f"\\newcommand{{\\MFDCodesignDwell}}{{{codesign_record['dwell_time']:.6g}}}",
            f"\\newcommand{{\\MFDDirectLevel}}{{{direct_common:.6g}}}",
            f"\\newcommand{{\\MFDMetricLevel}}{{{metric_common:.6g}}}",
            f"\\newcommand{{\\MFDSOSLevel}}{{{sos_common:.6g}}}",
            f"\\newcommand{{\\MFDSOSLevelOne}}{{{sos[0]['level']:.6g}}}",
            f"\\newcommand{{\\MFDSOSLevelTwo}}{{{sos[1]['level']:.6g}}}",
            f"\\newcommand{{\\MFDSOSSolver}}{{{output['solver']}}}",
            f"\\newcommand{{\\MFDMetricRadiusOne}}{{{remainders[0]['metric_radius']:.6g}}}",
            f"\\newcommand{{\\MFDMetricRadiusTwo}}{{{remainders[1]['metric_radius']:.6g}}}",
            f"\\newcommand{{\\MFDSOSRadiusOne}}{{{math.sqrt(sos[0]['level'] / np.max(eigvalsh(codesign_record['matrices'][0]))):.6g}}}",
            f"\\newcommand{{\\MFDSOSRadiusTwo}}{{{math.sqrt(sos[1]['level'] / np.max(eigvalsh(codesign_record['matrices'][1]))):.6g}}}",
            f"\\newcommand{{\\MFDSOSResidualOne}}{{{sos[0]['maximum_coefficient_residual']:.3e}}}",
            f"\\newcommand{{\\MFDSOSResidualTwo}}{{{sos[1]['maximum_coefficient_residual']:.3e}}}",
            f"\\newcommand{{\\MFDSOSGramOne}}{{{sos[0]['minimum_gram_eigenvalue']:.3e}}}",
            f"\\newcommand{{\\MFDSOSGramTwo}}{{{sos[1]['minimum_gram_eigenvalue']:.3e}}}",
            f"\\newcommand{{\\MFDSOSFalsificationOne}}{{{sos[0]['random_falsification']['minimum_minus_dotV_minus_lambdaV']:.3e}}}",
            f"\\newcommand{{\\MFDSOSFalsificationTwo}}{{{sos[1]['random_falsification']['minimum_minus_dotV_minus_lambdaV']:.3e}}}",
            "",
        ]
    )
    (RESULTS_DIR / "generated_mfd_certificate.tex").write_text(contents, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto", help="CVXPY solver name or auto")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--nu-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--sos-relative-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--skip-codesign", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cp = import_cvxpy()
    solver = select_solver(cp, args.solver)
    if args.skip_codesign:
        baseline = independent_certificate(args.rate)
        quadratic = {
            key: value for key, value in baseline.items() if key != "matrices"
        }
        quadratic.update(
            {
                "solver": "analytic shifted Lyapunov",
                "normalization": "independent records with optimal scalar balancing",
                "flow_slacks": [
                    -float(np.max(eigvalsh(a.T @ p + p @ a + args.rate * p)))
                    for a, p in zip(CLOSED_LOOP, baseline["matrices"])
                ],
                "matrices": baseline["matrices"],
            }
        )
    else:
        quadratic = codesign(args.rate, solver, args.nu_tolerance, args.verbose)
    matrices = quadratic["matrices"]
    remainder = remainder_records(matrices, args.rate)
    sos = [
        sos_search(
            cp,
            mode,
            matrices[mode],
            args.rate,
            solver,
            args.sos_relative_tolerance,
            args.verbose,
        )
        for mode in range(2)
    ]
    output = {
        "scope": "corrected six-state polynomial MFD fixed-gain library",
        "solver": solver,
        "quadratic_codesign": quadratic,
        "remainder_certificates": remainder,
        "sos_certificates": sos,
        "important_interpretation": (
            "The fixed-degree SOS records are sufficient certificates. "
            "The final publication record must retain solver residuals and direct falsification checks."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "mfd_certificate_synthesis_results.json"
    path.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")
    write_tex_macros(output)
    summary = {
        "solver": solver,
        "transition_factor": quadratic["transition_factor"],
        "dwell_time": quadratic["dwell_time"],
        "remainder_certificates": remainder,
        "sos_levels": [record["level"] for record in sos],
        "sos_residuals": [
            {
                "minimum_gram_eigenvalue": record["minimum_gram_eigenvalue"],
                "maximum_coefficient_residual": record["maximum_coefficient_residual"],
                "random_falsification": record["random_falsification"],
            }
            for record in sos
        ],
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
