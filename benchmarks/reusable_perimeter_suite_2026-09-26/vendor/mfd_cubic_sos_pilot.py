#!/usr/bin/env python3
"""Degree-six SOS pilot for the literature-cubic MFD controller pair.

The cubic regional production curve and affine state feedback produce a
quartic closed-loop vector field.  Since a quadratic certificate has a linear
gradient, ``-(dot(V)+lambda V)`` has degree five.  The local implication

    V(e) <= ell  =>  -(dot(V)+lambda V) >= 0

is certified with a degree-four SOS multiplier and a degree-six SOS target:

    -(dot(V)+lambda V) - sigma(e)(ell-V(e)) = s(e).

This is a sufficient fixed-degree certificate.  A positive constant margin is
reserved before solving.  The returned Gram matrices are then checked with an
a-posteriori bound that accounts for their smallest eigenvalues and the full
coefficient-residual one-norm on the unit ball.  Random sampling is retained
only as a separate falsification attempt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.linalg import eigvalsh

import mfd_cubic_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_sos_pilot_results.json"
DIMENSION = pilot.DIMENSION
ZERO = pilot.ZERO_EXPONENT
Exponent = pilot.Exponent
Polynomial = pilot.Polynomial


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def select_solver(requested: str) -> str:
    installed = set(cp.installed_solvers())
    if requested != "auto":
        if requested not in installed:
            raise SystemExit(f"Solver {requested!r} is unavailable; installed: {sorted(installed)}")
        return requested
    # Prefer the solver used for the archived pilot result.  A locally
    # installed MOSEK package may still lack a usable license, whereas
    # CLARABEL is both available in the checked environment and reproduces
    # the coordinate-scaled SDP result without external credentials.
    for candidate in ("CLARABEL", "MOSEK", "SCS"):
        if candidate in installed:
            return candidate
    raise SystemExit("No semidefinite-capable solver is installed")


def solve_problem(problem: cp.Problem, solver: str, verbose: bool) -> None:
    options: dict[str, Any] = {"solver": solver, "verbose": verbose}
    if solver == "SCS":
        options.update({"eps": 2.0e-7, "max_iters": 400_000, "acceleration_lookback": 20})
    problem.solve(**options)


def pclean(polynomial: Polynomial, tolerance: float = 1.0e-12) -> Polynomial:
    return {key: float(value) for key, value in polynomial.items() if abs(value) > tolerance}


def padd(left: Polynomial, right: Polynomial) -> Polynomial:
    result = left.copy()
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, 0.0) + value
    return pclean(result)


def pscale(polynomial: Polynomial, scale: float) -> Polynomial:
    return pclean({key: scale * value for key, value in polynomial.items()})


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


def linear_substitution(polynomial: Polynomial, matrix: np.ndarray) -> Polynomial:
    """Return p(My) for a polynomial p(e) and dense linear map e=My."""
    linear_forms = [linear_polynomial(matrix[row]) for row in range(DIMENSION)]
    result: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        term: Polynomial = {ZERO: float(coefficient)}
        for coordinate, power in enumerate(exponent):
            for _ in range(power):
                term = pmul(term, linear_forms[coordinate])
        result = padd(result, term)
    return result


def dissipation_polynomial(gain: np.ndarray, p_matrix: np.ndarray, rate: float) -> Polynomial:
    flow = pilot.closed_loop_polynomials(gain)
    derivative: Polynomial = {}
    for output in range(DIMENSION):
        derivative = padd(
            derivative,
            pmul(linear_polynomial(2.0 * p_matrix[output]), flow[output]),
        )
    value = quadratic_polynomial(p_matrix)
    result = pscale(padd(derivative, pscale(value, rate)), -1.0)
    if max(map(sum, result)) != 5:
        raise AssertionError("The cubic-MFD quadratic-certificate dissipation must have degree five")
    return result


def monomial_basis(max_degree: int, min_degree: int = 0) -> list[Exponent]:
    return [
        exponent
        for exponent in itertools.product(range(max_degree + 1), repeat=DIMENSION)
        if min_degree <= sum(exponent) <= max_degree
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


def evaluate_polynomial(polynomial: Polynomial, point: np.ndarray) -> float:
    return float(
        sum(value * np.prod(point ** np.asarray(exponent)) for exponent, value in polynomial.items())
    )


def sos_at_level(
    gain: np.ndarray,
    p_matrix: np.ndarray,
    rate: float,
    level: float,
    solver: str,
    proof_margin: float,
    verbose: bool,
) -> dict[str, Any]:
    # Both the dissipation and its reserved margin vanish quadratically at the
    # equilibrium.  Exact SOS decompositions must therefore have sigma(0)=0
    # and target(0)=0; omitting the constant monomial enforces that structure
    # numerically and makes the residual bound scale with ||y||^2.
    multiplier_basis = monomial_basis(2, min_degree=1)  # degree-four sigma
    target_basis = monomial_basis(3, min_degree=1)  # degree-six target
    q_multiplier = cp.Variable((len(multiplier_basis), len(multiplier_basis)), symmetric=True)
    q_target = cp.Variable((len(target_basis), len(target_basis)), symmetric=True)
    sigma = gram_coefficients(q_multiplier, multiplier_basis)
    target = gram_coefficients(q_target, target_basis)

    # Work on the unit ball y'y<=1 with e=sqrt(level)P^{-1/2}y.  Dividing
    # dissipation by level leaves the sign unchanged and avoids the severe
    # coefficient scaling present in the raw ellipsoid coordinates.
    values, vectors = np.linalg.eigh(p_matrix)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    coordinate_map = math.sqrt(level) * inverse_sqrt
    original_dissipation = dissipation_polynomial(gain, p_matrix, rate)
    dissipation = pscale(
        linear_substitution(original_dissipation, coordinate_map),
        1.0 / level,
    )
    unit_value = quadratic_polynomial(np.eye(DIMENSION))
    domain = padd({ZERO: 1.0}, pscale(unit_value, -1.0))
    product = expression_product(sigma, domain)
    dissipation_with_margin = padd(
        dissipation,
        pscale(unit_value, -proof_margin),
    )
    exponents = set(dissipation_with_margin) | set(product) | set(target)
    constraints = [q_multiplier >> 0.0, q_target >> 0.0]
    for exponent in exponents:
        constraints.append(
            dissipation_with_margin.get(exponent, 0.0)
            - product.get(exponent, 0.0)
            == target.get(exponent, 0.0)
        )
    objective = cp.Minimize(cp.trace(q_multiplier) + 1.0e-9 * cp.trace(q_target))
    problem = cp.Problem(objective, constraints)
    solve_started = time.perf_counter()
    solve_problem(problem, solver, verbose)
    wall_seconds = time.perf_counter() - solve_started
    solver_stats = problem.solver_stats
    timing = {
        "wall_seconds_including_canonicalization": wall_seconds,
        "reported_solver_seconds": (
            None if solver_stats.solve_time is None else float(solver_stats.solve_time)
        ),
        "reported_setup_seconds": (
            None if solver_stats.setup_time is None else float(solver_stats.setup_time)
        ),
        "reported_iterations": (
            None if solver_stats.num_iters is None else int(solver_stats.num_iters)
        ),
    }
    acceptable = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if not acceptable or q_multiplier.value is None or q_target.value is None:
        return {
            "feasible": False,
            "status": problem.status,
            "level": level,
            "timing": timing,
        }

    multiplier_value = symmetrize(np.asarray(q_multiplier.value, dtype=float))
    target_value = symmetrize(np.asarray(q_target.value, dtype=float))
    sigma_numeric = gram_coefficients(multiplier_value, multiplier_basis)
    target_numeric = gram_coefficients(target_value, target_basis)
    product_numeric = expression_product(sigma_numeric, domain)
    residuals = [
        float(
            dissipation_with_margin.get(exponent, 0.0)
            - product_numeric.get(exponent, 0.0)
            - target_numeric.get(exponent, 0.0)
        )
        for exponent in exponents
    ]
    multiplier_minimum_eigenvalue = float(np.min(eigvalsh(multiplier_value)))
    target_minimum_eigenvalue = float(np.min(eigvalsh(target_value)))
    minimum_gram_eigenvalue = min(
        multiplier_minimum_eigenvalue,
        target_minimum_eigenvalue,
    )
    maximum_coefficient_residual = max(abs(value) for value in residuals)
    coefficient_residual_one_norm = float(sum(abs(value) for value in residuals))
    # Every retained basis monomial has positive degree.  On ||y||_2 <= 1,
    # its square and every residual monomial (degree at least two) are bounded
    # by ||y||_2^2, while 0 <= 1-y'y <= 1.  Thus the expression below is a
    # verified coefficient multiplying ||y||_2^2.
    verified_normalized_lower_bound = (
        proof_margin
        + min(0.0, multiplier_minimum_eigenvalue) * len(multiplier_basis)
        + min(0.0, target_minimum_eigenvalue) * len(target_basis)
        - coefficient_residual_one_norm
    )
    feasible = verified_normalized_lower_bound > 0.0
    return {
        "feasible": feasible,
        "status": problem.status,
        "level": level,
        "minimum_gram_eigenvalue": minimum_gram_eigenvalue,
        "multiplier_minimum_eigenvalue": multiplier_minimum_eigenvalue,
        "target_minimum_eigenvalue": target_minimum_eigenvalue,
        "maximum_coefficient_residual": maximum_coefficient_residual,
        "coefficient_residual_one_norm": coefficient_residual_one_norm,
        "reserved_normalized_proof_margin": proof_margin,
        "verified_normalized_lower_bound": verified_normalized_lower_bound,
        "a_posteriori_residual_bound_passed": verified_normalized_lower_bound > 0.0,
        "multiplier_basis_size": len(multiplier_basis),
        "target_basis_size": len(target_basis),
        "timing": timing,
    }


def random_falsification(
    gain: np.ndarray,
    p_matrix: np.ndarray,
    rate: float,
    level: float,
    seed: int,
    samples: int = 20_000,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values, vectors = np.linalg.eigh(p_matrix)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    directions = rng.normal(size=(samples, DIMENSION))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.random(samples) ** (1.0 / DIMENSION)
    points = math.sqrt(level) * (directions * radii[:, None]) @ inverse_sqrt.T
    dissipation = dissipation_polynomial(gain, p_matrix, rate)
    evaluations = np.asarray([evaluate_polynomial(dissipation, point) for point in points])
    return {
        "sample_count": samples,
        "minimum_minus_dotV_minus_lambdaV": float(np.min(evaluations)),
        "violations_below_minus_1e-8": int(np.sum(evaluations < -1.0e-8)),
    }


def search_mode(
    mode: int,
    gain: np.ndarray,
    p_matrix: np.ndarray,
    rate: float,
    analytic_level: float,
    constraint_level: float,
    binding_constraint: str,
    solver: str,
    proof_margin: float,
    relative_tolerance: float,
    verbose: bool,
) -> dict[str, Any]:
    initial = sos_at_level(
        gain, p_matrix, rate, analytic_level, solver, proof_margin, verbose
    )
    if not initial["feasible"]:
        raise RuntimeError(
            f"Mode {mode + 1} failed SOS verification even at the analytic level: {initial}"
        )
    boundary = sos_at_level(
        gain, p_matrix, rate, constraint_level, solver, proof_margin, verbose
    )
    if boundary["feasible"]:
        best = boundary
        history = [initial, boundary]
        lower = constraint_level
        upper = constraint_level
    else:
        best = initial
        history = [initial, boundary]
        lower = analytic_level
        upper = constraint_level
    for _ in range(40):
        if (upper - lower) / upper <= relative_tolerance:
            break
        midpoint = 0.5 * (lower + upper)
        trial = sos_at_level(
            gain, p_matrix, rate, midpoint, solver, proof_margin, verbose
        )
        history.append(trial)
        if trial["feasible"]:
            lower = midpoint
            best = trial
        else:
            upper = midpoint
    level = float(best["level"])
    reported_solver_times = [
        item["timing"]["reported_solver_seconds"]
        for item in history
        if item["timing"]["reported_solver_seconds"] is not None
    ]
    return {
        **best,
        "mode": mode + 1,
        "analytic_level": analytic_level,
        "constraint_level": constraint_level,
        "binding_constraint": binding_constraint,
        "relative_bisection_tolerance": relative_tolerance,
        "inscribed_euclidean_radius_units": math.sqrt(level / float(np.max(eigvalsh(p_matrix)))),
        "inscribed_euclidean_radius_vehicles": (
            pilot.ACCUMULATION_UNIT_VEHICLES
            * math.sqrt(level / float(np.max(eigvalsh(p_matrix))))
        ),
        "history": history,
        "sdp_solve_count": len(history),
        "total_sdp_wall_seconds": float(
            sum(item["timing"]["wall_seconds_including_canonicalization"] for item in history)
        ),
        "total_reported_solver_seconds": float(sum(reported_solver_times)),
        "maximum_single_sdp_wall_seconds": float(
            max(item["timing"]["wall_seconds_including_canonicalization"] for item in history)
        ),
        "random_falsification": random_falsification(
            gain, p_matrix, rate, level, seed=20260829 + mode
        ),
    }


def json_ready(value: Any) -> Any:
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
    parser.add_argument("--relative-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--proof-margin", type=float, default=1.0e-4)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    solver = select_solver(args.solver)
    if not PILOT_JSON.exists():
        raise SystemExit("Run reproducibility/mfd_cubic_pilot.py first")
    pilot_record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    gains = tuple(np.asarray(value, dtype=float) for value in pilot_record["controller_search"]["selected_gains"])
    p_matrices = tuple(np.asarray(value, dtype=float) for value in pilot_record["certificate"]["matrices"])
    rate = float(pilot_record["certificate"]["rate"])
    tube_modes = pilot_record["tube"]["modes"]

    records = []
    for mode in range(2):
        constraint_level, binding, _ = pilot.exact_constraint_level(p_matrices[mode], gains[mode])
        records.append(
            search_mode(
                mode=mode,
                gain=gains[mode],
                p_matrix=p_matrices[mode],
                rate=rate,
                analytic_level=float(tube_modes[mode]["flow_level"]),
                constraint_level=constraint_level,
                binding_constraint=binding,
                solver=solver,
                proof_margin=args.proof_margin,
                relative_tolerance=args.relative_tolerance,
                verbose=args.verbose,
            )
        )
    output = {
        "scope": "literature-cubic MFD fixed-gain pilot",
        "solver": solver,
        "certificate_rate_per_hour": rate,
        "multiplier_degree": 4,
        "target_sos_degree": 6,
        "reserved_normalized_proof_margin": args.proof_margin,
        "sos_certificates": records,
        "common_level": min(float(record["level"]) for record in records),
        "offline_computation": {
            "sdp_solve_count": int(sum(record["sdp_solve_count"] for record in records)),
            "total_sdp_wall_seconds": float(
                sum(record["total_sdp_wall_seconds"] for record in records)
            ),
            "total_reported_solver_seconds": float(
                sum(record["total_reported_solver_seconds"] for record in records)
            ),
            "includes": (
                "two initial feasibility checks, two constraint-boundary checks, "
                "and all level-bisection SDPs; random falsification time excluded"
            ),
        },
        "interpretation": (
            "Fixed-degree SOS records reserve a positive normalized margin and pass "
            "an a-posteriori lower bound covering Gram eigenvalue defects and the "
            "full coefficient-residual one-norm. Random sampling is retained only "
            "as an independent falsification attempt."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")
    summary = {
        "solver": solver,
        "rate": rate,
        "common_level": output["common_level"],
        "offline_computation": output["offline_computation"],
        "modes": [
            {
                key: record[key]
                for key in (
                    "mode",
                    "level",
                    "analytic_level",
                    "constraint_level",
                    "binding_constraint",
                    "inscribed_euclidean_radius_vehicles",
                    "minimum_gram_eigenvalue",
                    "maximum_coefficient_residual",
                    "coefficient_residual_one_norm",
                    "reserved_normalized_proof_margin",
                    "verified_normalized_lower_bound",
                    "a_posteriori_residual_bound_passed",
                    "random_falsification",
                )
            }
            for record in records
        ],
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
