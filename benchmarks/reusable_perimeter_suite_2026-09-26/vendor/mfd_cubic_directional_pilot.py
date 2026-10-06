#!/usr/bin/env python3
"""High-resolution directional audit of the cubic-MFD quadratic certificates.

This is a numerical boundary-location and falsification study, not an SOS
proof.  It exploits the exact homogeneous degree-2,...,5 decomposition of
``-(dot(V)+lambda V)``.  In certificate coordinates e=r P^{-1/2}v with
||v||=1, positivity reduces to a cubic polynomial after division by r^2.
Random sphere coverage is followed by local optimization of the most limiting
directions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

import mfd_cubic_pilot as pilot
from mfd_cubic_sos_pilot import dissipation_polynomial


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_directional_pilot_results.json"


def homogeneous_coefficients(
    polynomial: pilot.Polynomial,
    physical_directions: np.ndarray,
) -> np.ndarray:
    coefficients = np.zeros((physical_directions.shape[0], 4))
    for exponent, value in polynomial.items():
        degree = sum(exponent)
        if degree < 2 or degree > 5:
            continue
        term = np.full(physical_directions.shape[0], float(value))
        for coordinate, power in enumerate(exponent):
            if power:
                term *= physical_directions[:, coordinate] ** power
        coefficients[:, degree - 2] += term
    return coefficients


def first_crossing(coefficients: np.ndarray, maximum_radius: float) -> np.ndarray:
    """First sampled/bisected radius at which q(r)=a2+a3r+a4r^2+a5r^3 crosses zero."""
    count = coefficients.shape[0]
    lower = np.zeros(count)
    upper = np.full(count, np.inf)
    grid = np.geomspace(maximum_radius * 1.0e-7, maximum_radius, 240)
    previous = np.zeros(count)
    active = np.ones(count, dtype=bool)
    for radius in grid:
        values = (
            coefficients[:, 0]
            + coefficients[:, 1] * radius
            + coefficients[:, 2] * radius**2
            + coefficients[:, 3] * radius**3
        )
        crossing = active & (values < 0.0)
        lower[crossing] = previous[crossing]
        upper[crossing] = radius
        active[crossing] = False
        previous[active] = radius
    finite = np.isfinite(upper)
    for _ in range(48):
        midpoint = 0.5 * (lower[finite] + upper[finite])
        local = coefficients[finite]
        values = (
            local[:, 0]
            + local[:, 1] * midpoint
            + local[:, 2] * midpoint**2
            + local[:, 3] * midpoint**3
        )
        negative = values < 0.0
        finite_indices = np.flatnonzero(finite)
        upper[finite_indices[negative]] = midpoint[negative]
        lower[finite_indices[~negative]] = midpoint[~negative]
    return upper


def root_for_direction(
    raw_direction: np.ndarray,
    inverse_sqrt: np.ndarray,
    polynomial: pilot.Polynomial,
    maximum_radius: float,
) -> float:
    norm = np.linalg.norm(raw_direction)
    if norm < 1.0e-14:
        return maximum_radius
    unit = raw_direction / norm
    physical = (inverse_sqrt @ unit)[None, :]
    coefficients = homogeneous_coefficients(polynomial, physical)
    root = first_crossing(coefficients, maximum_radius)[0]
    return float(root if np.isfinite(root) else maximum_radius)


def audit_mode(
    mode: int,
    gain: np.ndarray,
    p_matrix: np.ndarray,
    rate: float,
    samples: int,
) -> dict[str, Any]:
    constraint_level, binding, _ = pilot.exact_constraint_level(p_matrix, gain)
    maximum_radius = math.sqrt(constraint_level)
    values, vectors = np.linalg.eigh(p_matrix)
    inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    rng = np.random.default_rng(202608290 + mode)
    directions = rng.normal(size=(samples, pilot.DIMENSION))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    physical_directions = directions @ inverse_sqrt.T
    polynomial = dissipation_polynomial(gain, p_matrix, rate)
    coefficients = homogeneous_coefficients(polynomial, physical_directions)
    roots = first_crossing(coefficients, maximum_radius)
    ranked = np.argsort(roots)[: min(20, samples)]

    refinements = []
    for index in ranked:
        initial = directions[index]
        optimization = minimize(
            lambda direction: root_for_direction(
                direction, inverse_sqrt, polynomial, maximum_radius
            ),
            initial,
            method="Nelder-Mead",
            options={"maxiter": 2500, "xatol": 1.0e-10, "fatol": 1.0e-10},
        )
        direction = optimization.x / np.linalg.norm(optimization.x)
        radius = root_for_direction(direction, inverse_sqrt, polynomial, maximum_radius)
        refinements.append(
            {
                "radius": radius,
                "level": radius**2,
                "unit_certificate_direction": direction,
                "physical_error_direction": inverse_sqrt @ direction,
                "physical_error_norm_vehicles": (
                    pilot.ACCUMULATION_UNIT_VEHICLES
                    * radius
                    * np.linalg.norm(inverse_sqrt @ direction)
                ),
                "success": bool(optimization.success),
                "iterations": int(optimization.nit),
            }
        )
    best = min(refinements, key=lambda record: float(record["radius"]))
    finite_roots = roots[np.isfinite(roots)]
    quantiles = (
        np.quantile(finite_roots, [0.0, 0.001, 0.01, 0.05, 0.5]).tolist()
        if finite_roots.size
        else [maximum_radius] * 5
    )
    return {
        "mode": mode + 1,
        "sample_count": samples,
        "constraint_level": constraint_level,
        "binding_constraint": binding,
        "maximum_certificate_radius_from_constraints": maximum_radius,
        "directions_with_flow_crossing_before_constraint": int(np.sum(np.isfinite(roots))),
        "sampled_crossing_radius_quantiles_0_0.1_1_5_50_percent": quantiles,
        "locally_refined_limiting_record": best,
        "all_local_refinements": refinements,
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
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()
    if not PILOT_JSON.exists():
        raise SystemExit("Run reproducibility/mfd_cubic_pilot.py first")
    record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    gains = tuple(np.asarray(value, dtype=float) for value in record["controller_search"]["selected_gains"])
    matrices = tuple(np.asarray(value, dtype=float) for value in record["certificate"]["matrices"])
    rate = float(record["certificate"]["rate"])
    modes = [
        audit_mode(mode, gains[mode], matrices[mode], rate, args.samples)
        for mode in range(2)
    ]
    output = {
        "scope": "exact-polynomial directional boundary and falsification audit",
        "proof_status": "numerical upper-bound search; not an SOS certificate",
        "certificate_rate_per_hour": rate,
        "modes": modes,
        "estimated_common_level": min(
            float(mode["locally_refined_limiting_record"]["level"]) for mode in modes
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")
    summary = {
        "rate": rate,
        "estimated_common_level": output["estimated_common_level"],
        "modes": [
            {
                "mode": mode["mode"],
                "constraint_level": mode["constraint_level"],
                "binding_constraint": mode["binding_constraint"],
                "directions_with_crossing": mode["directions_with_flow_crossing_before_constraint"],
                "limiting_record": mode["locally_refined_limiting_record"],
            }
            for mode in modes
        ],
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
