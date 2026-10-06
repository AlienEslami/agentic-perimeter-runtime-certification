#!/usr/bin/env python3
"""Go/no-go pilot for the literature-standard cubic two-region MFD.

The existing normalized quadratic benchmark is intentionally left unchanged.
This module replaces only the regional production curve, keeps the paper's
six-state/four-command conservation structure, and regenerates every numerical
object that depends on the plant:

* a unit-consistent equilibrium demand record;
* finite-difference/complex-step Jacobians;
* two region-priority LQR modes selected on a declared deterministic grid;
* an equal-residence periodic-switching audit;
* exact degree-two through degree-four remainder bounds;
* independent quadratic flow/transition certificates; and
* matched held, unrestricted, and dwell-certified nonlinear trajectories.

Accumulation coordinates use units of 100 vehicles and time is measured in
hours.  Consequently the published curve in vehicles/second is represented as

    G(z) = 0.0014877 z^3 - 0.29815 z^2 + 15.0912 z

in 100-vehicle units/hour, where physical accumulation is n=100z vehicles.
The declared jam accumulation z=100 is a domain endpoint; because the published
coefficients are rounded, G(100) is small but nonzero.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, expm, solve_continuous_are
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import brentq, minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"

STATE_ORDER = ("n1", "n2", "n11", "n22", "n10", "n20")
INPUT_ORDER = ("u12", "u21", "u02", "u20")
DIMENSION = 6
ZERO_EXPONENT = (0,) * DIMENSION

ACCUMULATION_UNIT_VEHICLES = 100.0
X_STAR = np.array([30.0, 30.0, 15.0, 15.0, 5.0, 5.0])
# The outer-region exit gate is nominally more open than the transfer gate.
# This preserves the symmetric state composition while making d20 positive.
U_STAR = np.array([0.5, 0.5, 0.5, 0.52])
Q01 = 2.0
Q02 = 2.0
JAM_ACCUMULATION = 100.0
DIRECTIONAL_LIMIT = 0.75
PROPOSAL_INTERVAL = 1.0 / 60.0  # h (one minute, matching the literature benchmark sampling)
SIMULATION_HORIZON = 6.0  # h

# G(z)=a z^3+b z^2+c z in 100-vehicle units/hour.
MFD_A = 0.0014877
MFD_B = -0.29815
MFD_C = 15.0912

Exponent = tuple[int, ...]
Polynomial = dict[Exponent, float]


def mfd_outflow(accumulation: float | complex) -> float | complex:
    return (
        MFD_A * accumulation**3
        + MFD_B * accumulation**2
        + MFD_C * accumulation
    )


def class_flow(class_accumulation: float | complex, total: float | complex) -> float | complex:
    """Return (n_ij/n_i)G_i(n_i) after exact cancellation of n_i."""
    return class_accumulation * (MFD_A * total**2 + MFD_B * total + MFD_C)


def equilibrium_demand() -> np.ndarray:
    """Solve the six conservation equations for the primitive demands."""
    n1, n2, n11, n22, n10, n20 = X_STAR
    u12, u21, u02, u20 = U_STAR
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    q11, q12, q10 = [class_flow(value, n1) for value in (n11, n12, n10)]
    q22, q21, q20 = [class_flow(value, n2) for value in (n22, n21, n20)]

    d10 = q10 * u12
    d20 = q20 * u20 - q10 * u12
    d11 = q11 - q21 * u21
    d22 = q22 - q12 * u12 - Q02 * u02
    d1 = q11 + (q12 + q10) * u12 - q21 * u21
    d2 = (
        q22
        - (q12 + q10) * u12
        + q21 * u21
        - (Q01 + Q02) * u02
        + q20 * u20
    )
    d12 = d1 - d11 - d10
    d21 = d2 - d22 - d20
    demand = np.asarray([d11, d12, d10, d22, d21, d20], dtype=float)
    if np.min(demand) < -1.0e-12:
        raise RuntimeError(f"Equilibrium construction produced negative demand: {demand}")
    return demand


DEMAND = equilibrium_demand()


def mfd_rhs(
    state: np.ndarray,
    command: np.ndarray,
    demand: np.ndarray | None = None,
) -> np.ndarray:
    """Six conservation equations in normalized accumulation units/hour."""
    n1, n2, n11, n22, n10, n20 = state
    u12, u21, u02, u20 = command
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    d11, d12, d10, d22, d21, d20 = DEMAND if demand is None else demand
    d1 = d11 + d12 + d10
    d2 = d22 + d21 + d20

    q11, q12, q10 = [class_flow(value, n1) for value in (n11, n12, n10)]
    q22, q21, q20 = [class_flow(value, n2) for value in (n22, n21, n20)]

    dn11 = -q11 + q21 * u21 + d11
    dn22 = -q22 + q12 * u12 + Q02 * u02 + d22
    dn10 = -q10 * u12 + d10
    dn20 = -q20 * u20 + q10 * u12 + d20
    dn1 = -q11 - (q12 + q10) * u12 + q21 * u21 + d1
    dn2 = (
        -q22
        + (q12 + q10) * u12
        - q21 * u21
        + (Q01 + Q02) * u02
        - q20 * u20
        + d2
    )
    return np.asarray([dn1, dn2, dn11, dn22, dn10, dn20])


def transportation_rates(state: np.ndarray, command: np.ndarray) -> dict[str, float]:
    n1, n2, n11, n22, _n10, n20 = state
    _u12, _u21, _u02, u20 = command
    q11 = class_flow(n11, n1)
    q22 = class_flow(n22, n2)
    q20 = class_flow(n20, n2)
    return {
        "network_production_units_per_hour": float(mfd_outflow(n1) + mfd_outflow(n2)),
        "completed_trips_units_per_hour": float(q11 + q22 + q20 * u20),
    }


def centered_jacobian(fun, point: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
    columns = []
    for index in range(point.size):
        perturbation = np.zeros_like(point)
        perturbation[index] = step
        columns.append((fun(point + perturbation) - fun(point - perturbation)) / (2.0 * step))
    return np.column_stack(columns)


def complex_step_jacobian(fun, point: np.ndarray, step: float = 1.0e-25) -> np.ndarray:
    columns = []
    complex_point = point.astype(complex)
    for index in range(point.size):
        perturbation = np.zeros(point.size, dtype=complex)
        perturbation[index] = 1j * step
        columns.append(np.imag(fun(complex_point + perturbation)) / step)
    return np.column_stack(columns)


A = complex_step_jacobian(lambda state: mfd_rhs(state, U_STAR), X_STAR)
B = complex_step_jacobian(lambda command: mfd_rhs(X_STAR, command), U_STAR)


def lqr_gain(q_diagonal: np.ndarray, r_diagonal: np.ndarray) -> np.ndarray:
    riccati = solve_continuous_are(A, B, np.diag(q_diagonal), np.diag(r_diagonal))
    return -np.linalg.solve(np.diag(r_diagonal), B.T @ riccati)


def spectral_abscissa(matrix: np.ndarray) -> float:
    return float(np.max(np.real(np.linalg.eigvals(matrix))))


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def priority_weights(priority: int, ratio: float, class_weight: float) -> np.ndarray:
    """Mirrored region-priority state weights with an explicit interpretation."""
    if priority == 1:
        return np.array([ratio, 1.0, ratio * class_weight, class_weight, ratio * class_weight, class_weight])
    return np.array([1.0, ratio, class_weight, ratio * class_weight, class_weight, ratio * class_weight])


def pair_witness(
    closed_loop: tuple[np.ndarray, np.ndarray],
    minimum_residence: float = 0.002,
    maximum_residence: float = 2.0,
) -> dict[str, float | bool]:
    grid = np.geomspace(minimum_residence, maximum_residence, 900)
    radii = np.array(
        [spectral_radius(expm(closed_loop[1] * h) @ expm(closed_loop[0] * h)) for h in grid]
    )
    peak_index = int(np.argmax(radii))
    grid_peak = float(grid[peak_index])
    left = float(grid[max(0, peak_index - 1)])
    right = float(grid[min(grid.size - 1, peak_index + 1)])
    optimum = minimize_scalar(
        lambda residence: -spectral_radius(
            expm(closed_loop[1] * residence) @ expm(closed_loop[0] * residence)
        ),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    peak_radius = float(-optimum.fun)
    return {
        "unstable": peak_radius > 1.0 + 1.0e-8,
        "peak_residence_h": float(optimum.x),
        "peak_radius": peak_radius,
        "six_minute_radius": spectral_radius(
            expm(closed_loop[1] * PROPOSAL_INTERVAL)
            @ expm(closed_loop[0] * PROPOSAL_INTERVAL)
        ),
        "average_abscissa": spectral_abscissa(0.5 * (closed_loop[0] + closed_loop[1])),
    }


def controller_grid_search() -> dict[str, Any]:
    """Search a small declared grid of mirrored, interpretable LQR objectives."""
    ratios = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    class_weights = (0.1, 0.25, 0.5, 1.0)
    cross_penalties = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    external_multiplier = (1.0, 5.0, 20.0, 100.0)
    records = []
    best = None
    for ratio, class_weight, cross_penalty, multiplier in itertools.product(
        ratios, class_weights, cross_penalties, external_multiplier
    ):
        r_diagonal = np.array(
            [cross_penalty, cross_penalty, cross_penalty * multiplier, cross_penalty * multiplier]
        )
        gains = (
            lqr_gain(priority_weights(1, ratio, class_weight), r_diagonal),
            lqr_gain(priority_weights(2, ratio, class_weight), r_diagonal),
        )
        closed_loop = (A + B @ gains[0], A + B @ gains[1])
        abscissae = [spectral_abscissa(matrix) for matrix in closed_loop]
        if max(abscissae) >= -1.0e-8:
            continue
        witness = pair_witness(closed_loop)
        record = {
            "ratio": ratio,
            "class_weight": class_weight,
            "cross_penalty": cross_penalty,
            "external_multiplier": multiplier,
            "abscissae": abscissae,
            **witness,
        }
        records.append(record)
        if best is None or float(record["peak_radius"]) > float(best["record"]["peak_radius"]):
            best = {"record": record, "gains": gains, "closed_loop": closed_loop}
    if best is None:
        raise RuntimeError("Controller grid produced no stable LQR pair")
    records.sort(key=lambda item: float(item["peak_radius"]), reverse=True)
    best["top_records"] = records[:20]
    best["grid_size"] = len(records)
    return best


# These diagonal objectives are the two complementary region-priority
# templates already used in the repository.  Mode 1 places its dominant
# accumulation weight on Region 2, whereas Mode 2 places it on Region 1.
# Their actuator penalties also distinguish the economically preferred
# cross-boundary and external-flow commands in the two operating regimes.
Q_PRIORITY = (
    np.array([0.321885388639055, 41.931717889332300, 0.070571112642375,
              0.478628398066525, 1.518155717639728, 0.771979168972404]),
    np.array([41.906254418260010, 0.355409585242550, 0.214650404538254,
              0.110314543790901, 0.222263067493318, 7.547939186580425]),
)
R_PRIORITY = (
    np.array([26.286549185283980, 60.700013018766306, 1359.4883882660058,
              519.4558925269403]),
    np.array([78.913371715061610, 33.881268837504564, 202.33703257772174,
              8727.640869263709]),
)


def operational_controller_frontier() -> dict[str, Any]:
    """Balance a visible one-minute witness against nonlinear tube authority.

    A common actuator-penalty scale weakens both feedback laws without changing
    their declared objective ordering.  The selected point maximizes the
    analytic common physical radius subject to at least 0.5% amplification per
    two-minute alternating cycle at the one-minute proposal cadence.
    """
    scales = (1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)
    records = []
    candidates = []
    for scale in scales:
        gains = (
            lqr_gain(Q_PRIORITY[0], R_PRIORITY[0] * 2.5 * scale),
            lqr_gain(Q_PRIORITY[1], R_PRIORITY[1] * 2.0 * scale),
        )
        closed_loop = (A + B @ gains[0], A + B @ gains[1])
        witness = pair_witness(closed_loop, minimum_residence=5.0e-4, maximum_residence=0.2)
        one_minute_radius = spectral_radius(
            expm(closed_loop[1] * PROPOSAL_INTERVAL)
            @ expm(closed_loop[0] * PROPOSAL_INTERVAL)
        )
        certificate = independent_certificate(closed_loop, PROPOSAL_INTERVAL)
        tube = tube_record(gains, closed_loop, certificate)
        common_radius = min(float(mode["installed_radius_vehicles"]) for mode in tube["modes"])
        record = {
            "actuator_penalty_scale": scale,
            "mode_1_total_R_scale": 2.5 * scale,
            "mode_2_total_R_scale": 2.0 * scale,
            "closed_loop_spectral_abscissae": [spectral_abscissa(matrix) for matrix in closed_loop],
            "one_minute_monodromy_radius": one_minute_radius,
            "peak_residence_minutes": 60.0 * float(witness["peak_residence_h"]),
            "peak_monodromy_radius": float(witness["peak_radius"]),
            "analytic_common_radius_vehicles": common_radius,
            "constraint_radii_vehicles": [
                ACCUMULATION_UNIT_VEHICLES * float(mode["constraint_radius_units"])
                for mode in tube["modes"]
            ],
            "certificate_rate_per_hour": float(certificate["rate"]),
            "dwell_minutes": 60.0 * float(certificate["dwell_time"]),
        }
        records.append(record)
        if one_minute_radius >= 1.005:
            candidates.append((common_radius, scale, gains, closed_loop, certificate, tube, record))
    if not candidates:
        raise RuntimeError("No operational controller pair met the declared switching-witness margin")
    selected = max(candidates, key=lambda item: item[0])
    return {
        "selection_rule": (
            "maximize the analytic common physical radius subject to a one-minute "
            "alternating monodromy radius of at least 1.005"
        ),
        "frontier": records,
        "selected_scale": selected[1],
        "gains": selected[2],
        "closed_loop": selected[3],
        "certificate": selected[4],
        "tube": selected[5],
        "selected": selected[6],
    }


def generalized_max(successor: np.ndarray, predecessor: np.ndarray) -> float:
    return float(np.max(eigvalsh(successor, predecessor)))


def independent_certificate(
    closed_loop: tuple[np.ndarray, np.ndarray],
    proposal_interval: float,
) -> dict[str, Any]:
    maximum_rate = min(-2.0 * spectral_abscissa(matrix) for matrix in closed_loop)
    # Use a material fraction of the available linear rate while retaining
    # enough shifted-Lyapunov margin for nonlinear certification.
    rate = min(2.0, 0.35 * maximum_rate)
    matrices = []
    for matrix in closed_loop:
        shifted = matrix + 0.5 * rate * np.eye(DIMENSION)
        matrices.append(solve_continuous_lyapunov(shifted.T, -np.eye(DIMENSION)))
    nu12 = generalized_max(matrices[1], matrices[0])
    nu21 = generalized_max(matrices[0], matrices[1])
    scale2 = math.sqrt(nu21 / nu12)
    matrices[1] *= scale2
    transition_factor = math.sqrt(nu12 * nu21)
    dwell = math.log(transition_factor) / rate
    realized_dwell = math.ceil(dwell / proposal_interval - 1.0e-12) * proposal_interval
    flow_slacks = []
    for matrix, p_matrix in zip(closed_loop, matrices):
        residual = matrix.T @ p_matrix + p_matrix @ matrix + rate * p_matrix
        flow_slacks.append(-float(np.max(eigvalsh(residual))))
    return {
        "rate": rate,
        "maximum_linear_rate": maximum_rate,
        "matrices": matrices,
        "unscaled_transition_factors": [nu12, nu21],
        "scale_2": scale2,
        "transition_factor": transition_factor,
        "dwell_time": dwell,
        "realized_dwell": realized_dwell,
        "sampled_contraction": transition_factor * math.exp(-rate * realized_dwell),
        "flow_slacks": flow_slacks,
    }


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = left.copy()
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, 0.0) + value
    return {key: value for key, value in result.items() if abs(value) > 1.0e-13}


def poly_scale(polynomial: Polynomial, scale: float) -> Polynomial:
    return {exponent: scale * value for exponent, value in polynomial.items()}


def poly_sub(left: Polynomial, right: Polynomial) -> Polynomial:
    return poly_add(left, poly_scale(right, -1.0))


def poly_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for exponent_left, value_left in left.items():
        for exponent_right, value_right in right.items():
            exponent = tuple(a + b for a, b in zip(exponent_left, exponent_right))
            result[exponent] = result.get(exponent, 0.0) + value_left * value_right
    return {key: value for key, value in result.items() if abs(value) > 1.0e-13}


def constant(value: float) -> Polynomial:
    return {ZERO_EXPONENT: float(value)}


def variable(index: int) -> Polynomial:
    exponent = [0] * DIMENSION
    exponent[index] = 1
    return {tuple(exponent): 1.0}


def affine(offset: float, row: np.ndarray) -> Polynomial:
    polynomial = constant(offset)
    for index, coefficient in enumerate(row):
        polynomial = poly_add(polynomial, poly_scale(variable(index), float(coefficient)))
    return polynomial


def class_flow_polynomial(class_state: Polynomial, total_state: Polynomial) -> Polynomial:
    speed_factor = poly_add(
        poly_add(
            poly_scale(poly_mul(total_state, total_state), MFD_A),
            poly_scale(total_state, MFD_B),
        ),
        constant(MFD_C),
    )
    return poly_mul(class_state, speed_factor)


def closed_loop_polynomials(gain: np.ndarray) -> list[Polynomial]:
    d11, d12, d10, d22, d21, d20 = DEMAND
    d1 = float(d11 + d12 + d10)
    d2 = float(d22 + d21 + d20)
    state = [affine(X_STAR[index], np.eye(DIMENSION)[index]) for index in range(DIMENSION)]
    n1, n2, n11, n22, n10, n20 = state
    n12 = poly_sub(poly_sub(n1, n11), n10)
    n21 = poly_sub(poly_sub(n2, n22), n20)
    command = [affine(U_STAR[index], gain[index]) for index in range(4)]
    u12, u21, u02, u20 = command
    q11, q12, q10 = [class_flow_polynomial(value, n1) for value in (n11, n12, n10)]
    q22, q21, q20 = [class_flow_polynomial(value, n2) for value in (n22, n21, n20)]

    dn11 = poly_add(poly_add(poly_scale(q11, -1.0), poly_mul(q21, u21)), constant(d11))
    dn22 = poly_add(
        poly_add(poly_add(poly_scale(q22, -1.0), poly_mul(q12, u12)), poly_scale(u02, Q02)),
        constant(d22),
    )
    dn10 = poly_add(poly_scale(poly_mul(q10, u12), -1.0), constant(d10))
    dn20 = poly_add(
        poly_add(poly_scale(poly_mul(q20, u20), -1.0), poly_mul(q10, u12)),
        constant(d20),
    )
    dn1 = poly_add(
        poly_add(
            poly_add(poly_scale(q11, -1.0), poly_scale(poly_mul(poly_add(q12, q10), u12), -1.0)),
            poly_mul(q21, u21),
        ),
        constant(d1),
    )
    dn2 = poly_add(
        poly_add(
            poly_add(
                poly_add(
                    poly_add(poly_scale(q22, -1.0), poly_mul(poly_add(q12, q10), u12)),
                    poly_scale(poly_mul(q21, u21), -1.0),
                ),
                poly_scale(u02, Q01 + Q02),
            ),
            poly_scale(poly_mul(q20, u20), -1.0),
        ),
        constant(d2),
    )
    return [dn1, dn2, dn11, dn22, dn10, dn20]


def remainder_vectors(gain: np.ndarray, closed_loop: np.ndarray) -> dict[Exponent, np.ndarray]:
    polynomials = closed_loop_polynomials(gain)
    recovered_constant = np.zeros(DIMENSION)
    recovered_linear = np.zeros((DIMENSION, DIMENSION))
    vectors: dict[Exponent, np.ndarray] = {}
    for output, polynomial in enumerate(polynomials):
        recovered_constant[output] = polynomial.get(ZERO_EXPONENT, 0.0)
        for input_index in range(DIMENSION):
            exponent = tuple(1 if index == input_index else 0 for index in range(DIMENSION))
            recovered_linear[output, input_index] = polynomial.get(exponent, 0.0)
        for exponent, coefficient in polynomial.items():
            if sum(exponent) >= 2:
                vectors.setdefault(exponent, np.zeros(DIMENSION))[output] = coefficient
    if np.max(np.abs(recovered_constant)) > 2.0e-10:
        raise AssertionError(f"Polynomial equilibrium residual: {recovered_constant}")
    if np.max(np.abs(recovered_linear - closed_loop)) > 2.0e-8:
        raise AssertionError("Polynomial linear coefficient does not match the Jacobian")
    degrees = {sum(exponent) for exponent in vectors}
    if not degrees.issubset({2, 3, 4}) or 4 not in degrees:
        raise AssertionError(f"Unexpected nonlinear degrees: {sorted(degrees)}")
    return vectors


def state_constraint_rows() -> tuple[list[str], list[np.ndarray], list[float]]:
    names: list[str] = []
    rows: list[np.ndarray] = []
    margins: list[float] = []
    for index, name in enumerate(STATE_ORDER):
        row = np.zeros(DIMENSION)
        row[index] = 1.0
        names.append(f"{name} lower")
        rows.append(row)
        margins.append(float(X_STAR[index]))
    for index, name in ((0, "n1"), (1, "n2")):
        row = np.zeros(DIMENSION)
        row[index] = -1.0
        names.append(f"{name} upper")
        rows.append(row)
        margins.append(float(JAM_ACCUMULATION - X_STAR[index]))
    rows.extend(
        [
            np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0]),
        ]
    )
    names.extend(["n12 lower", "n21 lower"])
    margins.extend([10.0, 10.0])
    return names, rows, margins


def exact_constraint_level(p_matrix: np.ndarray, gain: np.ndarray) -> tuple[float, str, float]:
    inverse = np.linalg.inv(p_matrix)
    names, rows, margins = state_constraint_rows()
    candidates: list[tuple[float, str]] = [
        (float(margin**2 / (row @ inverse @ row)), name)
        for name, row, margin in zip(names, rows, margins)
    ]
    for index, row in enumerate(gain):
        candidates.append((float(0.25 / (row @ inverse @ row)), f"{INPUT_ORDER[index]} box"))
    for label, row in (
        ("u21-u12 directional", gain[1] - gain[0]),
        ("u20-u02 directional", gain[3] - gain[2]),
    ):
        candidates.append((float(DIRECTIONAL_LIMIT**2 / (row @ inverse @ row)), label))
    level, binding = min(candidates)
    euclidean_radius = math.sqrt(level / float(np.max(eigvalsh(p_matrix))))
    return level, binding, euclidean_radius


def positive_root(function, upper: float = 10.0) -> float:
    while function(upper) < 0.0:
        upper *= 2.0
        if upper > 1.0e5:
            raise RuntimeError("Unable to bracket nonlinear radius")
    return float(brentq(function, 0.0, upper))


def tube_record(
    gains: tuple[np.ndarray, np.ndarray],
    closed_loop: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
) -> dict[str, Any]:
    modes = []
    for mode, (gain, matrix, p_matrix, slack) in enumerate(
        zip(gains, closed_loop, certificate["matrices"], certificate["flow_slacks"])
    ):
        vectors = remainder_vectors(gain, matrix)
        values, directions = np.linalg.eigh(p_matrix)
        sqrt_p = directions @ np.diag(np.sqrt(values)) @ directions.T
        euclidean = {
            degree: float(
                sum(np.linalg.norm(vector) for exponent, vector in vectors.items() if sum(exponent) == degree)
            )
            for degree in (2, 3, 4)
        }
        metric = {
            degree: float(
                sum(
                    np.linalg.norm(sqrt_p @ vector)
                    for exponent, vector in vectors.items()
                    if sum(exponent) == degree
                )
            )
            for degree in (2, 3, 4)
        }
        p_norm = float(np.max(values))
        direct_radius = positive_root(
            lambda radius: 2.0
            * p_norm
            * (
                euclidean[2] * radius
                + euclidean[3] * radius**2
                + euclidean[4] * radius**3
            )
            - slack
        )
        metric_radius = positive_root(
            lambda radius: 2.0
            * math.sqrt(p_norm)
            * (
                metric[2] * radius
                + metric[3] * radius**2
                + metric[4] * radius**3
            )
            - slack
        )
        constraint_level, binding_constraint, constraint_radius = exact_constraint_level(p_matrix, gain)
        installed_radius = min(metric_radius, constraint_radius)
        flow_level = float(np.min(values) * installed_radius**2)
        modes.append(
            {
                "mode": mode + 1,
                "euclidean_coefficient_sums": euclidean,
                "metric_coefficient_sums": metric,
                "direct_radius_units": direct_radius,
                "metric_radius_units": metric_radius,
                "constraint_radius_units": constraint_radius,
                "installed_radius_units": installed_radius,
                "installed_radius_vehicles": ACCUMULATION_UNIT_VEHICLES * installed_radius,
                "flow_level": flow_level,
                "constraint_level": constraint_level,
                "binding_constraint": binding_constraint,
            }
        )
    common_level = min(float(mode["flow_level"]) for mode in modes)
    return {"modes": modes, "common_level": common_level}


def margin_components(state: np.ndarray, mode: int, gains: tuple[np.ndarray, np.ndarray]) -> tuple[list[str], np.ndarray]:
    error = state - X_STAR
    command = U_STAR + gains[mode] @ error
    n1, n2, n11, n22, n10, n20 = state
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    names = [f"{name} lower" for name in INPUT_ORDER] + [f"{name} upper" for name in INPUT_ORDER]
    margins = list(command) + list(1.0 - command)
    names += ["u21-u12 directional", "u20-u02 directional"]
    margins += [DIRECTIONAL_LIMIT - abs(command[1] - command[0]), DIRECTIONAL_LIMIT - abs(command[3] - command[2])]
    names += ["n1 lower", "n2 lower", "n1 upper", "n2 upper", "n11 lower", "n22 lower", "n10 lower", "n20 lower", "n12 lower", "n21 lower"]
    margins += [n1, n2, JAM_ACCUMULATION - n1, JAM_ACCUMULATION - n2, n11, n22, n10, n20, n12, n21]
    return names, np.asarray(margins)


def dominant_initial_state(
    gains: tuple[np.ndarray, np.ndarray],
    closed_loop: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> np.ndarray:
    residence = float(pair_witness(closed_loop)["peak_residence_h"])
    monodromy = expm(closed_loop[1] * residence) @ expm(closed_loop[0] * residence)
    eigenvalues, eigenvectors = np.linalg.eig(monodromy)
    direction = np.real(eigenvectors[:, int(np.argmax(np.abs(eigenvalues)))])
    direction /= np.linalg.norm(direction)
    p_matrix = certificate["matrices"][0]
    scale = math.sqrt(0.45 * tube["common_level"] / float(direction @ p_matrix @ direction))
    return X_STAR + scale * direction


def simulate(
    policy: str,
    gains: tuple[np.ndarray, np.ndarray],
    closed_loop: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    envelope_rate: float = 0.5,
    initial_state: np.ndarray | None = None,
    proposal_period_minutes: float | None = None,
    proposal_schedule: list[int] | tuple[int, ...] | None = None,
    include_trajectory: bool = False,
) -> dict[str, Any]:
    if proposal_schedule is not None:
        if proposal_period_minutes is None:
            raise ValueError("proposal_schedule requires proposal_period_minutes")
        if not proposal_schedule or any(mode not in (0, 1) for mode in proposal_schedule):
            raise ValueError("proposal_schedule must be a nonempty sequence of modes 0/1")

    def scheduled_mode(time: float) -> int:
        if proposal_period_minutes is None:
            raise ValueError("scheduled_mode requires proposal_period_minutes")
        block = int(
            math.floor((time + 1.0e-10) / (proposal_period_minutes / 60.0))
        )
        if proposal_schedule is None:
            # Mode 2 prioritizes Region 1; Mode 1 prioritizes Region 2.
            return 1 if block % 2 == 0 else 0
        return int(proposal_schedule[min(block, len(proposal_schedule) - 1)])

    state = (
        dominant_initial_state(gains, closed_loop, certificate, tube)
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    mode = 0
    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    accepted = []
    rejected = 0
    no_op_reselections = 0
    last_switch = 0.0
    initial_occupancy = float(
        (state - X_STAR)
        @ certificate["matrices"][mode]
        @ (state - X_STAR)
        / tube["common_level"]
    )
    witness_residence = float(pair_witness(closed_loop)["peak_residence_h"])
    if policy == "unrestricted_peak":
        interval = witness_residence
    else:
        interval = PROPOSAL_INTERVAL
    horizon = SIMULATION_HORIZON
    step = 0
    violation = None
    while times[-1] < horizon - 1.0e-12:
        time = times[-1]
        if proposal_period_minutes is None:
            proposed_mode = step % 2
        else:
            proposed_mode = scheduled_mode(time)
        requested = mode
        if policy == "held_1":
            requested = 0
        elif policy == "held_2":
            requested = 1
        elif policy in {"unrestricted_one_minute", "unrestricted_peak"}:
            requested = proposed_mode
        elif policy == "dwell_certified":
            requested = proposed_mode
        elif policy in {"landing_only", "time_envelope"}:
            requested = proposed_mode
        else:
            raise ValueError(policy)
        if requested != mode:
            candidate_names, candidate_margins = margin_components(state, requested, gains)
            error = state - X_STAR
            candidate_occupancy = float(
                error
                @ certificate["matrices"][requested]
                @ error
                / tube["common_level"]
            )
            admissible = np.min(candidate_margins) >= 0.0
            if policy in {"dwell_certified", "landing_only", "time_envelope"}:
                admissible = admissible and candidate_occupancy <= 1.0 + 1.0e-10
            if policy == "dwell_certified":
                admissible = admissible and (
                    time - last_switch + 1.0e-12 >= certificate["realized_dwell"]
                )
            if policy == "time_envelope":
                if not 0.0 < envelope_rate <= certificate["rate"]:
                    raise ValueError("envelope rate must lie in (0, certificate flow rate]")
                admissible = admissible and candidate_occupancy <= (
                    initial_occupancy * math.exp(-envelope_rate * time) + 1.0e-10
                )
            if policy in {"unrestricted_one_minute", "unrestricted_peak"} or admissible:
                mode = requested
                last_switch = time
                accepted.append(time)
            else:
                rejected += 1
            if np.min(candidate_margins) < 0.0 and policy in {"unrestricted_one_minute", "unrestricted_peak"}:
                index = int(np.argmin(candidate_margins))
                violation = {"time_h": time, "constraint": candidate_names[index], "margin": float(candidate_margins[index])}
                break
        else:
            no_op_reselections += 1
        names, margins = margin_components(state, mode, gains)
        if np.min(margins) < -1.0e-9:
            index = int(np.argmin(margins))
            violation = {"time_h": time, "constraint": names[index], "margin": float(margins[index])}
            break
        end = min(time + interval, horizon)
        solution = solve_ivp(
            lambda _time, current: mfd_rhs(current, U_STAR + gains[mode] @ (current - X_STAR)),
            (time, end),
            state,
            method="DOP853",
            t_eval=[end],
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=min(0.0025, interval / 10.0),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1]
        times.append(end)
        states.append(state.copy())
        modes.append(mode)
        step += 1

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    occupancies = np.asarray(
        [
            (state_value - X_STAR)
            @ certificate["matrices"][mode_value]
            @ (state_value - X_STAR)
            / tube["common_level"]
            for state_value, mode_value in zip(states_array, modes_array)
        ]
    )
    minimum_margin = min(
        float(np.min(margin_components(state_value, int(mode_value), gains)[1]))
        for state_value, mode_value in zip(states_array, modes_array)
    )
    commands = np.asarray(
        [U_STAR + gains[int(mode_value)] @ (state_value - X_STAR)
         for state_value, mode_value in zip(states_array, modes_array)]
    )
    rates = [
        transportation_rates(state_value, command)
        for state_value, command in zip(states_array, commands)
    ]
    production = np.asarray([item["network_production_units_per_hour"] for item in rates])
    completions = np.asarray([item["completed_trips_units_per_hour"] for item in rates])
    transportation = {
        "vehicle_hours_in_network": float(
            ACCUMULATION_UNIT_VEHICLES
            * np.trapezoid(states_array[:, 0] + states_array[:, 1], times_array)
        ),
        "excess_vehicle_hours_relative_to_target": float(
            ACCUMULATION_UNIT_VEHICLES
            * np.trapezoid(states_array[:, 0] + states_array[:, 1] - X_STAR[0] - X_STAR[1], times_array)
        ),
        "regional_imbalance_vehicle_hours": float(
            ACCUMULATION_UNIT_VEHICLES
            * np.trapezoid(np.abs(states_array[:, 0] - states_array[:, 1]), times_array)
        ),
        "cumulative_network_production_vehicles": float(
            ACCUMULATION_UNIT_VEHICLES * np.trapezoid(production, times_array)
        ),
        "completed_trips_vehicles": float(
            ACCUMULATION_UNIT_VEHICLES * np.trapezoid(completions, times_array)
        ),
        "quadratic_command_effort": float(
            np.trapezoid(np.sum((commands - U_STAR) ** 2, axis=1), times_array)
        ),
    }
    if proposal_period_minutes is not None:
        desired_modes = np.asarray(
            [scheduled_mode(float(time)) for time in times_array],
            dtype=int,
        )
        errors = states_array - X_STAR
        priority_cost = np.asarray(
            [
                float(np.sum(Q_PRIORITY[desired] * error**2))
                for error, desired in zip(errors, desired_modes)
            ]
        )
        transportation["time_varying_priority_tracking_cost"] = float(
            np.trapezoid(priority_cost, times_array)
        )
        transportation["fraction_time_executed_mode_matches_priority"] = float(
            np.mean(modes_array == desired_modes)
        )
    exit_indices = np.flatnonzero(occupancies > 1.0 + 1.0e-9)
    envelope_ratios = occupancies / (
        initial_occupancy * np.exp(-envelope_rate * times_array)
    )
    result = {
        "policy": policy,
        "horizon_h": float(times_array[-1]),
        "integration_interval_h": interval,
        "executed_changes": len(accepted),
        "rejected_changes": rejected,
        "no_op_reselections": no_op_reselections,
        "maximum_tube_occupancy": float(np.max(occupancies)),
        "final_tube_occupancy": float(occupancies[-1]),
        "maximum_envelope_ratio": (
            float(np.max(envelope_ratios)) if policy == "time_envelope" else None
        ),
        "first_tube_exit_time_h": (
            None if exit_indices.size == 0 else float(times_array[int(exit_indices[0])])
        ),
        "final_error_vehicles": float(ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(states_array[-1] - X_STAR)),
        "maximum_error_vehicles": float(ACCUMULATION_UNIT_VEHICLES * np.max(np.linalg.norm(states_array - X_STAR, axis=1))),
        "minimum_physical_margin": minimum_margin,
        "transportation": transportation,
        "violation": violation,
    }
    if include_trajectory:
        result["trajectory"] = {
            "time_h": times_array,
            "tube_occupancy": occupancies,
            "error_norm_vehicles": (
                ACCUMULATION_UNIT_VEHICLES
                * np.linalg.norm(states_array - X_STAR, axis=1)
            ),
            "executed_mode": modes_array,
        }
    return result


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    equilibrium_residual = mfd_rhs(X_STAR, U_STAR)
    finite_a = centered_jacobian(lambda state: mfd_rhs(state, U_STAR), X_STAR)
    finite_b = centered_jacobian(lambda command: mfd_rhs(X_STAR, command), U_STAR)
    critical_roots = np.sort(np.roots([3.0 * MFD_A, 2.0 * MFD_B, MFD_C]))
    critical = float(critical_roots[0])
    physical_maximum = float(ACCUMULATION_UNIT_VEHICLES * mfd_outflow(critical) / 3600.0)
    jam_flow = float(ACCUMULATION_UNIT_VEHICLES * mfd_outflow(JAM_ACCUMULATION) / 3600.0)

    mirrored_search = controller_grid_search()
    frontier = operational_controller_frontier()
    gains = frontier["gains"]
    closed_loop = frontier["closed_loop"]
    certificate = frontier["certificate"]
    tube = frontier["tube"]
    simulations = {
        policy: simulate(policy, gains, closed_loop, certificate, tube)
        for policy in (
            "held_1",
            "held_2",
            "unrestricted_one_minute",
            "unrestricted_peak",
            "landing_only",
            "time_envelope",
            "dwell_certified",
        )
    }

    controllability = np.column_stack([np.linalg.matrix_power(A, power) @ B for power in range(DIMENSION)])
    output = np.column_stack((np.eye(2), np.zeros((2, 4))))
    observability = np.vstack([output @ np.linalg.matrix_power(A, power) for power in range(DIMENSION)])
    result = {
        "status": "pilot",
        "model": {
            "state_order": list(STATE_ORDER),
            "input_order": list(INPUT_ORDER),
            "accumulation_unit_vehicles": ACCUMULATION_UNIT_VEHICLES,
            "time_unit": "hour",
            "production_coefficients_normalized_per_hour": [MFD_A, MFD_B, MFD_C],
            "critical_accumulation_units": critical,
            "critical_accumulation_vehicles": ACCUMULATION_UNIT_VEHICLES * critical,
            "maximum_trip_completion_vehicles_per_second": physical_maximum,
            "declared_jam_accumulation_vehicles": ACCUMULATION_UNIT_VEHICLES * JAM_ACCUMULATION,
            "trip_completion_at_jam_vehicles_per_second": jam_flow,
            "equilibrium": X_STAR,
            "equilibrium_command": U_STAR,
            "equilibrium_demand_units_per_hour": DEMAND,
            "equilibrium_demand_vehicles_per_second": ACCUMULATION_UNIT_VEHICLES * DEMAND / 3600.0,
            "equilibrium_residual_inf": float(np.max(np.abs(equilibrium_residual))),
            "jacobian_centered_vs_complex_A_inf": float(np.max(np.abs(finite_a - A))),
            "jacobian_centered_vs_complex_B_inf": float(np.max(np.abs(finite_b - B))),
            "controllability_rank": int(np.linalg.matrix_rank(controllability)),
            "two_total_output_observability_rank": int(np.linalg.matrix_rank(observability)),
            "A": A,
            "B": B,
        },
        "controller_search": {
            "mirrored_region_priority_grid": {
                "declared_grid_size": mirrored_search["grid_size"],
                "largest_peak_radius": mirrored_search["record"],
                "unstable_pair_found": bool(mirrored_search["record"]["unstable"]),
            },
            "operational_frontier": {
                key: value
                for key, value in frontier.items()
                if key not in {"gains", "closed_loop", "certificate", "tube"}
            },
            "selected_gains": gains,
        },
        "certificate": certificate,
        "tube": tube,
        "simulations": simulations,
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
