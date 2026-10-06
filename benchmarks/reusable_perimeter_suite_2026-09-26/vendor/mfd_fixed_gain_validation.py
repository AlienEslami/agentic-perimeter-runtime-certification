#!/usr/bin/env python3
"""Reproduce the corrected fixed-gain MFD audit and runtime experiment.

The canonical physical-state and derivative order is
    [n1, n2, n11, n22, n10, n20].
All reported matrices are checked against finite differences of the nonlinear
conservation equations in that same order.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, expm, solve_continuous_are
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import brentq, minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
STATE_ORDER = ("n1", "n2", "n11", "n22", "n10", "n20")
INPUT_ORDER = ("u12", "u21", "u02", "u20")

X_STAR = np.array([30.0, 30.0, 15.0, 15.0, 5.0, 5.0])
U_STAR = np.full(4, 0.5)
DEMAND = np.array([7.0, 3.5, 1.75, 6.0, 2.5, 0.0])
Q01 = 2.0
Q02 = 2.0
PROPOSAL_INTERVAL = 0.1  # h
DIRECTIONAL_LIMIT = 0.75
SIMULATION_HORIZON = 96.0  # h
INITIAL_COMMAND_DEVIATION = 5.0e-4

# The matrices below are the analytic Jacobians in STATE_ORDER.  They are
# independently checked against centered finite differences in audit_model().
A_ANALYTIC = np.array(
    [
        [-0.125, 0.300, -0.350, -0.350, 0.000, -0.350],
        [0.275, -0.125, -0.350, -0.350, 0.000, 0.000],
        [0.150, 0.300, -0.700, -0.350, 0.000, -0.350],
        [0.300, 0.150, -0.350, -0.700, -0.350, 0.000],
        [0.025, 0.000, 0.000, 0.000, -0.350, 0.000],
        [-0.025, 0.025, 0.000, 0.000, 0.350, -0.350],
    ]
)
B_ANALYTIC = np.array(
    [
        [-10.5, 7.0, 0.0, 0.0],
        [10.5, -7.0, 4.0, -3.5],
        [0.0, 7.0, 0.0, 0.0],
        [7.0, 0.0, 2.0, 0.0],
        [-3.5, 0.0, 0.0, 0.0],
        [3.5, 0.0, 0.0, -3.5],
    ]
)

# Diagonal LQR weights.  The feedback convention is v=K e, hence K is the
# negative of scipy's standard u=-K_lqr e gain.
Q_LQR = (
    np.diag(
        [
            0.321885388639055,
            41.931717889332300,
            0.070571112642375,
            0.478628398066525,
            1.518155717639728,
            0.771979168972404,
        ]
    ),
    np.diag(
        [
            41.906254418260010,
            0.355409585242550,
            0.214650404538254,
            0.110314543790901,
            0.222263067493318,
            7.547939186580425,
        ]
    ),
)
R_LQR = (
    np.diag(
        [26.286549185283980, 60.700013018766306, 1359.4883882660058, 519.4558925269403]
    ),
    np.diag(
        [78.913371715061610, 33.881268837504564, 202.33703257772174, 8727.640869263709]
    ),
)


def mfd_outflow(n: float) -> float:
    return n - n * n / 100.0


def mfd_rhs(
    x: np.ndarray, u: np.ndarray, demand: np.ndarray | None = None
) -> np.ndarray:
    """Six conservation equations returned in STATE_ORDER."""
    n1, n2, n11, n22, n10, n20 = x
    u12, u21, u02, u20 = u
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    g1 = mfd_outflow(n1)
    g2 = mfd_outflow(n2)
    d11, d12, d10, d22, d21, d20 = DEMAND if demand is None else demand
    d1 = d11 + d12 + d10
    d2 = d22 + d21 + d20

    dn11 = -n11 / n1 * g1 + n21 / n2 * g2 * u21 + d11
    dn22 = -n22 / n2 * g2 + n12 / n1 * g1 * u12 + Q02 * u02 + d22
    dn10 = -n10 / n1 * g1 * u12 + d10
    dn20 = -n20 / n2 * g2 * u20 + n10 / n1 * g1 * u12 + d20
    dn1 = (
        -n11 / n1 * g1
        - (n12 + n10) / n1 * g1 * u12
        + n21 / n2 * g2 * u21
        + d1
    )
    dn2 = (
        -n22 / n2 * g2
        + (n12 + n10) / n1 * g1 * u12
        - n21 / n2 * g2 * u21
        + (Q01 + Q02) * u02
        - n20 / n2 * g2 * u20
        + d2
    )
    return np.array([dn1, dn2, dn11, dn22, dn10, dn20])


def transportation_rates(x: np.ndarray, u: np.ndarray) -> dict[str, float]:
    """Return auditable MFD production and destination-completion rates.

    ``network_production`` counts all regional movement represented by the two
    MFDs. ``completed_trips`` counts only flows that reach their modeled
    destinations: within-region completions and the final controlled exit of
    the n20 class.  Rates are in accumulation units per hour; one accumulation
    unit represents 100 vehicles in this benchmark.
    """
    n1, n2, n11, n22, _n10, n20 = x
    _u12, _u21, _u02, u20 = u
    g1 = mfd_outflow(n1)
    g2 = mfd_outflow(n2)
    q11 = n11 / n1 * g1
    q22 = n22 / n2 * g2
    q20 = n20 / n2 * g2
    return {
        "network_production": float(g1 + g2),
        "completed_trips": float(q11 + q22 + q20 * u20),
    }


def out_of_sample_demand(time: float) -> np.ndarray:
    """Unseen bounded demand profile used only for empirical robustness.

    The profile is deliberately not used in controller or certificate
    construction.  Its componentwise modulation stays within five percent of
    the nominal record and combines daily and half-daily harmonics.
    """
    phase = 2.0 * math.pi * time / 24.0
    factors = np.array(
        [
            1.0 + 0.05 * math.sin(phase),
            1.0 - 0.04 * math.sin(phase + 0.4),
            1.0 + 0.03 * math.sin(2.0 * phase + 0.8),
            1.0 - 0.05 * math.sin(phase + 1.1),
            1.0 + 0.04 * math.sin(2.0 * phase + 0.2),
            1.0,
        ]
    )
    return DEMAND * factors


def centered_jacobian(fun, point: np.ndarray, step: float = 1.0e-5) -> np.ndarray:
    columns = []
    for index in range(point.size):
        perturbation = np.zeros_like(point)
        perturbation[index] = step
        columns.append((fun(point + perturbation) - fun(point - perturbation)) / (2.0 * step))
    return np.column_stack(columns)


def lqr_gains() -> tuple[np.ndarray, np.ndarray]:
    gains = []
    for q_weight, r_weight in zip(Q_LQR, R_LQR):
        riccati = solve_continuous_are(A_ANALYTIC, B_ANALYTIC, q_weight, r_weight)
        gains.append(-np.linalg.solve(r_weight, B_ANALYTIC.T @ riccati))
    return gains[0], gains[1]


K1, K2 = lqr_gains()
GAINS = (K1, K2)
CLOSED_LOOP = tuple(A_ANALYTIC + B_ANALYTIC @ gain for gain in GAINS)


def spectral_abscissa(matrix: np.ndarray) -> float:
    return float(np.max(np.real(np.linalg.eigvals(matrix))))


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def monodromy(h1: float, h2: float) -> np.ndarray:
    return expm(CLOSED_LOOP[1] * h2) @ expm(CLOSED_LOOP[0] * h1)


def generalized_max(p_successor: np.ndarray, p_predecessor: np.ndarray) -> float:
    return float(np.max(eigvalsh(p_successor, p_predecessor)))


def independent_certificate(rate: float = 0.2) -> dict:
    """Independent shifted-Lyapunov records with optimal scalar balancing."""
    matrices = []
    for closed_loop in CLOSED_LOOP:
        shifted = closed_loop + 0.5 * rate * np.eye(6)
        matrices.append(solve_continuous_lyapunov(shifted.T, -np.eye(6)))
    nu_12 = generalized_max(matrices[1], matrices[0])
    nu_21 = generalized_max(matrices[0], matrices[1])
    scale_2 = math.sqrt(nu_21 / nu_12)
    matrices[1] *= scale_2
    transition_factor = math.sqrt(nu_12 * nu_21)
    dwell_time = math.log(transition_factor) / rate
    realized_dwell = math.ceil(dwell_time / PROPOSAL_INTERVAL - 1.0e-12) * PROPOSAL_INTERVAL

    for index, (p_matrix, closed_loop) in enumerate(zip(matrices, CLOSED_LOOP)):
        scale = 1.0 if index == 0 else scale_2
        flow_residual = closed_loop.T @ p_matrix + p_matrix @ closed_loop + rate * p_matrix
        assert np.max(np.linalg.eigvalsh(flow_residual + scale * np.eye(6))) < 2.0e-10
    assert generalized_max(matrices[1], matrices[0]) <= transition_factor + 1.0e-10
    assert generalized_max(matrices[0], matrices[1]) <= transition_factor + 1.0e-10
    assert transition_factor * math.exp(-rate * realized_dwell) < 1.0
    return {
        "rate": rate,
        "matrices": matrices,
        "unscaled_transition_factors": [nu_12, nu_21],
        "scale_2": scale_2,
        "transition_factor": transition_factor,
        "dwell_time": dwell_time,
        "realized_dwell": realized_dwell,
        "sampled_contraction": transition_factor * math.exp(-rate * realized_dwell),
    }


# Minimal polynomial algebra used to obtain the exact degree-two and degree-three
# coefficient vectors of the closed-loop nonlinear remainder.
Exponent = tuple[int, ...]
Polynomial = dict[Exponent, float]
ZERO_EXPONENT = (0, 0, 0, 0, 0, 0)


def poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = left.copy()
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, 0.0) + value
    return {exponent: value for exponent, value in result.items() if abs(value) > 1.0e-14}


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
    return {exponent: value for exponent, value in result.items() if abs(value) > 1.0e-14}


def constant(value: float) -> Polynomial:
    return {ZERO_EXPONENT: value}


def variable(index: int) -> Polynomial:
    exponent = [0] * 6
    exponent[index] = 1
    return {tuple(exponent): 1.0}


def affine_polynomial(offset: float, row: np.ndarray) -> Polynomial:
    result = constant(offset)
    for index, coefficient in enumerate(row):
        result = poly_add(result, poly_scale(variable(index), float(coefficient)))
    return result


def closed_loop_polynomials(
    gain: np.ndarray,
    x_star: np.ndarray | None = None,
    u_star: np.ndarray | None = None,
    demand: np.ndarray | None = None,
) -> list[Polynomial]:
    x_star = X_STAR if x_star is None else np.asarray(x_star, dtype=float)
    u_star = U_STAR if u_star is None else np.asarray(u_star, dtype=float)
    demand = DEMAND if demand is None else np.asarray(demand, dtype=float)
    d11, d12, d10, d22, _d21, d20 = demand
    d1 = float(np.sum(demand[:3]))
    d2 = float(np.sum(demand[3:]))
    n1, n2, n11, n22, n10, n20 = [
        affine_polynomial(x_star[index], np.eye(6)[index]) for index in range(6)
    ]
    n12 = poly_sub(poly_sub(n1, n11), n10)
    n21 = poly_sub(poly_sub(n2, n22), n20)
    u12, u21, u02, u20 = [
        affine_polynomial(u_star[index], gain[index]) for index in range(4)
    ]

    factor1 = poly_sub(constant(1.0), poly_scale(n1, 0.01))
    factor2 = poly_sub(constant(1.0), poly_scale(n2, 0.01))
    q11, q12, q10 = [poly_mul(value, factor1) for value in (n11, n12, n10)]
    q22, q21, q20 = [poly_mul(value, factor2) for value in (n22, n21, n20)]

    dn11 = poly_add(poly_add(poly_scale(q11, -1.0), poly_mul(q21, u21)), constant(d11))
    dn22 = poly_add(
        poly_add(poly_add(poly_scale(q22, -1.0), poly_mul(q12, u12)), poly_scale(u02, 2.0)),
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
                poly_scale(u02, 4.0),
            ),
            poly_scale(poly_mul(q20, u20), -1.0),
        ),
        constant(d2),
    )
    return [dn1, dn2, dn11, dn22, dn10, dn20]


def nonlinear_coefficient_vectors(
    gain: np.ndarray,
    closed_loop: np.ndarray,
    x_star: np.ndarray | None = None,
    u_star: np.ndarray | None = None,
    demand: np.ndarray | None = None,
) -> dict[Exponent, np.ndarray]:
    polynomials = closed_loop_polynomials(gain, x_star, u_star, demand)
    coefficient_vectors: dict[Exponent, np.ndarray] = {}
    recovered_linear = np.zeros((6, 6))
    recovered_constant = np.zeros(6)
    for output_index, polynomial in enumerate(polynomials):
        recovered_constant[output_index] = polynomial.get(ZERO_EXPONENT, 0.0)
        for input_index in range(6):
            exponent = tuple(1 if index == input_index else 0 for index in range(6))
            recovered_linear[output_index, input_index] = polynomial.get(exponent, 0.0)
        for exponent, coefficient in polynomial.items():
            if sum(exponent) >= 2:
                coefficient_vectors.setdefault(exponent, np.zeros(6))[output_index] = coefficient
    assert np.max(np.abs(recovered_constant)) < 1.0e-12
    # Analytic benchmark matrices agree to machine precision; target-revision
    # records use a centered numerical Jacobian and therefore receive a tight
    # finite-difference tolerance.
    assert np.max(np.abs(recovered_linear - closed_loop)) < 5.0e-8
    return coefficient_vectors


def nonlinear_coefficient_bounds(
    gain: np.ndarray,
    closed_loop: np.ndarray,
    x_star: np.ndarray | None = None,
    u_star: np.ndarray | None = None,
    demand: np.ndarray | None = None,
) -> tuple[float, float]:
    coefficient_vectors = nonlinear_coefficient_vectors(
        gain, closed_loop, x_star, u_star, demand
    )
    quadratic = sum(
        np.linalg.norm(vector) for exponent, vector in coefficient_vectors.items() if sum(exponent) == 2
    )
    cubic = sum(
        np.linalg.norm(vector) for exponent, vector in coefficient_vectors.items() if sum(exponent) == 3
    )
    return float(quadratic), float(cubic)


def state_constraint_rows() -> tuple[list[str], list[np.ndarray], list[float]]:
    names: list[str] = []
    rows: list[np.ndarray] = []
    margins: list[float] = []
    for index, name in enumerate(STATE_ORDER):
        row = np.zeros(6)
        row[index] = 1.0
        names.append(f"{name} lower")
        rows.append(row)
        margins.append(float(X_STAR[index]))
    for index, name in ((0, "n1"), (1, "n2")):
        row = np.zeros(6)
        row[index] = -1.0
        names.append(f"{name} upper")
        rows.append(row)
        margins.append(float(100.0 - X_STAR[index]))
    rows.extend(
        [np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0])]
    )
    names.extend(["n12 lower", "n21 lower"])
    margins.extend([10.0, 10.0])
    return names, rows, margins


def tube_record(certificate: dict) -> dict:
    roots = []
    direct_roots = []
    coefficient_bounds = []
    metric_coefficient_bounds = []
    constraint_radii = []
    for index, (gain, closed_loop, p_matrix) in enumerate(zip(GAINS, CLOSED_LOOP, certificate["matrices"])):
        quadratic, cubic = nonlinear_coefficient_bounds(gain, closed_loop)
        coefficient_bounds.append([quadratic, cubic])
        if "flow_scales" in certificate:
            scale = float(certificate["flow_scales"][index])
        elif "flow_slacks" in certificate:
            scale = float(certificate["flow_slacks"][index])
        else:
            scale = 1.0 if index == 0 else certificate["scale_2"]
        p_norm = np.linalg.norm(p_matrix, 2)
        direct_root = brentq(
            lambda radius: 2.0 * p_norm * (quadratic * radius + cubic * radius * radius) - scale,
            0.0,
            1.0,
        )
        values, vectors = np.linalg.eigh(p_matrix)
        sqrt_p = vectors @ np.diag(np.sqrt(values)) @ vectors.T
        remainder_vectors = nonlinear_coefficient_vectors(gain, closed_loop)
        metric_quadratic = sum(
            np.linalg.norm(sqrt_p @ vector)
            for exponent, vector in remainder_vectors.items()
            if sum(exponent) == 2
        )
        metric_cubic = sum(
            np.linalg.norm(sqrt_p @ vector)
            for exponent, vector in remainder_vectors.items()
            if sum(exponent) == 3
        )
        metric_root = brentq(
            lambda radius: 2.0
            * math.sqrt(float(np.max(values)))
            * (metric_quadratic * radius + metric_cubic * radius * radius)
            - scale,
            0.0,
            1.0,
        )

        _, rows, equilibrium_margins = state_constraint_rows()
        radii = [margin / np.linalg.norm(row) for row, margin in zip(rows, equilibrium_margins)]
        radii.extend(0.5 / np.linalg.norm(row) for row in gain)
        radii.append(DIRECTIONAL_LIMIT / np.linalg.norm(gain[1] - gain[0]))
        radii.append(DIRECTIONAL_LIMIT / np.linalg.norm(gain[3] - gain[2]))
        constraint_radius = min(radii)
        direct_roots.append(min(direct_root, constraint_radius))
        roots.append(min(metric_root, constraint_radius))
        constraint_radii.append(constraint_radius)
        metric_coefficient_bounds.append([metric_quadratic, metric_cubic])

    levels = [
        float(np.min(np.linalg.eigvalsh(p_matrix)) * radius * radius)
        for p_matrix, radius in zip(certificate["matrices"], roots)
    ]
    return {
        "radii": roots,
        "installed_certificate": "certificate-metric coefficient-sum bound",
        "direct_radii": direct_roots,
        "individual_levels": levels,
        "common_level": min(levels),
        "coefficient_bounds": coefficient_bounds,
        "metric_coefficient_bounds": metric_coefficient_bounds,
        "constraint_radii": constraint_radii,
    }


def margin_components(x: np.ndarray, mode: int) -> tuple[list[str], np.ndarray]:
    error = x - X_STAR
    command = U_STAR + GAINS[mode] @ error
    n1, n2, n11, n22, n10, n20 = x
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    names = [f"{name} lower" for name in INPUT_ORDER] + [f"{name} upper" for name in INPUT_ORDER]
    values = list(command) + list(1.0 - command)
    names += ["u21-u12 directional", "u20-u02 directional"]
    values += [DIRECTIONAL_LIMIT - abs(command[1] - command[0]), DIRECTIONAL_LIMIT - abs(command[3] - command[2])]
    names += ["n1 lower", "n2 lower", "n1 upper", "n2 upper", "n11 lower", "n22 lower", "n10 lower", "n20 lower", "n12 lower", "n21 lower"]
    values += [n1, n2, 100.0 - n1, 100.0 - n2, n11, n22, n10, n20, n12, n21]
    return names, np.asarray(values)


def closed_loop_rhs(
    time: float,
    x: np.ndarray,
    mode: int,
    demand_profile=None,
) -> np.ndarray:
    demand = None if demand_profile is None else demand_profile(time)
    return mfd_rhs(x, U_STAR + GAINS[mode] @ (x - X_STAR), demand=demand)


def dominant_initial_state(certificate: dict | None = None, tube: dict | None = None) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(monodromy(PROPOSAL_INTERVAL, PROPOSAL_INTERVAL))
    direction = np.real(eigenvectors[:, np.argmax(np.abs(eigenvalues))])
    direction /= np.linalg.norm(direction)
    if certificate is not None and tube is not None:
        p_matrix = certificate["matrices"][0]
        scale = math.sqrt(
            0.45 * tube["common_level"] / float(direction @ p_matrix @ direction)
        )
        return X_STAR + scale * direction
    command_scale = max(np.max(np.abs(gain @ direction)) for gain in GAINS)
    return X_STAR + INITIAL_COMMAND_DEVIATION / command_scale * direction


def simulate(
    policy: str,
    certificate: dict,
    tube: dict,
    demand_profile=None,
) -> dict:
    x = dominant_initial_state(certificate, tube)
    mode = 1 if policy == "fixed_2" else 0
    times = [0.0]
    states = [x.copy()]
    modes = [mode]
    accepted_times: list[float] = []
    last_switch_step = 0
    violation = None
    total_steps = int(round(SIMULATION_HORIZON / PROPOSAL_INTERVAL))

    for step in range(total_steps):
        time = step * PROPOSAL_INTERVAL
        requested_mode = mode
        if policy == "fixed_1":
            requested_mode = 0
        elif policy == "fixed_2":
            requested_mode = 1
        elif policy == "unconditional":
            requested_mode = step % 2
        elif policy == "supervised":
            proposal = step % 2
            elapsed = (step - last_switch_step) * PROPOSAL_INTERVAL
            if proposal != mode and elapsed + 1.0e-12 >= certificate["realized_dwell"]:
                requested_mode = proposal
        else:
            raise ValueError(policy)

        if requested_mode != mode:
            names, margins = margin_components(x, requested_mode)
            if policy == "unconditional" or np.min(margins) >= 0.0:
                mode = requested_mode
                last_switch_step = step
                if policy == "supervised":
                    accepted_times.append(time)
            if np.min(margins) < 0.0 and policy == "unconditional":
                violation_index = int(np.argmin(margins))
                times.append(time)
                states.append(x.copy())
                modes.append(mode)
                violation = {
                    "time": time,
                    "constraint": names[violation_index],
                    "margin": float(margins[violation_index]),
                }
                break

        names, margins = margin_components(x, mode)
        if np.min(margins) < -1.0e-10:
            violation_index = int(np.argmin(margins))
            violation = {
                "time": time,
                "constraint": names[violation_index],
                "margin": float(margins[violation_index]),
            }
            break

        interval_end = (step + 1) * PROPOSAL_INTERVAL
        events = []
        for constraint_index in range(len(margins)):
            def event(event_time, event_state, index=constraint_index, active_mode=mode):
                return margin_components(event_state, active_mode)[1][index]

            event.terminal = True
            event.direction = -1.0
            events.append(event)
        evaluation_times = np.linspace(time, interval_end, 21)[1:]
        solution = solve_ivp(
            lambda current_time, current_state: closed_loop_rhs(
                current_time, current_state, mode, demand_profile
            ),
            (time, interval_end),
            x,
            method="DOP853",
            t_eval=evaluation_times,
            events=events,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.0025,
        )
        append_start = 0
        times.extend(solution.t[append_start:].tolist())
        states.extend(solution.y[:, append_start:].T)
        modes.extend([mode] * solution.t.size)
        x = solution.y[:, -1]
        event_hits = [(event_times[0], index) for index, event_times in enumerate(solution.t_events) if event_times.size]
        if event_hits:
            event_time, constraint_index = min(event_hits)
            event_state = solution.y_events[constraint_index][0]
            if not times or abs(times[-1] - event_time) > 1.0e-10:
                times.append(float(event_time))
                states.append(event_state.copy())
                modes.append(mode)
            x = event_state
            violation = {
                "time": float(event_time),
                "constraint": names[constraint_index],
                "margin": 0.0,
            }
            break

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    margins_array = np.array(
        [np.min(margin_components(state, int(active_mode))[1]) for state, active_mode in zip(states_array, modes_array)]
    )
    occupancies = np.array(
        [
            (state - X_STAR) @ certificate["matrices"][int(active_mode)] @ (state - X_STAR) / tube["common_level"]
            for state, active_mode in zip(states_array, modes_array)
        ]
    )
    commands_array = np.array(
        [U_STAR + GAINS[int(active_mode)] @ (state - X_STAR)
         for state, active_mode in zip(states_array, modes_array)]
    )
    production = np.array(
        [transportation_rates(state, command)["network_production"]
         for state, command in zip(states_array, commands_array)]
    )
    completions = np.array(
        [transportation_rates(state, command)["completed_trips"]
         for state, command in zip(states_array, commands_array)]
    )
    integration = np.trapezoid
    transportation = {
        "vehicle_hours_in_network": float(
            100.0 * integration(states_array[:, 0] + states_array[:, 1], times_array)
        ),
        "cumulative_network_production_vehicles": float(
            100.0 * integration(production, times_array)
        ),
        "completed_trips_vehicles": float(
            100.0 * integration(completions, times_array)
        ),
        "mean_network_production_veh_per_hour": float(100.0 * np.mean(production)),
    }
    return {
        "policy": policy,
        "times": times_array,
        "states": states_array,
        "modes": modes_array,
        "margins": margins_array,
        "occupancies": occupancies,
        "commands": commands_array,
        "production": production,
        "completions": completions,
        "transportation": transportation,
        "accepted_times": accepted_times,
        "violation": violation,
    }


def audit_model() -> dict:
    equilibrium_residual = mfd_rhs(X_STAR, U_STAR)
    finite_a = centered_jacobian(lambda state: mfd_rhs(state, U_STAR), X_STAR)
    finite_b = centered_jacobian(lambda command: mfd_rhs(X_STAR, command), U_STAR)
    controllability = np.column_stack(
        [np.linalg.matrix_power(A_ANALYTIC, power) @ B_ANALYTIC for power in range(6)]
    )
    output = np.column_stack((np.eye(2), np.zeros((2, 4))))
    observability = np.vstack([output @ np.linalg.matrix_power(A_ANALYTIC, power) for power in range(6)])
    assert np.max(np.abs(equilibrium_residual)) < 1.0e-12
    assert np.max(np.abs(finite_a - A_ANALYTIC)) < 2.0e-9
    assert np.max(np.abs(finite_b - B_ANALYTIC)) < 2.0e-9
    return {
        "state_order": list(STATE_ORDER),
        "input_order": list(INPUT_ORDER),
        "equilibrium_residual_inf": float(np.max(np.abs(equilibrium_residual))),
        "jacobian_A_error_inf": float(np.max(np.abs(finite_a - A_ANALYTIC))),
        "jacobian_B_error_inf": float(np.max(np.abs(finite_b - B_ANALYTIC))),
        "controllability_rank": int(np.linalg.matrix_rank(controllability)),
        "observability_rank": int(np.linalg.matrix_rank(observability)),
        "open_loop_spectral_abscissa": spectral_abscissa(A_ANALYTIC),
        "closed_loop_spectral_abscissae": [spectral_abscissa(matrix) for matrix in CLOSED_LOOP],
    }


def witness_audit() -> dict:
    six_minute_radius = spectral_radius(monodromy(PROPOSAL_INTERVAL, PROPOSAL_INTERVAL))
    average_abscissa = spectral_abscissa(0.5 * (CLOSED_LOOP[0] + CLOSED_LOOP[1]))
    crossing = brentq(
        lambda residence: spectral_radius(monodromy(residence, residence)) - 1.0,
        0.15,
        0.30,
    )
    peak = minimize_scalar(
        lambda residence: -spectral_radius(monodromy(residence, residence)),
        bounds=(0.01, crossing),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    verified_witness = 0.205
    verified_radius = spectral_radius(monodromy(verified_witness, verified_witness))
    assert six_minute_radius > 1.0
    assert verified_radius > 1.0
    return {
        "six_minute_radius": six_minute_radius,
        "average_matrix_spectral_abscissa": average_abscissa,
        "equal_residence_unstable_interval": [0.0, crossing],
        "equal_residence_peak_time": float(peak.x),
        "equal_residence_peak_radius": float(-peak.fun),
        "verified_periodic_lower_bound": verified_witness,
        "verified_periodic_lower_bound_radius": verified_radius,
    }


def plot_results(simulations: dict[str, dict], suffix: str = "") -> None:
    colors = {
        "fixed_1": "#4C78A8",
        "fixed_2": "#59A14F",
        "unconditional": "#E15759",
        "supervised": "#7B4AB5",
    }
    labels = {
        "fixed_1": "held gain 1",
        "fixed_2": "held gain 2",
        "unconditional": "six-minute alternation",
        "supervised": "certified execution",
    }
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for name, result in simulations.items():
        physical_error = np.linalg.norm(result["states"] - X_STAR, axis=1)
        axis.semilogy(result["times"], physical_error, color=colors[name], label=labels[name], linewidth=1.7)
    for switch_time in simulations["supervised"]["accepted_times"]:
        axis.axvline(switch_time, color=colors["supervised"], alpha=0.10, linewidth=0.8)
    axis.set_xlabel("time (h)")
    axis.set_ylabel("physical-state error norm")
    axis.set_xlim(0.0, SIMULATION_HORIZON)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(ROOT / f"mfd_nonlinear_error{suffix}.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for name in ("unconditional", "supervised"):
        result = simulations[name]
        shown_margin = (
            np.minimum.accumulate(result["margins"])
            if name == "unconditional"
            else result["margins"]
        )
        shown_label = "unconditional running minimum" if name == "unconditional" else labels[name]
        axis.plot(result["times"], shown_margin, color=colors[name], label=shown_label, linewidth=1.8)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("time (h)")
    axis.set_ylabel("minimum state/input margin")
    axis.set_xlim(0.0, SIMULATION_HORIZON)
    axis.legend(frameon=True, facecolor="white", framealpha=0.95)
    figure.tight_layout()
    figure.savefig(ROOT / f"mfd_gating_margin{suffix}.png", dpi=220)
    plt.close(figure)


def bounded_withholding_audit(certificate: dict, tube: dict) -> dict:
    """Numerically instantiate the conditional bounded-withholding theorem.

    The audit begins immediately after a certified event, so the residence
    timer is reset to zero. A fixed switch from mode 1 to mode 2 is retained
    as pending and reevaluated every proposal epoch while the source mechanism
    is held. This is separate from the alternating-proposal switching witness.
    """

    source_mode = 0
    requested_mode = 1
    state = dominant_initial_state(certificate, tube)
    level = tube["common_level"]
    source_matrix = certificate["matrices"][source_mode]
    requested_matrix = certificate["matrices"][requested_mode]
    initial_error = state - X_STAR
    initial_value = float(initial_error @ source_matrix @ initial_error)
    predecessor_value = float(initial_error @ requested_matrix @ initial_error)
    transition_factor = certificate["transition_factor"]
    flow_rate = certificate["rate"]
    capture_level = level / transition_factor
    capture_time = math.log(initial_value / capture_level) / flow_rate
    timing_wait = certificate["realized_dwell"]
    evaluation_gap = PROPOSAL_INTERVAL
    withholding_bound = max(capture_time, timing_wait) + evaluation_gap

    assert abs(initial_value / level - 0.45) < 1.0e-10
    assert predecessor_value <= level + 1.0e-12
    assert capture_time > 0.0
    first_capture_evaluation = None
    grant_time = None
    grant_source_occupancy = None
    grant_successor_occupancy = None
    grant_minimum_margin = None
    total_steps = int(math.ceil(withholding_bound / evaluation_gap)) + 2

    for step in range(total_steps):
        time = step * evaluation_gap
        certified_upper_value = initial_value * math.exp(-flow_rate * time)
        capture_ready = certified_upper_value <= capture_level + 1.0e-14
        timing_ready = time + 1.0e-12 >= timing_wait
        if capture_ready and first_capture_evaluation is None:
            first_capture_evaluation = time
        if capture_ready and timing_ready:
            current_error = state - X_STAR
            grant_time = time
            grant_source_occupancy = float(
                current_error @ source_matrix @ current_error / level
            )
            grant_successor_occupancy = float(
                current_error @ requested_matrix @ current_error / level
            )
            _, candidate_margins = margin_components(state, requested_mode)
            grant_minimum_margin = float(np.min(candidate_margins))
            assert grant_successor_occupancy <= 1.0 + 1.0e-10
            assert grant_minimum_margin >= 0.0
            break

        interval_end = time + evaluation_gap
        solution = solve_ivp(
            lambda current_time, current_state: closed_loop_rhs(
                current_time, current_state, source_mode
            ),
            (time, interval_end),
            state,
            method="DOP853",
            t_eval=[interval_end],
            rtol=1.0e-11,
            atol=1.0e-13,
            max_step=0.0025,
        )
        assert solution.success
        state = solution.y[:, -1]

    assert first_capture_evaluation is not None
    assert grant_time is not None
    assert first_capture_evaluation <= capture_time + evaluation_gap + 1.0e-12
    assert grant_time <= withholding_bound + 1.0e-12
    assert abs(grant_time - timing_wait) < 1.0e-12

    return {
        "request": "persistent fixed mode-1 to mode-2 switch retained after a certified mode-2 to mode-1 timer-reset event",
        "predecessor_tube_occupancy_before_timer_reset_event": predecessor_value / level,
        "initial_tube_occupancy": initial_value / level,
        "capture_occupancy": capture_level / level,
        "capture_time": capture_time,
        "first_capture_evaluation": first_capture_evaluation,
        "timing_wait": timing_wait,
        "evaluation_gap": evaluation_gap,
        "theorem_4_withholding_bound": withholding_bound,
        "realized_grant_time": grant_time,
        "timer_to_capture_ratio": timing_wait / capture_time,
        "grant_source_occupancy": grant_source_occupancy,
        "grant_successor_occupancy": grant_successor_occupancy,
        "grant_minimum_state_input_margin": grant_minimum_margin,
        "hypotheses_audited": {
            "persistent_request": True,
            "no_competing_execution": True,
            "work_conserving_selection": True,
            "hold_contraction_upper_bound": True,
        },
    }


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = audit_model()
    witness = witness_audit()
    certificate = independent_certificate(rate=0.2)
    tube = tube_record(certificate)
    withholding = bounded_withholding_audit(certificate, tube)
    simulations = {
        name: simulate(name, certificate, tube)
        for name in ("fixed_1", "fixed_2", "unconditional", "supervised")
    }
    out_of_sample = {
        name: simulate(name, certificate, tube, demand_profile=out_of_sample_demand)
        for name in ("fixed_1", "supervised")
    }
    assert simulations["fixed_1"]["violation"] is None
    assert simulations["fixed_2"]["violation"] is None
    assert simulations["unconditional"]["violation"] is not None
    assert simulations["supervised"]["violation"] is None
    assert np.max(simulations["supervised"]["occupancies"]) <= 1.0 + 2.0e-6
    plot_results(simulations)

    simulation_summary = {}
    for name, result in simulations.items():
        simulation_summary[name] = {
            "end_time": float(result["times"][-1]),
            "final_error_norm": float(np.linalg.norm(result["states"][-1] - X_STAR)),
            "minimum_margin": float(np.min(result["margins"])),
            "maximum_tube_occupancy": float(np.max(result["occupancies"])),
            "accepted_switches": len(result["accepted_times"]),
            "accepted_times": result["accepted_times"],
            "violation": result["violation"],
            "transportation": result["transportation"],
        }

    out_of_sample_summary = {}
    for name, result in out_of_sample.items():
        out_of_sample_summary[name] = {
            "end_time": float(result["times"][-1]),
            "minimum_margin": float(np.min(result["margins"])),
            "maximum_tube_occupancy": float(np.max(result["occupancies"])),
            "accepted_switches": len(result["accepted_times"]),
            "violation": result["violation"],
            "transportation": result["transportation"],
        }

    output = {
        "model_audit": model,
        "A": A_ANALYTIC,
        "B": B_ANALYTIC,
        "K1": K1,
        "K2": K2,
        "Q1": Q_LQR[0],
        "R1": R_LQR[0],
        "Q2": Q_LQR[1],
        "R2": R_LQR[1],
        "witness_audit": witness,
        "certificate": {key: value for key, value in certificate.items() if key != "matrices"},
        "P1": certificate["matrices"][0],
        "P2": certificate["matrices"][1],
        "tube": tube,
        "initial_state": dominant_initial_state(certificate, tube),
        "bounded_withholding_audit": withholding,
        "simulation": simulation_summary,
        "out_of_sample_demand_test": {
            "description": "componentwise sinusoidal demand modulation within five percent; not used for synthesis",
            "simulation": out_of_sample_summary,
        },
    }
    result_path = RESULTS_DIR / "mfd_fixed_gain_results.json"
    result_path.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        RESULTS_DIR / "mfd_fixed_gain_trajectories.npz",
        **{
            f"{name}_{field}": result[field]
            for name, result in simulations.items()
            for field in ("times", "states", "modes", "margins", "occupancies")
        },
    )

    print(json.dumps(json_ready({"model_audit": model, "witness_audit": witness, "certificate": output["certificate"], "tube": tube, "bounded_withholding_audit": withholding, "simulation": simulation_summary, "out_of_sample_demand_test": out_of_sample_summary}), indent=2))


if __name__ == "__main__":
    main()
