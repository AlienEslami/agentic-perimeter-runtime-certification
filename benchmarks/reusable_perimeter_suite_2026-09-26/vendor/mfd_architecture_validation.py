#!/usr/bin/env python3
"""Audit the output-feedback P/PI architecture on the literature-cubic MFD.

The physical and observer errors use the canonical order
    [n1, n2, n11, n22, n10, n20].
Only n1 and n2 are measured.  The script constructs the observer and P/PI
fibers, certifies local nonlinear tubes, evaluates information-set landing
tests exactly, and checks a common-matrix projected-adaptation record.
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
from scipy.linalg import eigvalsh, solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import brentq

import mfd_cubic_pilot as mfd


A = mfd.A
B = mfd.B
DIRECTIONAL_LIMIT = mfd.DIRECTIONAL_LIMIT
INPUT_ORDER = mfd.INPUT_ORDER
STATE_ORDER = mfd.STATE_ORDER
U_STAR = mfd.U_STAR
X_STAR = mfd.X_STAR
centered_jacobian = mfd.centered_jacobian
mfd_rhs = mfd.mfd_rhs


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
C = np.column_stack((np.eye(2), np.zeros((2, 4))))
FLOW_RATE = 0.10
ADAPTATION_FLOW_RATE = 0.05
THETA_INTERVAL = (-0.10, 0.10)
THETA_RATE_CAP = 0.02  # per hour; an authority cap, not needed by common-P stability
P_TO_PI_HEADROOM = 0.98
# The architecture transfer and two later adaptation requests isolate the
# information-set landing predicate; this pathwise experiment has no dwell
# requirement because it contains one declared architecture transition.
P_TO_PI_TIME = 1.0
COARSE_ADAPTATION_REQUEST_TIME = 8.0
REFINED_ADAPTATION_REQUEST_TIME = 16.0
FINAL_TIME = 24.0


def spectral_abscissa(matrix: np.ndarray) -> float:
    return float(np.max(np.real(np.linalg.eigvals(matrix))))


def synthesize() -> dict:
    """Create one proportional gain shared by the P and PI fibers."""
    a_aug = np.block([[A, np.zeros((6, 2))], [C, np.zeros((2, 2))]])
    b_aug = np.vstack((B, np.zeros((2, 4))))
    q_aug = np.diag([1.0] * 6 + [0.1] * 2)
    r_aug = 10.0 * np.eye(4)
    p_lqr = solve_continuous_are(a_aug, b_aug, q_aug, r_aug)
    gain_aug = -np.linalg.solve(r_aug, b_aug.T @ p_lqr)
    kp, ki = gain_aug[:, :6], gain_aug[:, 6:]

    p_obs = solve_continuous_are(A.T, C.T, np.eye(6), np.eye(2))
    observer_gain = p_obs @ C.T
    ao = A - observer_gain @ C

    ap = np.block(
        [
            [A + B @ kp, -B @ kp],
            [np.zeros((6, 6)), ao],
        ]
    )
    api = np.block(
        [
            [A + B @ kp, -B @ kp, B @ ki],
            [np.zeros((6, 6)), ao, np.zeros((6, 2))],
            [C, -C, np.zeros((2, 2))],
        ]
    )
    assert spectral_abscissa(ao) < 0.0
    assert spectral_abscissa(ap) < 0.0
    assert spectral_abscissa(api) < 0.0
    return {"Kp": kp, "Ki": ki, "L": observer_gain, "Ao": ao, "AP": ap, "API": api}


# Small multivariate-polynomial implementation.  It is used only to obtain a
# rigorous coefficient-sum bound on the exact degree-two through degree-four
# remainder induced by the cubic MFD and affine feedback.
Exponent = tuple[int, ...]
Polynomial = dict[Exponent, float]


def pclean(poly: Polynomial) -> Polynomial:
    return {key: value for key, value in poly.items() if abs(value) > 1.0e-14}


def padd(left: Polynomial, right: Polynomial) -> Polynomial:
    out = left.copy()
    for key, value in right.items():
        out[key] = out.get(key, 0.0) + value
    return pclean(out)


def pscale(poly: Polynomial, scale: float) -> Polynomial:
    return pclean({key: scale * value for key, value in poly.items()})


def psub(left: Polynomial, right: Polynomial) -> Polynomial:
    return padd(left, pscale(right, -1.0))


def pmul(left: Polynomial, right: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for kl, vl in left.items():
        for kr, vr in right.items():
            key = tuple(a + b for a, b in zip(kl, kr))
            out[key] = out.get(key, 0.0) + vl * vr
    return pclean(out)


def psquare(poly: Polynomial) -> Polynomial:
    return pmul(poly, poly)


def pconst(value: float, dimension: int) -> Polynomial:
    return {(0,) * dimension: value}


def paffine(offset: float, row: np.ndarray) -> Polynomial:
    dimension = row.size
    out = pconst(offset, dimension)
    for index, value in enumerate(row):
        if abs(value) > 0.0:
            key = [0] * dimension
            key[index] = 1
            out[tuple(key)] = float(value)
    return pclean(out)


def physical_polynomials(command_rows: np.ndarray) -> list[Polynomial]:
    """Exact MFD physical RHS as polynomials of the active analytical state."""
    dimension = command_rows.shape[1]
    state_rows = np.zeros((6, dimension))
    state_rows[:, :6] = np.eye(6)
    n1, n2, n11, n22, n10, n20 = [
        paffine(X_STAR[index], state_rows[index]) for index in range(6)
    ]
    n12 = psub(psub(n1, n11), n10)
    n21 = psub(psub(n2, n22), n20)
    u12, u21, u02, u20 = [
        paffine(U_STAR[index], command_rows[index]) for index in range(4)
    ]
    factor1 = padd(
        padd(pscale(psquare(n1), mfd.MFD_A), pscale(n1, mfd.MFD_B)),
        pconst(mfd.MFD_C, dimension),
    )
    factor2 = padd(
        padd(pscale(psquare(n2), mfd.MFD_A), pscale(n2, mfd.MFD_B)),
        pconst(mfd.MFD_C, dimension),
    )
    q11, q12, q10 = [pmul(value, factor1) for value in (n11, n12, n10)]
    q22, q21, q20 = [pmul(value, factor2) for value in (n22, n21, n20)]
    d11, d12, d10, d22, d21, d20 = mfd.DEMAND
    d1 = d11 + d12 + d10
    d2 = d22 + d21 + d20

    dn11 = padd(padd(pscale(q11, -1.0), pmul(q21, u21)), pconst(d11, dimension))
    dn22 = padd(
        padd(padd(pscale(q22, -1.0), pmul(q12, u12)), pscale(u02, 2.0)),
        pconst(d22, dimension),
    )
    dn10 = padd(pscale(pmul(q10, u12), -1.0), pconst(d10, dimension))
    dn20 = padd(
        padd(pscale(pmul(q20, u20), -1.0), pmul(q10, u12)),
        pconst(d20, dimension),
    )
    dn1 = padd(
        padd(
            padd(pscale(q11, -1.0), pscale(pmul(padd(q12, q10), u12), -1.0)),
            pmul(q21, u21),
        ),
        pconst(d1, dimension),
    )
    dn2 = padd(
        padd(
            padd(
                padd(
                    padd(pscale(q22, -1.0), pmul(padd(q12, q10), u12)),
                    pscale(pmul(q21, u21), -1.0),
                ),
                pscale(u02, mfd.Q01 + mfd.Q02),
            ),
            pscale(pmul(q20, u20), -1.0),
        ),
        pconst(d2, dimension),
    )
    return [dn1, dn2, dn11, dn22, dn10, dn20]


def coefficient_bounds(command_rows: np.ndarray) -> tuple[float, float, float]:
    """Bound the augmented remainder R_aug=[R;R;0] coefficient by coefficient."""
    dimension = command_rows.shape[1]
    coefficients: dict[Exponent, np.ndarray] = {}
    for output, poly in enumerate(physical_polynomials(command_rows)):
        for exponent, value in poly.items():
            if sum(exponent) >= 2:
                coefficients.setdefault(exponent, np.zeros(dimension))[output] = value
                coefficients[exponent][6 + output] = value
    return tuple(
        float(sum(np.linalg.norm(value) for key, value in coefficients.items() if sum(key) == degree))
        for degree in (2, 3, 4)
    )


def state_constraint_rows(dimension: int) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    margins: list[float] = []
    for index in range(6):
        row = np.zeros(dimension)
        row[index] = 1.0
        rows.append(row)
        margins.append(float(X_STAR[index]))
    for index in (0, 1):
        row = np.zeros(dimension)
        row[index] = -1.0
        rows.append(row)
        margins.append(float(100.0 - X_STAR[index]))
    row12 = np.zeros(dimension)
    row12[[0, 2, 4]] = [1.0, -1.0, -1.0]
    row21 = np.zeros(dimension)
    row21[[1, 3, 5]] = [1.0, -1.0, -1.0]
    rows.extend((row12, row21))
    margins.extend((10.0, 10.0))
    return rows, margins


def flow_record(matrix: np.ndarray, command_rows: np.ndarray, rate: float) -> dict:
    shifted = matrix + 0.5 * rate * np.eye(matrix.shape[0])
    p_matrix = solve_continuous_lyapunov(shifted.T, -np.eye(matrix.shape[0]))
    residual = matrix.T @ p_matrix + p_matrix @ matrix + rate * p_matrix
    linear_slack = -float(np.max(eigvalsh(residual)))
    assert linear_slack > 0.0
    c2, c3, c4 = coefficient_bounds(command_rows)
    pnorm = float(np.linalg.norm(p_matrix, 2))
    nonlinear_radius = brentq(
        lambda radius: 2.0 * pnorm * (
            c2 * radius + c3 * radius**2 + c4 * radius**3
        ) - linear_slack,
        0.0,
        1.0,
    )
    rows, margins = state_constraint_rows(matrix.shape[0])
    constraint_radii = [margin / np.linalg.norm(row) for row, margin in zip(rows, margins)]
    constraint_radii.extend(0.5 / np.linalg.norm(row) for row in command_rows)
    constraint_radii.append(DIRECTIONAL_LIMIT / np.linalg.norm(command_rows[1] - command_rows[0]))
    constraint_radii.append(DIRECTIONAL_LIMIT / np.linalg.norm(command_rows[3] - command_rows[2]))
    constraint_radius = float(min(constraint_radii))
    radius = min(nonlinear_radius, constraint_radius)
    level = float(np.min(eigvalsh(p_matrix)) * radius * radius)
    return {
        "P": p_matrix,
        "rate": rate,
        "linear_slack": linear_slack,
        "coefficient_bounds": [c2, c3, c4],
        "nonlinear_radius": nonlinear_radius,
        "constraint_radius": constraint_radius,
        "radius": radius,
        "level": level,
    }


def adaptation_matrix(design: dict, theta: float) -> tuple[np.ndarray, np.ndarray]:
    kp = (1.0 + theta) * design["Kp"]
    ki, ao = design["Ki"], design["Ao"]
    matrix = np.block(
        [
            [A + B @ kp, -B @ kp, B @ ki],
            [np.zeros((6, 6)), ao, np.zeros((6, 2))],
            [C, -C, np.zeros((2, 2))],
        ]
    )
    command_rows = np.column_stack((kp, -kp, ki))
    return matrix, command_rows


def adaptation_record(design: dict, p_common: np.ndarray) -> dict:
    endpoint_records = []
    for theta in THETA_INTERVAL:
        matrix, command_rows = adaptation_matrix(design, theta)
        residual = matrix.T @ p_common + p_common @ matrix + ADAPTATION_FLOW_RATE * p_common
        c2, c3, c4 = coefficient_bounds(command_rows)
        endpoint_records.append(
            {
                "theta": theta,
                "spectral_abscissa": spectral_abscissa(matrix),
                "linear_slack": -float(np.max(eigvalsh(residual))),
                "coefficient_bounds": [c2, c3, c4],
            }
        )
    slack = min(record["linear_slack"] for record in endpoint_records)
    c2 = max(record["coefficient_bounds"][0] for record in endpoint_records)
    c3 = max(record["coefficient_bounds"][1] for record in endpoint_records)
    c4 = max(record["coefficient_bounds"][2] for record in endpoint_records)
    radius = brentq(
        lambda value: 2.0 * np.linalg.norm(p_common, 2) * (
            c2 * value + c3 * value**2 + c4 * value**3
        ) - slack,
        0.0,
        1.0,
    )
    level = float(np.min(eigvalsh(p_common)) * radius * radius)

    grid = np.linspace(*THETA_INTERVAL, 401)
    worst_residual = -math.inf
    worst_abscissa = -math.inf
    for theta in grid:
        matrix, _ = adaptation_matrix(design, float(theta))
        residual = matrix.T @ p_common + p_common @ matrix + ADAPTATION_FLOW_RATE * p_common
        worst_residual = max(worst_residual, float(np.max(eigvalsh(residual))))
        worst_abscissa = max(worst_abscissa, spectral_abscissa(matrix))
    assert worst_residual < 0.0
    return {
        "theta_interval": THETA_INTERVAL,
        "theta_rate_cap_per_hour": THETA_RATE_CAP,
        "rate_cap_needed_by_stability_proof": False,
        "flow_rate": ADAPTATION_FLOW_RATE,
        "endpoint_records": endpoint_records,
        "worst_grid_flow_residual": worst_residual,
        "worst_grid_spectral_abscissa": worst_abscissa,
        "radius": radius,
        "level": level,
    }


def trust_region_max(
    b_active: np.ndarray,
    m_active: np.ndarray,
    p_active: np.ndarray,
    level_active: float,
    b_successor: np.ndarray,
    m_successor: np.ndarray,
    p_successor: np.ndarray,
) -> float:
    """Exact maximum of successor quadratic over an affine ellipsoidal slice."""
    q = m_active.T @ p_active @ m_active
    linear = m_active.T @ p_active @ b_active
    center = -np.linalg.solve(q, linear)
    radius_squared = float(
        level_active - b_active @ p_active @ b_active + linear @ np.linalg.solve(q, linear)
    )
    if radius_squared < -1.0e-12:
        raise ValueError("The information set is empty")
    radius_squared = max(radius_squared, 0.0)
    values, vectors = np.linalg.eigh(q)
    q_inverse_sqrt = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    y_center = b_successor + m_successor @ center
    direction = math.sqrt(radius_squared) * m_successor @ q_inverse_sqrt
    hessian = direction.T @ p_successor @ direction
    gradient = direction.T @ p_successor @ y_center
    constant = float(y_center @ p_successor @ y_center)
    if np.linalg.norm(gradient) < 1.0e-12 * (1.0 + abs(constant)):
        return constant + float(np.max(eigvalsh(hessian)))
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    largest = float(eigenvalues[-1])

    def secular(multiplier: float) -> float:
        z = np.linalg.solve(multiplier * np.eye(gradient.size) - hessian, gradient)
        return float(z @ z - 1.0)

    lower = largest + max(1.0e-13, abs(largest) * 1.0e-13)
    if secular(lower) <= 0.0:
        # Trust-region ``hard case'': the linear term is orthogonal (up to
        # numerical tolerance) to the dominant eigenspace.  The secular root
        # lies at lambda_max(H), and the remaining norm is placed in that
        # eigenspace.
        transformed_gradient = eigenvectors.T @ gradient
        transformed_solution = np.zeros_like(transformed_gradient)
        gap = largest - eigenvalues
        regular = gap > max(1.0e-12, 1.0e-10 * max(1.0, abs(largest)))
        transformed_solution[regular] = transformed_gradient[regular] / gap[regular]
        remaining = max(0.0, 1.0 - float(transformed_solution @ transformed_solution))
        transformed_solution[-1] = math.sqrt(remaining)
        z = eigenvectors @ transformed_solution
        return float(z @ hessian @ z + 2.0 * gradient @ z + constant)
    upper = max(1.0, 2.0 * lower)
    while secular(upper) > 0.0:
        upper *= 2.0
    multiplier = brentq(secular, lower, upper, xtol=1.0e-14, rtol=1.0e-14)
    z = np.linalg.solve(multiplier * np.eye(gradient.size) - hessian, gradient)
    return float(z @ hessian @ z + 2.0 * gradient @ z + constant)


def information_affine(
    ehat: np.ndarray, measured_error: np.ndarray, chi: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Affine information set after enforcing the two measured coordinates."""
    fixed_tilde = np.zeros(6)
    fixed_tilde[:2] = measured_error - C @ ehat
    unmeasured = np.zeros((6, 4))
    unmeasured[2:, :] = np.eye(4)
    if chi is None:
        b = np.concatenate((ehat + fixed_tilde, fixed_tilde))
        m = np.vstack((unmeasured, unmeasured))
    else:
        b = np.concatenate((ehat + fixed_tilde, fixed_tilde, chi))
        m = np.vstack((unmeasured, unmeasured, np.zeros((2, 4))))
    return b, m


def ball_information_max(
    b_successor: np.ndarray,
    m_successor: np.ndarray,
    p_successor: np.ndarray,
    radius: float,
) -> float:
    """Maximize a successor certificate over ||xi|| <= radius.

    This represents an implementable set-estimator accuracy record for the four
    unmeasured class coordinates.  The center uses only the observer state and
    measured totals; the true unmeasured error is used only after the decision
    to audit containment.
    """
    return trust_region_max(
        np.zeros(4),
        np.eye(4),
        np.eye(4),
        radius * radius,
        b_successor,
        m_successor,
        p_successor,
    )


def information_radius_threshold(
    b: np.ndarray,
    m: np.ndarray,
    p_matrix: np.ndarray,
    target_level: float,
) -> float:
    """Largest ball radius whose robust quadratic landing value is admissible."""
    center_value = ball_information_max(b, m, p_matrix, 0.0)
    if center_value >= target_level:
        raise ValueError("Even the zero-radius information set fails the target level")
    upper = 1.0e-8
    while ball_information_max(b, m, p_matrix, upper) < target_level:
        upper *= 2.0
        if upper > 10.0:
            raise ValueError("Could not bracket the information-radius threshold")
    return float(
        brentq(
            lambda radius: ball_information_max(b, m, p_matrix, radius) - target_level,
            0.0,
            upper,
            xtol=1.0e-14,
            rtol=1.0e-12,
        )
    )


def augmented_rhs(_time: float, eta: np.ndarray, design: dict, architecture: str) -> np.ndarray:
    e, tilde = eta[:6], eta[6:12]
    ehat = e - tilde
    if architecture == "P":
        chi = np.zeros(2)
    elif architecture == "PI":
        chi = eta[12:14]
    else:
        raise ValueError(architecture)
    command_deviation = design["Kp"] @ ehat + design["Ki"] @ chi
    de = mfd_rhs(X_STAR + e, U_STAR + command_deviation)
    dehat = A @ ehat + B @ command_deviation + design["L"] @ (C @ e - C @ ehat)
    out = [de, de - dehat]
    if architecture == "PI":
        out.append(C @ ehat)
    return np.concatenate(out)


def margins(e: np.ndarray, command: np.ndarray) -> tuple[float, float]:
    x = X_STAR + e
    n1, n2, n11, n22, n10, n20 = x
    state = np.array(
        [n1, n2, mfd.JAM_ACCUMULATION - n1, mfd.JAM_ACCUMULATION - n2, n11, n22, n10, n20,
         n1 - n11 - n10, n2 - n22 - n20]
    )
    input_values = np.concatenate(
        (
            command,
            1.0 - command,
            [DIRECTIONAL_LIMIT - abs(command[1] - command[0])],
            [DIRECTIONAL_LIMIT - abs(command[3] - command[2])],
        )
    )
    return float(np.min(state)), float(np.min(input_values))


def simulate(design: dict, p_record: dict, pi_record: dict, adaptation: dict) -> dict:
    base_e = np.array([1.0, -0.7, 0.4, -0.3, 0.2, -0.1])
    base_tilde = np.array([0.2, -0.1, 0.1, -0.08, 0.05, -0.04])
    base = np.concatenate((base_e, base_tilde))
    scale = math.sqrt(0.25 * p_record["installed_level"] / (base @ p_record["P"] @ base))
    eta0 = scale * base

    first = solve_ivp(
        lambda t, value: augmented_rhs(t, value, design, "P"),
        (0.0, P_TO_PI_TIME),
        eta0,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.0025,
        dense_output=True,
    )
    eta_p = first.y[:, -1]
    ehat_p = eta_p[:6] - eta_p[6:12]
    measured_p = C @ eta_p[:6]
    b_p, m_p = information_affine(ehat_p, measured_p, None)
    b_pi, m_pi = information_affine(ehat_p, measured_p, np.zeros(2))
    landing_max = trust_region_max(
        b_p, m_p, p_record["P"], p_record["installed_level"],
        b_pi, m_pi, pi_record["P"],
    )
    accepted = landing_max <= pi_record["level"] + 1.0e-12
    assert accepted
    eta_pi0 = np.concatenate((eta_p, np.zeros(2)))
    command_before = U_STAR + design["Kp"] @ ehat_p
    command_after = U_STAR + design["Kp"] @ ehat_p + design["Ki"] @ np.zeros(2)

    second = solve_ivp(
        lambda t, value: augmented_rhs(t, value, design, "PI"),
        (P_TO_PI_TIME, REFINED_ADAPTATION_REQUEST_TIME),
        eta_pi0,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.0025,
        dense_output=True,
    )
    adaptation_decisions = []
    for request_index, request_time in enumerate(
        (COARSE_ADAPTATION_REQUEST_TIME, REFINED_ADAPTATION_REQUEST_TIME)
    ):
        eta_request = second.sol(request_time)
        ehat_request = eta_request[:6] - eta_request[6:12]
        measured_request = C @ eta_request[:6]
        b_request, m_request = information_affine(
            ehat_request, measured_request, eta_request[12:14]
        )
        adaptation_threshold = information_radius_threshold(
            b_request, m_request, pi_record["P"], adaptation["level"]
        )
        active_threshold = information_radius_threshold(
            b_request, m_request, pi_record["P"], pi_record["level"]
        )
        true_unmeasured_error = eta_request[8:12]
        true_radius = float(np.linalg.norm(true_unmeasured_error))
        if request_index == 0:
            information_radius = min(
                0.90 * active_threshold,
                max(1.25 * adaptation_threshold, 1.10 * true_radius),
            )
        else:
            information_radius = max(1.10 * true_radius, 0.50 * adaptation_threshold)
        robust_value = ball_information_max(
            b_request, m_request, pi_record["P"], information_radius
        )
        active_value = ball_information_max(
            b_request, m_request, pi_record["P"], information_radius
        )
        accepted_request = robust_value <= adaptation["level"] + 1.0e-12
        assert true_radius <= information_radius + 1.0e-12
        assert active_value <= pi_record["level"] + 1.0e-12
        if request_index == 0:
            assert not accepted_request
        else:
            assert accepted_request
        adaptation_decisions.append(
            {
                "time": request_time,
                "accepted": accepted_request,
                "information_radius": information_radius,
                "adaptation_threshold_radius": adaptation_threshold,
                "active_tube_threshold_radius": active_threshold,
                "true_unmeasured_error_norm_audit_only": true_radius,
                "robust_landing_value": robust_value,
                "robust_landing_ratio": robust_value / adaptation["level"],
                "active_tube_compatibility_ratio": active_value / pi_record["level"],
                "actual_landing_ratio_audit_only": float(
                    eta_request @ pi_record["P"] @ eta_request / adaptation["level"]
                ),
            }
        )

    eta_request = second.y[:, -1]

    third = solve_ivp(
        lambda t, value: augmented_rhs(t, value, design, "PI"),
        (REFINED_ADAPTATION_REQUEST_TIME, FINAL_TIME),
        eta_request,
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.0025,
        dense_output=True,
    )

    t = np.linspace(0.0, FINAL_TIME, 2401)
    eta = np.full((t.size, 14), np.nan)
    p_mask = t <= P_TO_PI_TIME
    mid_mask = (t > P_TO_PI_TIME) & (t <= REFINED_ADAPTATION_REQUEST_TIME)
    end_mask = t > REFINED_ADAPTATION_REQUEST_TIME
    eta[p_mask, :12] = first.sol(t[p_mask]).T
    eta[mid_mask] = second.sol(t[mid_mask]).T
    eta[end_mask] = third.sol(t[end_mask]).T
    eta[np.argmin(abs(t - P_TO_PI_TIME)), 12:] = 0.0

    physical = eta[:, :6]
    estimated = eta[:, :6] - eta[:, 6:12]
    chi = np.nan_to_num(eta[:, 12:14], nan=0.0)
    commands = U_STAR + estimated @ design["Kp"].T + chi @ design["Ki"].T
    state_margins, input_margins = zip(
        *(margins(error, command) for error, command in zip(physical, commands))
    )
    p_occupancy = np.einsum("ni,ij,nj->n", eta[p_mask, :12], p_record["P"], eta[p_mask, :12]) / p_record["installed_level"]
    pi_occupancy = np.einsum("ni,ij,nj->n", eta[~p_mask], pi_record["P"], eta[~p_mask]) / pi_record["level"]
    return {
        "time": t,
        "eta": eta,
        "physical_error": physical,
        "estimated_error": estimated,
        "integrator": chi,
        "commands": commands,
        "initial_scale": scale,
        "p_to_pi": {
            "accepted": accepted,
            "robust_landing_value": landing_max,
            "robust_landing_ratio": landing_max / pi_record["level"],
            "actual_landing_ratio": float(eta_pi0 @ pi_record["P"] @ eta_pi0 / pi_record["level"]),
            "command_jump_inf": float(np.max(np.abs(command_after - command_before))),
        },
        "adaptation_requests": adaptation_decisions,
        "maximum_p_occupancy": float(np.max(p_occupancy)),
        "maximum_pi_occupancy": float(np.max(pi_occupancy)),
        "minimum_state_margin": float(np.min(state_margins)),
        "minimum_input_margin": float(np.min(input_margins)),
        "p_final_error_norm": float(np.linalg.norm(eta_p[:6])),
        "pi_final_error_norm": float(np.linalg.norm(eta[-1, :6])),
    }


def audit_augmented_jacobians(design: dict) -> dict:
    zero_p = np.zeros(12)
    zero_pi = np.zeros(14)
    finite_p = centered_jacobian(lambda value: augmented_rhs(0.0, value, design, "P"), zero_p)
    finite_pi = centered_jacobian(lambda value: augmented_rhs(0.0, value, design, "PI"), zero_pi)
    error_p = float(np.max(np.abs(finite_p - design["AP"])))
    error_pi = float(np.max(np.abs(finite_pi - design["API"])))
    assert error_p < 3.0e-9
    assert error_pi < 3.0e-9
    return {"P_jacobian_error_inf": error_p, "PI_jacobian_error_inf": error_pi}


def plot(simulation: dict) -> None:
    t = simulation["time"]
    physical_norm = np.linalg.norm(simulation["physical_error"], axis=1)
    observer_error_norm = np.linalg.norm(
        simulation["physical_error"] - simulation["estimated_error"], axis=1
    )
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.semilogy(t, physical_norm, linewidth=1.8, label="physical-state error")
    ax.semilogy(t, observer_error_norm, linewidth=1.6, label="observer error")
    ax.axvline(P_TO_PI_TIME, color="#2E8B57", linestyle="--", label="P to PI accepted")
    ax.axvline(COARSE_ADAPTATION_REQUEST_TIME, color="#C44E52", linestyle=":", label="coarse set: rejected")
    ax.axvline(REFINED_ADAPTATION_REQUEST_TIME, color="#2E8B57", linestyle="-.", label="refined set: accepted")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("error norm")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "mfd_architecture_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for index, name in enumerate(INPUT_ORDER):
        ax.plot(t, 1.0e4 * (simulation["commands"][:, index] - U_STAR[index]), label=name)
    ax.axvline(P_TO_PI_TIME, color="#2E8B57", linestyle="--")
    ax.axvline(COARSE_ADAPTATION_REQUEST_TIME, color="#C44E52", linestyle=":")
    ax.axvline(REFINED_ADAPTATION_REQUEST_TIME, color="#2E8B57", linestyle="-.")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("command deviation (times 1e-4)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "mfd_architecture_commands.png", dpi=220)
    plt.close(fig)


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
    design = synthesize()
    command_p = np.column_stack((design["Kp"], -design["Kp"]))
    command_pi = np.column_stack((design["Kp"], -design["Kp"], design["Ki"]))
    p_record = flow_record(design["AP"], command_p, FLOW_RATE)
    pi_record = flow_record(design["API"], command_pi, FLOW_RATE)

    transfer = np.vstack((np.eye(12), np.zeros((2, 12))))
    transition_factor = float(np.max(eigvalsh(transfer.T @ pi_record["P"] @ transfer, p_record["P"])))
    installed_p_level = P_TO_PI_HEADROOM * pi_record["level"] / transition_factor
    assert installed_p_level <= p_record["level"]
    p_record["raw_level"] = p_record["level"]
    p_record["installed_level"] = installed_p_level
    p_record["P_to_PI_transition_factor"] = transition_factor
    p_record["transition_headroom_ratio"] = transition_factor * installed_p_level / pi_record["level"]

    adaptation = adaptation_record(design, pi_record["P"])
    simulation = simulate(design, p_record, pi_record, adaptation)
    jacobian_audit = audit_augmented_jacobians(design)
    assert simulation["maximum_p_occupancy"] <= 1.0 + 1.0e-8
    assert simulation["maximum_pi_occupancy"] <= 1.0 + 1.0e-8
    assert simulation["minimum_state_margin"] >= 0.0
    assert simulation["minimum_input_margin"] >= 0.0
    plot(simulation)

    output = {
        "state_order": STATE_ORDER,
        "measured_outputs": ["n1", "n2"],
        "guarantee_profile": {
            "profile": "PW",
            "reason": (
                "one-time architecture transfer and decision-sensitive adaptation landing audit; "
                "no MDT or ADT timing predicate is claimed"
            ),
            "full_path_bound": "event-specific pathwise flow and transition records",
        },
        "design": design,
        "spectral_abscissae": {
            "observer_error": spectral_abscissa(design["Ao"]),
            "P_fiber": spectral_abscissa(design["AP"]),
            "PI_fiber": spectral_abscissa(design["API"]),
        },
        "augmented_jacobian_audit": jacobian_audit,
        "P_record": p_record,
        "PI_record": pi_record,
        "adaptation_record": adaptation,
        "simulation": {
            key: value for key, value in simulation.items()
            if key not in {"time", "eta", "physical_error", "estimated_error", "integrator", "commands"}
        },
    }
    with open(RESULTS_DIR / "mfd_architecture_results.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(output), handle, indent=2)
    np.savez_compressed(
        RESULTS_DIR / "mfd_architecture_trajectories.npz",
        time=simulation["time"],
        eta=simulation["eta"],
        physical_error=simulation["physical_error"],
        estimated_error=simulation["estimated_error"],
        integrator=simulation["integrator"],
        commands=simulation["commands"],
    )

    print("P/PI spectral abscissae:", spectral_abscissa(design["AP"]), spectral_abscissa(design["API"]))
    print("P/PI raw levels:", p_record["raw_level"], pi_record["level"])
    print("P installed level and transition factor:", installed_p_level, transition_factor)
    print("P-to-PI robust ratio:", simulation["p_to_pi"]["robust_landing_ratio"])
    print("P-to-PI command jump:", simulation["p_to_pi"]["command_jump_inf"])
    print(
        "Adaptation level and coarse/refined robust ratios:",
        adaptation["level"],
        [item["robust_landing_ratio"] for item in simulation["adaptation_requests"]],
    )
    print("Minimum state/input margins:", simulation["minimum_state_margin"], simulation["minimum_input_margin"])


if __name__ == "__main__":
    main()
