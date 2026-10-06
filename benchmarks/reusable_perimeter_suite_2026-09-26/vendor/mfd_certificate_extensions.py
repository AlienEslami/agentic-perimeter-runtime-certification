#!/usr/bin/env python3
"""Co-design, target-change, disturbance, and set-valued MFD audits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import brentq, minimize, minimize_scalar

import mfd_certificate_synthesis as synthesis
import mfd_fixed_gain_validation as mfd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
DIMENSION = 6


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


def numerical_linearization(x_star: np.ndarray, u_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_matrix = mfd.centered_jacobian(lambda x: mfd.mfd_rhs(x, u_star), x_star)
    b_matrix = mfd.centered_jacobian(lambda u: mfd.mfd_rhs(x_star, u), u_star)
    return a_matrix, b_matrix


def lqr_library(a_matrix: np.ndarray, b_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gains = []
    for q_matrix, r_matrix in zip(mfd.Q_LQR, mfd.R_LQR):
        riccati = solve_continuous_are(a_matrix, b_matrix, q_matrix, r_matrix)
        gains.append(-np.linalg.solve(r_matrix, b_matrix.T @ riccati))
    return tuple(gains)


def independent_custom_certificate(closed_loops: tuple[np.ndarray, ...], rate: float = 0.2) -> dict:
    matrices = [
        solve_continuous_lyapunov((closed_loop + 0.5 * rate * np.eye(DIMENSION)).T, -np.eye(DIMENSION))
        for closed_loop in closed_loops
    ]
    nu_12 = mfd.generalized_max(matrices[1], matrices[0])
    nu_21 = mfd.generalized_max(matrices[0], matrices[1])
    scale_2 = math.sqrt(nu_21 / nu_12)
    matrices[1] *= scale_2
    factor = math.sqrt(nu_12 * nu_21)
    return {
        "rate": rate,
        "matrices": matrices,
        "flow_scales": [1.0, scale_2],
        "transition_factor": factor,
        "dwell_time": math.log(factor) / rate,
    }


def state_halfspaces(x_star: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    margins: list[float] = []
    for index in range(DIMENSION):
        row = np.zeros(DIMENSION)
        row[index] = 1.0
        rows.append(row)
        margins.append(float(x_star[index]))
    for index in (0, 1):
        row = np.zeros(DIMENSION)
        row[index] = -1.0
        rows.append(row)
        margins.append(float(100.0 - x_star[index]))
    rows.extend([
        np.array([1.0, 0.0, -1.0, 0.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0]),
    ])
    margins.extend([
        float(x_star[0] - x_star[2] - x_star[4]),
        float(x_star[1] - x_star[3] - x_star[5]),
    ])
    return rows, margins


def minimum_physical_margin(state: np.ndarray, command: np.ndarray) -> float:
    n1, n2, n11, n22, n10, n20 = state
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    values = list(command) + list(1.0 - command)
    values.extend([
        mfd.DIRECTIONAL_LIMIT - abs(command[1] - command[0]),
        mfd.DIRECTIONAL_LIMIT - abs(command[3] - command[2]),
        n1, n2, 100.0 - n1, 100.0 - n2,
        n11, n22, n10, n20, n12, n21,
    ])
    return float(np.min(values))


def custom_tube_record(
    x_star: np.ndarray,
    u_star: np.ndarray,
    gains: tuple[np.ndarray, ...],
    closed_loops: tuple[np.ndarray, ...],
    certificate: dict,
) -> dict:
    roots = []
    constraint_roots = []
    levels = []
    details = []
    rows, state_margins = state_halfspaces(x_star)
    for index, (gain, closed_loop, p_matrix) in enumerate(zip(gains, closed_loops, certificate["matrices"])):
        flow_residual = closed_loop.T @ p_matrix + p_matrix @ closed_loop + certificate["rate"] * p_matrix
        slack = -float(np.max(eigvalsh(flow_residual)))
        if slack <= 0.0:
            raise RuntimeError(f"nonpositive flow slack for mode {index + 1}: {slack}")
        remainder_vectors = mfd.nonlinear_coefficient_vectors(
            gain, closed_loop, x_star=x_star, u_star=u_star, demand=mfd.DEMAND
        )
        values, vectors = np.linalg.eigh(p_matrix)
        sqrt_p = vectors @ np.diag(np.sqrt(values)) @ vectors.T
        metric_quadratic = sum(
            np.linalg.norm(sqrt_p @ vector)
            for exponent, vector in remainder_vectors.items() if sum(exponent) == 2
        )
        metric_cubic = sum(
            np.linalg.norm(sqrt_p @ vector)
            for exponent, vector in remainder_vectors.items() if sum(exponent) == 3
        )
        nonlinear_root = brentq(
            lambda radius: 2.0 * math.sqrt(float(np.max(values)))
            * (metric_quadratic * radius + metric_cubic * radius * radius) - slack,
            0.0, 5.0,
        )
        physical_radii = [margin / np.linalg.norm(row) for row, margin in zip(rows, state_margins)]
        physical_radii.extend([
            min(u_star[row_index], 1.0 - u_star[row_index]) / np.linalg.norm(gain[row_index])
            for row_index in range(4)
        ])
        for first, second in ((1, 0), (3, 2)):
            equilibrium_difference = abs(u_star[first] - u_star[second])
            physical_radii.append(
                (mfd.DIRECTIONAL_LIMIT - equilibrium_difference)
                / np.linalg.norm(gain[first] - gain[second])
            )
        constraint_root = min(physical_radii)
        radius = min(nonlinear_root, constraint_root)
        level = float(np.min(values) * radius * radius)
        roots.append(radius)
        constraint_roots.append(constraint_root)
        levels.append(level)
        details.append({
            "flow_slack": slack,
            "metric_quadratic": metric_quadratic,
            "metric_cubic": metric_cubic,
            "nonlinear_radius": nonlinear_root,
            "constraint_radius": constraint_root,
        })
    return {
        "radii": roots,
        "radii_vehicles": [100.0 * radius for radius in roots],
        "constraint_radii": constraint_roots,
        "constraint_radii_vehicles": [100.0 * radius for radius in constraint_roots],
        "individual_levels": levels,
        "common_level": min(levels),
        "details": details,
    }


def codesign_frontier(solver: str) -> tuple[dict, list[dict]]:
    cp = synthesis.import_cvxpy()
    baseline = mfd.independent_certificate(rate=0.2)
    factors = np.array([1.31, 1.4, 1.6, 2.0, 2.5, baseline["transition_factor"]])
    records = []
    best_certificate = None
    for factor in factors:
        result = synthesis.codesign_for_factor(cp, float(factor), 0.2, solver, False)
        if not result["feasible"]:
            records.append({"factor_limit": factor, "feasible": False, "status": result["status"]})
            continue
        matrices = result.pop("matrices")
        actual_factor = max(
            mfd.generalized_max(matrices[1], matrices[0]),
            mfd.generalized_max(matrices[0], matrices[1]),
        )
        certificate = {
            "rate": 0.2,
            "matrices": matrices,
            "flow_scales": result["actual_flow_slacks"],
            "transition_factor": actual_factor,
            "dwell_time": math.log(actual_factor) / 0.2,
            "realized_dwell": math.ceil(math.log(actual_factor) / 0.2 / mfd.PROPOSAL_INTERVAL - 1.0e-12) * mfd.PROPOSAL_INTERVAL,
        }
        tube = custom_tube_record(mfd.X_STAR, mfd.U_STAR, mfd.GAINS, mfd.CLOSED_LOOP, certificate)
        record = {
            "factor_limit": factor,
            "feasible": True,
            "actual_factor": actual_factor,
            "dwell_h": certificate["dwell_time"],
            "realized_dwell_h": certificate["realized_dwell"],
            "minimum_radius_vehicles": min(tube["radii_vehicles"]),
            "mode_radii_vehicles": tube["radii_vehicles"],
            "common_level": tube["common_level"],
            "minimum_flow_slack": min(result["actual_flow_slacks"]),
        }
        records.append(record)
        if best_certificate is None or actual_factor < best_certificate["transition_factor"]:
            best_certificate = certificate
    if best_certificate is None:
        raise RuntimeError("no co-design frontier point was feasible")
    return best_certificate, records


def common_p_controller_library(solver: str) -> dict:
    cp = synthesis.import_cvxpy()
    s_matrix = cp.Variable((DIMENSION, DIMENSION), symmetric=True)
    y_matrices = [cp.Variable((4, DIMENSION)) for _ in range(2)]
    slack = cp.Variable(nonneg=True)
    # A conditioning floor prevents the convex synthesis from manufacturing a
    # formally common but operationally meaningless, nearly singular metric.
    constraints = [s_matrix >> 2.0e-2 * np.eye(DIMENSION), cp.trace(s_matrix) == 1.0]
    deviations = []
    for gain, y_matrix in zip(mfd.GAINS, y_matrices):
        flow = mfd.A_ANALYTIC @ s_matrix + mfd.B_ANALYTIC @ y_matrix
        constraints.append(flow + flow.T + 0.2 * s_matrix << -slack * np.eye(DIMENSION))
        deviations.append(cp.sum_squares(y_matrix - gain @ s_matrix))
    problem = cp.Problem(cp.Minimize(sum(deviations) - 1.0e-3 * slack), constraints)
    synthesis.solve_problem(problem, solver, False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"common-P controller synthesis failed: {problem.status}")
    s_value = 0.5 * (s_matrix.value + s_matrix.value.T)
    q_common = np.linalg.inv(s_value)
    gains = tuple(np.asarray(y.value) @ np.linalg.inv(s_value) for y in y_matrices)
    closed_loops = tuple(mfd.A_ANALYTIC + mfd.B_ANALYTIC @ gain for gain in gains)
    residuals = [
        closed_loop.T @ q_common + q_common @ closed_loop + 0.2 * q_common
        for closed_loop in closed_loops
    ]
    worst_residual = max(float(np.max(eigvalsh(residual))) for residual in residuals)
    gain_separation = float(np.linalg.norm(gains[0] - gains[1]))
    if worst_residual >= -1.0e-7:
        raise RuntimeError(f"common-P verification residual is not strict: {worst_residual}")
    certificate = {
        "rate": 0.2,
        "matrices": [q_common, q_common],
        "transition_factor": 1.0,
        "dwell_time": 0.0,
    }
    tube = custom_tube_record(mfd.X_STAR, mfd.U_STAR, gains, closed_loops, certificate)
    return {
        "solver": solver,
        "status": problem.status,
        "gains": gains,
        "common_P": q_common,
        "gain_separation_frobenius": gain_separation,
        "closed_loop_spectral_abscissae": [mfd.spectral_abscissa(matrix) for matrix in closed_loops],
        "worst_common_P_flow_residual": worst_residual,
        "transition_factor": 1.0,
        "dwell_h": 0.0,
        "tube": tube,
    }


def solve_target_equilibrium(delta: float) -> tuple[np.ndarray, np.ndarray]:
    target = np.array([30.0 - delta, 30.0 + delta])
    initial = np.concatenate((mfd.X_STAR, mfd.U_STAR))
    scales = np.concatenate((np.full(6, 5.0), np.full(4, 0.2)))
    objective = lambda value: float(np.sum(((value - initial) / scales) ** 2))
    constraints = [
        {"type": "eq", "fun": lambda value: mfd.mfd_rhs(value[:6], value[6:])},
        {"type": "eq", "fun": lambda value: value[:2] - target},
    ]
    bounds = [(1.0, 99.0), (1.0, 99.0)] + [(0.05, 98.0)] * 4 + [(0.05, 0.95)] * 4
    result = minimize(
        objective, initial, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1.0e-12, "maxiter": 3000},
    )
    if not result.success or np.linalg.norm(mfd.mfd_rhs(result.x[:6], result.x[6:]), np.inf) > 1.0e-8:
        raise RuntimeError(f"target equilibrium solve failed for delta={delta}: {result.message}")
    return result.x[:6], result.x[6:]


def global_offset_pair(p_source: np.ndarray, p_successor: np.ndarray, delta: np.ndarray) -> dict:
    p_source = p_source.copy()
    p_successor = p_successor.copy()
    mu = float(np.max(eigvalsh(p_successor, p_source)))

    def record(nu: float) -> tuple[float, float]:
        gap = nu * p_source - p_successor
        vector = p_successor @ delta
        kappa = float(delta @ p_successor @ delta + vector @ np.linalg.solve(gap, vector))
        return nu, kappa

    optimum = minimize_scalar(
        lambda nu: sum(record(nu)),
        bounds=(mu * (1.0 + 1.0e-7), max(10.0, 5.0 * mu)),
        method="bounded",
        options={"xatol": 1.0e-11},
    )
    nu, kappa = record(float(optimum.x))
    return {"mu": mu, "nu": nu, "kappa": kappa, "unit_tube_bound": nu + kappa}


def target_change_audit() -> dict:
    old_certificate = mfd.independent_certificate(rate=0.2)
    old_tube = mfd.tube_record(old_certificate)
    source_fraction = 0.25

    def construct(delta_value: float) -> dict:
        x_new, u_new = solve_target_equilibrium(delta_value)
        a_new, b_new = numerical_linearization(x_new, u_new)
        gains_new = lqr_library(a_new, b_new)
        closed_new = tuple(a_new + b_new @ gain for gain in gains_new)
        certificate_new = independent_custom_certificate(closed_new, rate=0.2)
        tube_new = custom_tube_record(x_new, u_new, gains_new, closed_new, certificate_new)
        p_old = old_certificate["matrices"][0] / (source_fraction * old_tube["common_level"])
        p_new = certificate_new["matrices"][0] / tube_new["common_level"]
        offset = mfd.X_STAR - x_new
        pair = global_offset_pair(p_old, p_new, offset)
        return {
            "delta_units": delta_value,
            "x_star": x_new,
            "u_star": u_new,
            "A": a_new,
            "B": b_new,
            "gains": gains_new,
            "closed_loops": closed_new,
            "certificate": certificate_new,
            "tube": tube_new,
            "normalized_transition_pair": pair,
            "landing_bound": pair["unit_tube_bound"],
        }

    lower, upper = 0.0, 0.2
    lower_record = construct(lower)
    if lower_record["landing_bound"] > 0.98:
        raise RuntimeError("event-specific source contraction is insufficient even at zero target offset")
    upper_record = construct(upper)
    while upper_record["landing_bound"] <= 0.98 and upper < 2.0:
        lower, lower_record = upper, upper_record
        upper *= 2.0
        upper_record = construct(upper)
    for _ in range(24):
        midpoint = 0.5 * (lower + upper)
        trial = construct(midpoint)
        if trial["landing_bound"] <= 0.98:
            lower, lower_record = midpoint, trial
        else:
            upper, upper_record = midpoint, trial
    installed = construct(0.8 * lower)

    switch_time = 6.0
    final_time = 24.0
    initial_state = mfd.dominant_initial_state(old_certificate, old_tube)
    first = solve_ivp(
        lambda time_value, state_value: mfd.mfd_rhs(
            state_value, mfd.U_STAR + mfd.GAINS[0] @ (state_value - mfd.X_STAR)
        ),
        (0.0, switch_time), initial_state, method="DOP853",
        dense_output=True, rtol=1.0e-10, atol=1.0e-12, max_step=0.01,
    )
    switch_state = first.y[:, -1]
    new_gain = installed["gains"][0]
    x_new = installed["x_star"]
    u_new = installed["u_star"]
    second = solve_ivp(
        lambda time_value, state_value: mfd.mfd_rhs(
            state_value, u_new + new_gain @ (state_value - x_new)
        ),
        (switch_time, final_time), switch_state, method="DOP853",
        dense_output=True, rtol=1.0e-10, atol=1.0e-12, max_step=0.01,
    )
    before = mfd.U_STAR + mfd.GAINS[0] @ (switch_state - mfd.X_STAR)
    after = u_new + new_gain @ (switch_state - x_new)
    p_new = installed["certificate"]["matrices"][0]
    actual_ratio = float((switch_state - x_new) @ p_new @ (switch_state - x_new) / installed["tube"]["common_level"])
    sampled_times = np.linspace(switch_time, final_time, 361)
    sampled_states = second.sol(sampled_times).T
    sampled_commands = np.asarray([u_new + new_gain @ (state - x_new) for state in sampled_states])
    minimum_margin = min(
        minimum_physical_margin(state, command)
        for state, command in zip(sampled_states, sampled_commands)
    )
    return {
        "source_installed_level_fraction": source_fraction,
        "maximum_certified_delta_units": lower,
        "maximum_certified_delta_vehicles_per_region": 100.0 * lower,
        "installed_delta_units": installed["delta_units"],
        "installed_delta_vehicles_per_region": 100.0 * installed["delta_units"],
        "new_equilibrium": installed["x_star"],
        "new_equilibrium_command": installed["u_star"],
        "normalized_transition_pair": installed["normalized_transition_pair"],
        "actual_post_transition_ratio": actual_ratio,
        "command_jump_inf": float(np.max(np.abs(after - before))),
        "minimum_post_transition_margin": minimum_margin,
        "final_error_norm": float(np.linalg.norm(sampled_states[-1] - x_new)),
        "new_tube": installed["tube"],
        "bracket_first_infeasible_delta_units": upper,
    }


def demand_envelope() -> dict:
    times = np.linspace(0.0, 24.0, 100_001)
    deviations = np.asarray([mfd.out_of_sample_demand(time_value) - mfd.DEMAND for time_value in times])
    norms = np.linalg.norm(deviations, axis=1)
    index = int(np.argmax(norms))
    return {
        "w_bar": float(norms[index]),
        "attainment_time_h": float(times[index]),
        "componentwise_max_abs": np.max(np.abs(deviations), axis=0),
    }


def robust_radius_for_dwell(certificate: dict, tube: dict, w_bar: float, dwell: float) -> dict | None:
    gamma = certificate["rate"]
    nu = certificate["transition_factor"]
    epsilon_upper = gamma - math.log(nu) / dwell
    if epsilon_upper <= 0.0:
        return None
    p_max = max(float(np.max(eigvalsh(matrix))) for matrix in certificate["matrices"])

    def complete_radius(epsilon: float) -> float:
        rate = gamma - epsilon
        d_bar = p_max * w_bar * w_bar / epsilon
        c_value = d_bar / rate
        beta = math.exp(-rate * dwell)
        return nu * (1.0 - beta) * c_value / (1.0 - nu * beta)

    optimum = minimize_scalar(
        complete_radius,
        bounds=(max(1.0e-10, epsilon_upper * 1.0e-7), epsilon_upper * (1.0 - 1.0e-7)),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    epsilon = float(optimum.x)
    radius = complete_radius(epsilon)
    return {
        "dwell_h": dwell,
        "epsilon": epsilon,
        "lambda": gamma - epsilon,
        "R_all": radius,
        "tube_level": tube["common_level"],
        "R_all_to_tube_level": radius / tube["common_level"],
        "feasible_in_installed_nonlinear_tube": radius <= tube["common_level"],
    }


def robust_demand_audit(analytic: dict, analytic_tube: dict, codesigned: dict, codesigned_tube: dict) -> dict:
    envelope = demand_envelope()
    candidates = {}
    for name, certificate, tube in (
        ("analytic", analytic, analytic_tube),
        ("codesigned", codesigned, codesigned_tube),
    ):
        records = []
        minimum_dwell = math.log(certificate["transition_factor"]) / certificate["rate"]
        for dwell in np.linspace(minimum_dwell + 0.05, 96.0, 80):
            record = robust_radius_for_dwell(certificate, tube, envelope["w_bar"], float(dwell))
            if record is not None:
                records.append(record)
        best = min(records, key=lambda item: item["R_all_to_tube_level"])
        installed = robust_radius_for_dwell(
            certificate, tube, envelope["w_bar"], certificate.get("realized_dwell", math.ceil(minimum_dwell * 10.0) / 10.0)
        )
        candidates[name] = {
            "installed_dwell_record": installed,
            "best_record_over_dwell_up_to_96_h": best,
            "any_feasible_record": any(record["feasible_in_installed_nonlinear_tube"] for record in records),
        }
    return {
        "disturbance_envelope": envelope,
        "sharp_radius_identity": "R_all=[kappa_bar+nu(1-beta)c]/[1-nu*beta], with kappa_bar=0 in this audit",
        "certificates": candidates,
        "interpretation": (
            "The regenerated robust-radius test is infeasible for this demand envelope. "
            "The accompanying finite stress-run trajectory retains positive physical margins."
        ),
    }


def non_singleton_withholding(certificate: dict, tube: dict) -> dict:
    initial_fraction = 0.45
    capture_fraction = 1.0 / certificate["transition_factor"]
    capture_time = math.log(initial_fraction / capture_fraction) / certificate["rate"]
    evaluation_time = math.ceil(capture_time / mfd.PROPOSAL_INTERVAL - 1.0e-12) * mfd.PROPOSAL_INTERVAL
    grant_time = max(evaluation_time, certificate["realized_dwell"])
    initial_volume_proxy = (initial_fraction * tube["common_level"]) ** (DIMENSION / 2.0)
    grant_volume_proxy = (
        initial_fraction * tube["common_level"] * math.exp(-certificate["rate"] * grant_time)
    ) ** (DIMENSION / 2.0)
    return {
        "information_set": "I_B(t)={e: e^T P_source e <= 0.45 ell exp(-lambda t)}",
        "is_non_singleton_at_grant": grant_volume_proxy > 0.0,
        "initial_supremum_fraction": initial_fraction,
        "capture_fraction": capture_fraction,
        "capture_time_h": capture_time,
        "first_capture_evaluation_h": evaluation_time,
        "realized_grant_time_h": grant_time,
        "capture_to_grant_gap_h": grant_time - capture_time,
        "relative_volume_proxy_at_grant": grant_volume_proxy / initial_volume_proxy,
        "successor_supremum_bound_at_grant": (
            certificate["transition_factor"] * initial_fraction
            * math.exp(-certificate["rate"] * grant_time)
        ),
    }


def plot_frontier(records: list[dict]) -> None:
    feasible = [record for record in records if record["feasible"]]
    dwell = [record["dwell_h"] for record in feasible]
    radii = [record["minimum_radius_vehicles"] for record in feasible]
    figure, axis = plt.subplots(figsize=(4.8, 3.2))
    axis.plot(dwell, radii, marker="o", color="#315f8a")
    axis.set_xlabel("sufficient dwell lower bound (h)")
    axis.set_ylabel("minimum certified radius (vehicles)")
    axis.grid(alpha=0.3)
    for index, record in enumerate(feasible):
        offset = (-34, 4) if index == len(feasible) - 1 else (0, 3)
        axis.annotate(
            rf"$\bar{{\nu}}={record['actual_factor']:.2f}$",
            (record["dwell_h"], record["minimum_radius_vehicles"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "mfd_codesign_frontier.pdf", bbox_inches="tight")
    figure.savefig(RESULTS_DIR / "mfd_codesign_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="CLARABEL")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    analytic = mfd.independent_certificate(rate=0.2)
    analytic_tube = mfd.tube_record(analytic)
    codesigned, frontier = codesign_frontier(args.solver)
    codesigned_tube = custom_tube_record(mfd.X_STAR, mfd.U_STAR, mfd.GAINS, mfd.CLOSED_LOOP, codesigned)
    common_library = common_p_controller_library(args.solver)
    target = target_change_audit()
    robust = robust_demand_audit(analytic, analytic_tube, codesigned, codesigned_tube)
    withholding = non_singleton_withholding(analytic, analytic_tube)
    output = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "solver": args.solver,
        "analytic_reference": {
            "transition_factor": analytic["transition_factor"],
            "dwell_h": analytic["dwell_time"],
            "installed_dwell_h": analytic["realized_dwell"],
            "radii_vehicles": [100.0 * radius for radius in analytic_tube["radii"]],
        },
        "quadratic_codesign_frontier": frontier,
        "selected_codesigned_certificate": {
            "transition_factor": codesigned["transition_factor"],
            "dwell_h": codesigned["dwell_time"],
            "installed_dwell_h": codesigned["realized_dwell"],
            "tube": codesigned_tube,
        },
        "common_P_controller_library": common_library,
        "target_change": target,
        "robust_demand": robust,
        "non_singleton_withholding": withholding,
    }
    (RESULTS_DIR / "mfd_certificate_extensions.json").write_text(
        json.dumps(json_ready(output), indent=2), encoding="utf-8"
    )
    plot_frontier(frontier)
    print(json.dumps(json_ready(output), indent=2))


if __name__ == "__main__":
    main()
