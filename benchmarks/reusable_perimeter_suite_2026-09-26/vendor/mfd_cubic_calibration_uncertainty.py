#!/usr/bin/env python3
"""Calibration-informed production-curve uncertainty study.

The predeclared protocol is ``MFD_CALIBRATION_UNCERTAINTY_PROTOCOL.md``.
The external magnitude is the approximately 15% loading-to-recovery capacity
drop reported by Ma et al. (Transportation Research Part A, 2026).  It is
transferred to the installed Haddad cubic as G_gamma = gamma G,
gamma in [0.85, 1].  Endpoint SOS records certify the full affine family.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh

import mfd_cubic_nonstationary_study as study
import mfd_cubic_pilot as pilot
import mfd_cubic_runtime_pilot as runtime
import mfd_cubic_sos_pilot as sos


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_calibration_uncertainty_results.json"
FIGURE_PDF = RESULTS_DIR / "mfd_cubic_calibration_uncertainty.pdf"
FIGURE_PNG = RESULTS_DIR / "mfd_cubic_calibration_uncertainty.png"

GAMMA_ENDPOINTS = (0.85, 1.0)
GAMMA_SIMULATION_GRID = (0.85, 0.90, 0.95, 1.0)
RATE_GRID = tuple(np.round(np.arange(1.00, 1.451, 0.05), 2))
PROOF_MARGIN = 1.0e-4
RELATIVE_AFFINITY_TOLERANCE = 2.0e-11
HORIZON_H = 8.0
PROPOSAL_MINUTES = 15.0
AUDIT_MINUTES = 1.0
PRIORITY_EPISODE_MINUTES = 60.0
INITIAL_OCCUPANCY = 0.8
BURST_CAPACITY = 2

BASE_COEFFICIENTS = (pilot.MFD_A, pilot.MFD_B, pilot.MFD_C)


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


def install_gamma(gamma: float) -> np.ndarray:
    pilot.MFD_A = gamma * BASE_COEFFICIENTS[0]
    pilot.MFD_B = gamma * BASE_COEFFICIENTS[1]
    pilot.MFD_C = gamma * BASE_COEFFICIENTS[2]
    pilot.DEMAND = pilot.equilibrium_demand()
    return pilot.DEMAND.copy()


def restore_nominal() -> None:
    pilot.MFD_A, pilot.MFD_B, pilot.MFD_C = BASE_COEFFICIENTS
    pilot.DEMAND = pilot.equilibrium_demand()


def linear_record(
    gamma: float,
    gains: tuple[np.ndarray, ...],
    matrices: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    demand = install_gamma(gamma)
    a_matrix = pilot.complex_step_jacobian(
        lambda state: pilot.mfd_rhs(state, pilot.U_STAR), pilot.X_STAR
    )
    b_matrix = pilot.complex_step_jacobian(
        lambda command: pilot.mfd_rhs(pilot.X_STAR, command), pilot.U_STAR
    )
    closed_loop = tuple(a_matrix + b_matrix @ gain for gain in gains)
    certified_rates = [
        float(eigvalsh(-(flow.T @ metric + metric @ flow), metric)[0])
        for flow, metric in zip(closed_loop, matrices)
    ]
    return {
        "gamma": gamma,
        "equilibrium_demand_units_per_hour": demand,
        "equilibrium_residual_infinity_norm": float(
            np.linalg.norm(pilot.mfd_rhs(pilot.X_STAR, pilot.U_STAR), ord=np.inf)
        ),
        "spectral_abscissae_per_hour": [
            float(np.max(np.real(np.linalg.eigvals(flow)))) for flow in closed_loop
        ],
        "maximum_fixed_metric_rates_per_hour": certified_rates,
    }


def polynomial_affinity_audit(
    gain: np.ndarray,
    matrix: np.ndarray,
    rate: float,
) -> float:
    polynomials: dict[float, dict[tuple[int, ...], float]] = {}
    for gamma in (0.85, 0.925, 1.0):
        install_gamma(gamma)
        polynomials[gamma] = sos.dissipation_polynomial(gain, matrix, rate)
    exponents = set().union(*[set(item) for item in polynomials.values()])
    residual = max(
        abs(
            polynomials[0.925].get(exponent, 0.0)
            - 0.5 * polynomials[0.85].get(exponent, 0.0)
            - 0.5 * polynomials[1.0].get(exponent, 0.0)
        )
        for exponent in exponents
    )
    scale = max(
        1.0,
        max(abs(value) for item in polynomials.values() for value in item.values()),
    )
    return float(residual / scale)


def endpoint_sos_at_rate(
    rate: float,
    gains: tuple[np.ndarray, ...],
    matrices: tuple[np.ndarray, ...],
    common_level: float,
    solver: str,
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    passed = True
    for gamma in GAMMA_ENDPOINTS:
        install_gamma(gamma)
        modes = []
        for mode, (gain, matrix) in enumerate(zip(gains, matrices)):
            try:
                record = sos.sos_at_level(
                    gain,
                    matrix,
                    rate,
                    common_level,
                    solver,
                    PROOF_MARGIN,
                    False,
                )
            except sos.cp.error.SolverError as error:
                # CLARABEL can terminate with a solver error when an SDP is
                # outside its feasible numerical cone. Treat that grid point
                # as failed and continue to the next predeclared lower rate.
                record = {
                    "feasible": False,
                    "status": "solver_error",
                    "level": common_level,
                    "solver_error": str(error),
                }
            modes.append({"mode": mode + 1, **record})
            passed = passed and bool(record["feasible"])
        endpoints[f"gamma_{gamma:.2f}"] = modes
    affinity = [
        polynomial_affinity_audit(gain, matrix, rate)
        for gain, matrix in zip(gains, matrices)
    ]
    passed = passed and max(affinity) <= RELATIVE_AFFINITY_TOLERANCE
    return {
        "rate_per_hour": rate,
        "passed": bool(passed),
        "endpoint_sos": endpoints,
        "relative_polynomial_affinity_residuals": affinity,
        "affinity_tolerance": RELATIVE_AFFINITY_TOLERANCE,
    }


def candidate_landing(
    state: np.ndarray,
    requested: int,
    gains: tuple[np.ndarray, ...],
    matrices: tuple[np.ndarray, ...],
    common_level: float,
) -> tuple[bool, float, float, float]:
    error = state - pilot.X_STAR
    value = float(error @ matrices[requested] @ error)
    state_margin, input_margin = study.constraint_margins(state, requested, gains)
    admissible = (
        value <= common_level + 1.0e-10
        and state_margin >= -1.0e-8
        and input_margin >= -1.0e-10
    )
    return admissible, value, state_margin, input_margin


def simulate(
    gamma: float,
    policy: str,
    gains: tuple[np.ndarray, ...],
    matrices: tuple[np.ndarray, ...],
    transition_factor: float,
    common_level: float,
    family_rate: float,
    include_trajectory: bool = False,
) -> dict[str, Any]:
    if policy not in {"family_dwell", "family_adt_envelope", "family_envelope"}:
        raise ValueError(policy)
    demand = install_gamma(gamma)
    proposal_h = PROPOSAL_MINUTES / 60.0
    audit_h = AUDIT_MINUTES / 60.0
    critical_average_h = math.log(transition_factor) / family_rate
    grid_average_h = proposal_h * math.ceil(
        (critical_average_h + 1.0e-12) / proposal_h
    )
    grid_dwell_h = grid_average_h
    envelope_rate = min(runtime.SELECTED_ENVELOPE_RATE, 0.5 * family_rate)

    state = runtime.state_on_tube(
        runtime.structured_directions()["region_1_heavy"],
        INITIAL_OCCUPANCY,
        {"matrices": matrices},
        {"common_level": common_level},
        mode=0,
    )
    mode = 0
    error = state - pilot.X_STAR
    initial_level = float(error @ matrices[mode] @ error)
    last_switch_h = -1.0e9
    tokens = float(BURST_CAPACITY)
    token_update_h = 0.0
    executed_times: list[float] = []
    rejected = 0
    no_ops = 0
    physical_violation = None

    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    priorities = [study.block_priority(0.0, PRIORITY_EPISODE_MINUTES)]
    token_trace: list[dict[str, float]] = []
    decisions: list[dict[str, Any]] = []

    decision_h = 0.0
    terminated = False
    while decision_h < HORIZON_H - 1.0e-12 and not terminated:
        if policy == "family_adt_envelope":
            tokens = min(
                float(BURST_CAPACITY),
                tokens + (decision_h - token_update_h) / grid_average_h,
            )
            token_update_h = decision_h
        requested = study.block_priority(decision_h, PRIORITY_EPISODE_MINUTES)
        if requested == mode:
            no_ops += 1
            outcome = "no_op_reselection"
        else:
            admissible, candidate_value, state_margin, input_margin = candidate_landing(
                state, requested, gains, matrices, common_level
            )
            threshold = common_level
            if policy in {"family_adt_envelope", "family_envelope"}:
                threshold = initial_level * math.exp(-envelope_rate * decision_h)
                admissible = admissible and candidate_value <= threshold + 1.0e-10
            if policy == "family_dwell":
                admissible = admissible and (
                    decision_h - last_switch_h + 1.0e-12 >= grid_dwell_h
                )
            if policy == "family_adt_envelope":
                admissible = admissible and tokens >= 1.0 - 1.0e-12
            if admissible:
                mode = requested
                last_switch_h = decision_h
                executed_times.append(decision_h)
                if policy == "family_adt_envelope":
                    tokens -= 1.0
                outcome = "executed"
            else:
                rejected += 1
                outcome = "rejected"
            decisions.append(
                {
                    "time_h": decision_h,
                    "requested_mode": requested + 1,
                    "candidate_level": candidate_value,
                    "threshold_level": threshold,
                    "state_margin_vehicles": state_margin,
                    "input_margin": input_margin,
                    "tokens_after_decision": tokens,
                    "outcome": outcome,
                }
            )
        if policy == "family_adt_envelope":
            token_trace.append({"time_h": decision_h, "tokens": tokens})
        modes[-1] = mode

        end_h = min(decision_h + proposal_h, HORIZON_H)
        sample_times = np.arange(decision_h + audit_h, end_h + 0.5 * audit_h, audit_h)
        if sample_times.size == 0 or sample_times[-1] < end_h - 1.0e-12:
            sample_times = np.append(sample_times, end_h)
        sample_times[-1] = end_h

        def rhs(_time_h: float, current: np.ndarray) -> np.ndarray:
            command = pilot.U_STAR + gains[mode] @ (current - pilot.X_STAR)
            return pilot.mfd_rhs(current, command, demand=demand)

        solution = solve_ivp(
            rhs,
            (decision_h, end_h),
            state,
            method="DOP853",
            t_eval=sample_times,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=min(0.0025, proposal_h / 20.0),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        for time_value, state_value in zip(solution.t, solution.y.T):
            state = np.asarray(state_value, dtype=float)
            state_margin, input_margin = study.constraint_margins(state, mode, gains)
            times.append(float(time_value))
            states.append(state.copy())
            modes.append(mode)
            priorities.append(
                study.block_priority(float(time_value), PRIORITY_EPISODE_MINUTES)
            )
            if state_margin < -1.0e-6 or input_margin < -1.0e-9:
                physical_violation = {
                    "time_h": float(time_value),
                    "state_margin_vehicles": state_margin,
                    "input_margin": input_margin,
                }
                terminated = True
                break
        decision_h = end_h

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    priorities_array = np.asarray(priorities, dtype=int)
    errors = states_array - pilot.X_STAR
    values = np.asarray(
        [error @ matrices[active] @ error for error, active in zip(errors, modes_array)]
    )
    commands = np.asarray(
        [pilot.U_STAR + gains[active] @ error for error, active in zip(errors, modes_array)]
    )
    rates = [
        pilot.transportation_rates(state_value, command)
        for state_value, command in zip(states_array, commands)
    ]
    production = np.asarray([item["network_production_units_per_hour"] for item in rates])
    completions = np.asarray([item["completed_trips_units_per_hour"] for item in rates])
    priority_cost = np.asarray(
        [
            float(np.sum(pilot.Q_PRIORITY[priority] * error**2))
            for error, priority in zip(errors, priorities_array)
        ]
    )
    all_margins = [
        study.constraint_margins(state_value, int(active), gains)
        for state_value, active in zip(states_array, modes_array)
    ]

    count_residual = None
    if policy == "family_adt_envelope":
        count_residual = -math.inf
        endpoints = [0.0, *executed_times, HORIZON_H]
        for left in endpoints:
            for right in endpoints:
                if right + 1.0e-12 < left:
                    continue
                count = sum(left < item <= right + 1.0e-12 for item in executed_times)
                count_residual = max(
                    count_residual,
                    count - (BURST_CAPACITY + (right - left) / grid_average_h),
                )

    result: dict[str, Any] = {
        "gamma": gamma,
        "policy": policy,
        "equilibrium_demand_units_per_hour": demand,
        "equilibrium_residual_infinity_norm": float(
            np.linalg.norm(pilot.mfd_rhs(pilot.X_STAR, pilot.U_STAR, demand), ord=np.inf)
        ),
        "family_rate_per_hour": family_rate,
        "critical_average_dwell_minutes": 60.0 * critical_average_h,
        "proposal_grid_dwell_minutes": 60.0 * grid_dwell_h,
        "average_dwell_minutes": (
            60.0 * grid_average_h if policy == "family_adt_envelope" else None
        ),
        "burst_capacity": BURST_CAPACITY if policy == "family_adt_envelope" else None,
        "maximum_count_bound_residual": count_residual,
        "count_bound_verified": (
            None if count_residual is None else bool(count_residual <= 1.0e-10)
        ),
        "executed_changes": len(executed_times),
        "rejected_changes": rejected,
        "no_op_reselections": no_ops,
        "minimum_executed_interval_minutes": (
            None
            if len(executed_times) < 2
            else 60.0 * float(np.min(np.diff(np.asarray(executed_times))))
        ),
        "maximum_tube_occupancy": float(np.max(values) / common_level),
        "final_tube_occupancy": float(values[-1] / common_level),
        "minimum_state_constraint_margin_vehicles": min(item[0] for item in all_margins),
        "minimum_input_constraint_margin": min(item[1] for item in all_margins),
        "physical_violation": physical_violation,
        "transportation": {
            "vehicle_hours_in_network": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(states_array[:, 0] + states_array[:, 1], times_array)
            ),
            "regional_imbalance_vehicle_hours": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(np.abs(states_array[:, 0] - states_array[:, 1]), times_array)
            ),
            "cumulative_network_production_vehicles": float(
                pilot.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(production, times_array)
            ),
            "completed_trips_vehicles": float(
                pilot.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(completions, times_array)
            ),
            "time_varying_priority_tracking_cost": float(
                np.trapezoid(priority_cost, times_array)
            ),
            "fraction_time_executed_mode_matches_priority": float(
                np.mean(modes_array == priorities_array)
            ),
        },
        "decisions": decisions,
    }
    if include_trajectory:
        result["trajectory"] = {
            "time_h": times_array,
            "state": states_array,
            "mode": modes_array,
            "priority_mode": priorities_array,
            "tube_occupancy": values / common_level,
            "token_trace": token_trace,
        }
    return result


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"trajectory", "decisions"}}


def make_figure(
    linear_records: list[dict[str, Any]],
    simulations: dict[str, Any],
    selected_rate: float,
) -> None:
    gammas = np.asarray([item["gamma"] for item in linear_records])
    mode_rates = np.asarray(
        [item["maximum_fixed_metric_rates_per_hour"] for item in linear_records]
    )
    policies = ("family_dwell", "family_adt_envelope", "family_envelope")
    labels = ("fixed dwell", "ADT + envelope", "envelope")
    colors = ("#8f6b42", "#245a85", "#2d7c5b")

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 6.9))
    axes[0, 0].plot(gammas, mode_rates[:, 0], marker="o", label="mode 1")
    axes[0, 0].plot(gammas, mode_rates[:, 1], marker="s", label="mode 2")
    axes[0, 0].axhline(selected_rate, color="#9b2c2c", linestyle="--", label="SOS family rate")
    axes[0, 0].set_xlabel(r"production multiplier $\gamma$")
    axes[0, 0].set_ylabel("fixed-metric linear rate (1/h)")
    axes[0, 0].set_title("External-magnitude calibration family")
    axes[0, 0].legend(frameon=False)

    selected = simulations["gamma_0.85"]
    for policy, label, color in zip(policies, labels, colors):
        trajectory = selected[policy]["trajectory"]
        axes[0, 1].plot(
            trajectory["time_h"], trajectory["tube_occupancy"],
            label=label, color=color, linewidth=1.8,
        )
    axes[0, 1].axhline(1.0, color="#9b2c2c", linestyle="--", linewidth=1.0)
    axes[0, 1].set_xlabel("time (h)")
    axes[0, 1].set_ylabel("tube occupancy")
    axes[0, 1].set_title(r"Containment at $\gamma=0.85$")
    axes[0, 1].legend(frameon=False)

    positions = np.arange(len(GAMMA_SIMULATION_GRID))
    width = 0.24
    for offset, policy, label, color in zip((-1, 0, 1), policies, labels, colors):
        costs = [
            simulations[f"gamma_{gamma:.2f}"][policy]["transportation"]
            ["time_varying_priority_tracking_cost"]
            for gamma in GAMMA_SIMULATION_GRID
        ]
        axes[1, 0].bar(positions + offset * width, costs, width, label=label, color=color)
    axes[1, 0].set_xticks(positions, [f"{gamma:.2f}" for gamma in GAMMA_SIMULATION_GRID])
    axes[1, 0].set_xlabel(r"production multiplier $\gamma$")
    axes[1, 0].set_ylabel("priority-tracking cost")
    axes[1, 0].set_title("Policy result across the family")
    axes[1, 0].legend(frameon=False, fontsize=8)

    completed = {
        policy: np.asarray(
            [
                simulations[f"gamma_{gamma:.2f}"][policy]["transportation"]
                ["completed_trips_vehicles"]
                for gamma in GAMMA_SIMULATION_GRID
            ]
        )
        for policy in policies
    }
    calibration_spread = max(np.max(values) for values in completed.values()) - min(
        np.min(values) for values in completed.values()
    )
    within_case_spread = max(
        max(completed[policy][index] for policy in policies)
        - min(completed[policy][index] for policy in policies)
        for index in range(len(GAMMA_SIMULATION_GRID))
    )
    axes[1, 1].bar(
        [0, 1], [calibration_spread, within_case_spread],
        color=["#756bb1", "#d07c47"],
    )
    axes[1, 1].set_xticks([0, 1], ["between\ncalibrations", "within-calibration\npolicy effect"])
    axes[1, 1].set_ylabel("completed-trip spread (vehicles)")
    axes[1, 1].set_title("Scale of aggregate differences")
    axes[1, 1].set_yscale("log")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    gains_loaded, certificate, tube = study.load_installed()
    gains = tuple(np.asarray(item, dtype=float) for item in gains_loaded)
    matrices = tuple(np.asarray(item, dtype=float) for item in certificate["matrices"])
    common_level = float(tube["common_level"])
    transition_factor = float(certificate["transition_factor"])
    solver = sos.select_solver("auto")

    try:
        linear_records = [
            linear_record(gamma, gains, matrices)
            for gamma in np.linspace(GAMMA_ENDPOINTS[0], GAMMA_ENDPOINTS[1], 7)
        ]
        rate_trials = []
        selected = None
        for rate in reversed(RATE_GRID):
            trial = endpoint_sos_at_rate(
                float(rate), gains, matrices, common_level, solver
            )
            rate_trials.append(trial)
            if trial["passed"]:
                selected = trial
                break
        if selected is None:
            raise RuntimeError("No predeclared family rate passed the endpoint SOS audit")
        selected_rate = float(selected["rate_per_hour"])

        simulations: dict[str, Any] = {}
        for gamma in GAMMA_SIMULATION_GRID:
            gamma_runs = {}
            for policy in ("family_dwell", "family_adt_envelope", "family_envelope"):
                gamma_runs[policy] = simulate(
                    gamma,
                    policy,
                    gains,
                    matrices,
                    transition_factor,
                    common_level,
                    selected_rate,
                    include_trajectory=gamma == 0.85,
                )
            simulations[f"gamma_{gamma:.2f}"] = gamma_runs

        safety = {
            "all_runs_complete": all(
                abs(run["trajectory"]["time_h"][-1] - HORIZON_H) < 1.0e-10
                if "trajectory" in run
                else run["physical_violation"] is None
                for gamma_runs in simulations.values()
                for run in gamma_runs.values()
            ),
            "all_tubes_contained": all(
                run["maximum_tube_occupancy"] <= 1.0 + 1.0e-8
                for gamma_runs in simulations.values()
                for run in gamma_runs.values()
            ),
            "all_physical_constraints_satisfied": all(
                run["physical_violation"] is None
                and run["minimum_state_constraint_margin_vehicles"] >= -1.0e-6
                and run["minimum_input_constraint_margin"] >= -1.0e-9
                for gamma_runs in simulations.values()
                for run in gamma_runs.values()
            ),
            "all_average_dwell_counts_verified": all(
                run["count_bound_verified"] is True
                for gamma_runs in simulations.values()
                for name, run in gamma_runs.items()
                if name == "family_adt_envelope"
            ),
        }

        completed_values = [
            run["transportation"]["completed_trips_vehicles"]
            for gamma_runs in simulations.values()
            for run in gamma_runs.values()
        ]
        calibration_spread = max(completed_values) - min(completed_values)
        within_spreads = []
        for gamma_runs in simulations.values():
            values = [
                run["transportation"]["completed_trips_vehicles"]
                for run in gamma_runs.values()
            ]
            within_spreads.append(max(values) - min(values))
        effect_scale = {
            "completed_trip_between_calibration_spread_vehicles": calibration_spread,
            "maximum_within_calibration_policy_spread_vehicles": max(within_spreads),
            "policy_to_calibration_spread_ratio": max(within_spreads) / calibration_spread,
        }

        output = {
            "protocol": "MFD_CALIBRATION_UNCERTAINTY_PROTOCOL.md",
            "external_basis": {
                "source": (
                    "Ma, Huang, Zheng, Pan, and Zhong, Transportation Research "
                    "Part A 211 (2026), 105127"
                ),
                "doi": "10.1016/j.tra.2026.105127",
                "reported_loading_to_recovery_capacity_drop_percent": 15.0,
                "transferred_family": "G_gamma(n)=gamma G(n), gamma in [0.85,1]",
                "scope": "calibration-informed transfer test, not calibration of the benchmark network",
            },
            "fixed_elements": (
                "gains, Lyapunov matrices, transition factor, physical constraints, "
                "common SOS level, proposal clock, and integration tolerances"
            ),
            "solver": solver,
            "common_level": common_level,
            "linear_family": linear_records,
            "rate_grid_per_hour": RATE_GRID,
            "rate_trials_descending": rate_trials,
            "selected_family_rate_per_hour": selected_rate,
            "selected_family_dwell_threshold_minutes": (
                60.0 * math.log(transition_factor) / selected_rate
            ),
            "selected_endpoint_certificate": selected,
            "simulations": simulations,
            "safety_audit": safety,
            "effect_scale": effect_scale,
            "convex_interval_argument": (
                "The dissipation polynomial is affine in gamma. Convexly combining "
                "the endpoint SOS multipliers and targets certifies every gamma in "
                "the closed interval at the fixed level and selected rate."
            ),
        }
        OUTPUT_JSON.write_text(
            json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
        )
        make_figure(linear_records, simulations, selected_rate)

        summary = {
            "selected_family_rate_per_hour": selected_rate,
            "family_dwell_threshold_minutes": output[
                "selected_family_dwell_threshold_minutes"
            ],
            "common_tube_inscribed_radius_vehicles": min(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * math.sqrt(common_level / float(np.max(eigvalsh(matrix))))
                for matrix in matrices
            ),
            "safety_audit": safety,
            "effect_scale": effect_scale,
            "runs": {
                gamma: {
                    policy: compact(run)
                    for policy, run in gamma_runs.items()
                }
                for gamma, gamma_runs in simulations.items()
            },
        }
        print(json.dumps(json_ready(summary), indent=2))
    finally:
        restore_nominal()


if __name__ == "__main__":
    main()
