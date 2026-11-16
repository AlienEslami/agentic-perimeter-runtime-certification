#!/usr/bin/env python3
"""Predeclared nonstationary demand-shift study for the cubic-MFD supervisor.

The protocol is fixed in ``MFD_NONSTATIONARY_STUDY_PROTOCOL.md``.  The plant,
controllers, Lyapunov metrics, and SOS tube are inherited unchanged.  A
one-dimensional bounded demand redistribution enters through the exact demand
channel of the six conservation equations.  The nominal SOS flow inequality is
combined with that structured direction to construct a practical robust flow
envelope.  A floor-aware uniform-dwell comparator is used only when its
invariance condition is feasible.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

import mfd_cubic_pilot as pilot
import mfd_cubic_runtime_pilot as runtime


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
SOS_JSON = RESULTS_DIR / "mfd_cubic_sos_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_nonstationary_results.json"
FIGURE_PDF = RESULTS_DIR / "mfd_cubic_nonstationary.pdf"
FIGURE_PNG = RESULTS_DIR / "mfd_cubic_nonstationary.png"
SENSITIVITY_PDF = RESULTS_DIR / "mfd_cubic_nonstationary_sensitivity.pdf"
SENSITIVITY_PNG = RESULTS_DIR / "mfd_cubic_nonstationary_sensitivity.png"
MECHANISM_PDF = RESULTS_DIR / "mfd_cubic_nonstationary_mechanism.pdf"
MECHANISM_PNG = RESULTS_DIR / "mfd_cubic_nonstationary_mechanism.png"

HORIZON_H = 8.0
PRIMARY_AMPLITUDE = 0.006
PRIMARY_INITIAL_OCCUPANCY = 0.8
PRIMARY_PROPOSAL_MINUTES = 15.0
AUDIT_STEP_MINUTES = 1.0
BASE_POLICIES = (
    "unrestricted",
    "landing_only",
    "robust_dwell",
    "robust_envelope",
    "nominal_dwell",
    "nominal_envelope",
)
ADT_POLICIES = (
    "robust_adt_envelope",
    "nominal_adt_envelope",
)
POLICIES = BASE_POLICIES + ADT_POLICIES


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


def linear_segment(time_h: float, start_h: float, end_h: float, left: float, right: float) -> float:
    fraction = (time_h - start_h) / (end_h - start_h)
    return left + fraction * (right - left)


def primary_delta(time_h: float, amplitude: float = PRIMARY_AMPLITUDE) -> float:
    recovery_amplitude = -(2.0 / 3.0) * amplitude
    if time_h < 1.0:
        return 0.0
    if time_h < 2.0:
        return linear_segment(time_h, 1.0, 2.0, 0.0, amplitude)
    if time_h < 3.0:
        return amplitude
    if time_h < 4.0:
        return linear_segment(time_h, 3.0, 4.0, amplitude, recovery_amplitude)
    if time_h < 5.5:
        return recovery_amplitude
    if time_h < 6.5:
        return linear_segment(time_h, 5.5, 6.5, recovery_amplitude, 0.0)
    return 0.0


def primary_priority(time_h: float) -> int:
    # Mode index 1 is the installed Region-1-priority controller.
    if time_h < 3.25:
        return 1
    if time_h < 6.5:
        return 0
    return 1


def block_priority(time_h: float, episode_minutes: float) -> int:
    block = int(math.floor((time_h + 1.0e-10) / (episode_minutes / 60.0)))
    return 1 if block % 2 == 0 else 0


def block_delta(time_h: float, amplitude: float, episode_minutes: float) -> float:
    priority = block_priority(time_h, episode_minutes)
    return amplitude if priority == 1 else -(2.0 / 3.0) * amplitude


def load_installed() -> tuple[tuple[np.ndarray, ...], dict[str, Any], dict[str, Any]]:
    if not PILOT_JSON.exists() or not SOS_JSON.exists():
        raise SystemExit("Run the cubic model and SOS pilot scripts first")
    record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = tuple(
        np.asarray(value, dtype=float)
        for value in record["controller_search"]["selected_gains"]
    )
    certificate = record["certificate"]
    certificate["matrices"] = [
        np.asarray(value, dtype=float) for value in certificate["matrices"]
    ]
    tube = {
        "common_level": float(sos["common_level"]),
        "individual_levels": [
            float(item["level"]) for item in sos["sos_certificates"]
        ],
    }
    return gains, certificate, tube


def demand_channel() -> np.ndarray:
    return pilot.complex_step_jacobian(
        lambda demand: pilot.mfd_rhs(pilot.X_STAR, pilot.U_STAR, demand),
        pilot.DEMAND,
    )


def shift_direction() -> np.ndarray:
    return np.concatenate((pilot.DEMAND[:3], -pilot.DEMAND[3:]))


def robust_record(
    amplitude: float,
    proposal_minutes: float,
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> dict[str, Any]:
    nominal_rate = float(certificate["rate"])
    epsilon = nominal_rate / 2.0
    practical_rate = nominal_rate - epsilon
    induced = demand_channel() @ (amplitude * shift_direction())
    floors = []
    disturbance_rates = []
    for metric in certificate["matrices"]:
        amplitude_squared = float(induced @ metric @ induced)
        disturbance_rate = amplitude_squared / epsilon
        disturbance_rates.append(disturbance_rate)
        floors.append(disturbance_rate / practical_rate)
    common_floor = max(floors)
    transition_factor = float(certificate["transition_factor"])
    common_level = float(tube["common_level"])

    # If a flow interval begins at V<=ell, then just before a switch
    # V^- <= c + exp(-r T)(ell-c), and the worst metric reset gives
    # V^+ <= nu V^-.  A uniform dwell therefore exists only when
    # ell > nu*c; when it exists, solve nu[c+exp(-rT)(ell-c)] <= ell.
    uniform_dwell_feasible = bool(
        common_level > transition_factor * common_floor + 1.0e-12
    )
    proposal_h = proposal_minutes / 60.0
    if uniform_dwell_feasible:
        continuous_dwell = math.log(
            transition_factor
            * (common_level - common_floor)
            / (common_level - transition_factor * common_floor)
        ) / practical_rate
        grid_dwell = (
            math.ceil(continuous_dwell / proposal_h - 1.0e-12) * proposal_h
        )
        post_switch_bound = transition_factor * (
            common_floor
            + math.exp(-practical_rate * grid_dwell)
            * (common_level - common_floor)
        )
    else:
        continuous_dwell = None
        grid_dwell = None
        post_switch_bound = None
    return {
        "amplitude": amplitude,
        "induced_state_derivative_direction": induced,
        "young_split_epsilon": epsilon,
        "practical_flow_rate_per_hour": practical_rate,
        "modewise_disturbance_rates": disturbance_rates,
        "modewise_ultimate_levels": floors,
        "common_ultimate_level": common_floor,
        "common_ultimate_tube_occupancy": common_floor / common_level,
        "tube_strictly_contains_ultimate_level": bool(
            common_floor < common_level - 1.0e-12
        ),
        "uniform_robust_dwell_feasible": uniform_dwell_feasible,
        "uniform_dwell_feasibility_margin": (
            common_level - transition_factor * common_floor
        ),
        "continuous_floor_aware_robust_dwell_h": continuous_dwell,
        "proposal_grid_floor_aware_robust_dwell_h": grid_dwell,
        "proposal_grid_floor_aware_robust_dwell_minutes": (
            None if grid_dwell is None else 60.0 * grid_dwell
        ),
        "sampled_post_switch_level": post_switch_bound,
        "sampled_post_switch_tube_occupancy": (
            None if post_switch_bound is None else post_switch_bound / common_level
        ),
    }


def constraint_margins(
    state: np.ndarray, mode: int, gains: tuple[np.ndarray, ...]
) -> tuple[float, float]:
    _names, margins = pilot.margin_components(state, mode, gains)
    input_margin = float(np.min(margins[:10]))
    state_margin_vehicles = float(
        pilot.ACCUMULATION_UNIT_VEHICLES * np.min(margins[10:])
    )
    return state_margin_vehicles, input_margin


def evaluate_candidate(
    state: np.ndarray,
    requested: int,
    time_h: float,
    initial_level: float,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    robust: dict[str, Any],
    policy: str,
    last_switch_h: float,
    adt_tokens: float | None = None,
) -> tuple[bool, dict[str, float]]:
    error = state - pilot.X_STAR
    candidate_value = float(
        error @ certificate["matrices"][requested] @ error
    )
    state_margin, input_margin = constraint_margins(state, requested, gains)
    admissible = candidate_value <= tube["common_level"] + 1.0e-10
    admissible = admissible and state_margin >= -1.0e-8 and input_margin >= -1.0e-10
    threshold = tube["common_level"]
    if policy == "robust_dwell":
        dwell = robust["proposal_grid_floor_aware_robust_dwell_h"]
        admissible = admissible and dwell is not None
        if dwell is not None:
            admissible = admissible and (
                time_h - last_switch_h + 1.0e-12 >= dwell
            )
    elif policy == "nominal_dwell":
        admissible = admissible and (
            time_h - last_switch_h + 1.0e-12
            >= float(certificate["realized_dwell"])
        )
    elif policy in {"robust_envelope", "robust_adt_envelope"}:
        floor = float(robust["common_ultimate_level"])
        rate = float(robust["practical_flow_rate_per_hour"])
        threshold = floor + math.exp(-rate * time_h) * (initial_level - floor)
        admissible = admissible and candidate_value <= threshold + 1.0e-10
        if policy == "robust_adt_envelope":
            if adt_tokens is None:
                raise ValueError("robust_adt_envelope requires an update budget")
            admissible = admissible and adt_tokens >= 1.0 - 1.0e-12
    elif policy in {"nominal_envelope", "nominal_adt_envelope"}:
        threshold = initial_level * math.exp(-runtime.SELECTED_ENVELOPE_RATE * time_h)
        admissible = admissible and candidate_value <= threshold + 1.0e-10
        if policy == "nominal_adt_envelope":
            if adt_tokens is None:
                raise ValueError("nominal_adt_envelope requires an update budget")
            admissible = admissible and adt_tokens >= 1.0 - 1.0e-12
    elif policy in {"landing_only", "unrestricted"}:
        pass
    else:
        raise ValueError(policy)
    if policy == "unrestricted":
        admissible = True
    return admissible, {
        "candidate_value": candidate_value,
        "candidate_occupancy": candidate_value / tube["common_level"],
        "threshold_value": threshold,
        "threshold_occupancy": threshold / tube["common_level"],
        "state_margin_vehicles": state_margin,
        "input_margin": input_margin,
        "adt_tokens_before_decision": adt_tokens,
    }


def simulate(
    policy: str,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    amplitude: float = PRIMARY_AMPLITUDE,
    initial_occupancy: float = PRIMARY_INITIAL_OCCUPANCY,
    proposal_minutes: float = PRIMARY_PROPOSAL_MINUTES,
    delta_function: Callable[[float], float] | None = None,
    priority_function: Callable[[float], int] | None = None,
    include_trajectory: bool = False,
    adt_average_minutes: float | None = None,
    adt_burst_capacity: int | None = None,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(policy)
    delta_function = delta_function or (lambda time_h: primary_delta(time_h, amplitude))
    priority_function = priority_function or primary_priority
    robust = robust_record(amplitude, proposal_minutes, certificate, tube)
    uses_adt = policy in {"robust_adt_envelope", "nominal_adt_envelope"}
    if uses_adt:
        if adt_average_minutes is None or adt_average_minutes <= 0.0:
            raise ValueError("average-dwell policies require adt_average_minutes > 0")
        if adt_burst_capacity is None or adt_burst_capacity < 1:
            raise ValueError("average-dwell policies require adt_burst_capacity >= 1")
        adt_average_h = adt_average_minutes / 60.0
        adt_flow_rate = (
            float(robust["practical_flow_rate_per_hour"])
            if policy == "robust_adt_envelope"
            else float(certificate["rate"])
        )
        adt_critical_h = math.log(float(certificate["transition_factor"])) / adt_flow_rate
        if adt_average_h <= adt_critical_h + 1.0e-12:
            raise ValueError(
                f"adt_average_minutes must exceed {60.0 * adt_critical_h:.6f} min "
                f"for {policy}"
            )
        adt_decay_margin = adt_flow_rate - math.log(
            float(certificate["transition_factor"])
        ) / adt_average_h
    else:
        adt_average_h = None
        adt_critical_h = None
        adt_decay_margin = None
    if not robust["tube_strictly_contains_ultimate_level"] and policy in {
        "landing_only", "robust_dwell", "robust_envelope", "robust_adt_envelope"
    }:
        # The simulation remains useful as a feasibility diagnostic, but it is
        # not labeled formally robust when the flow floor reaches the boundary.
        robust_profile_feasible = False
    else:
        robust_profile_feasible = True
    if policy == "robust_dwell":
        robust_profile_feasible = (
            robust_profile_feasible
            and robust["uniform_robust_dwell_feasible"]
        )

    state = runtime.state_on_tube(
        runtime.structured_directions()["region_1_heavy"],
        initial_occupancy,
        certificate,
        tube,
        mode=0,
    )
    mode = 0
    initial_level = float(
        (state - pilot.X_STAR)
        @ certificate["matrices"][mode]
        @ (state - pilot.X_STAR)
    )
    last_switch_h = -1.0e9
    executed_times: list[float] = []
    rejected = 0
    no_ops = 0
    decisions = []
    violation = None
    adt_tokens = float(adt_burst_capacity) if uses_adt else None
    adt_last_update_h = 0.0
    adt_token_trace: list[dict[str, float]] = []

    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    deltas = [float(delta_function(0.0))]
    priorities = [int(priority_function(0.0))]

    proposal_h = proposal_minutes / 60.0
    audit_h = AUDIT_STEP_MINUTES / 60.0
    decision_time = 0.0
    terminated = False
    while decision_time < HORIZON_H - 1.0e-12 and not terminated:
        if uses_adt:
            assert adt_tokens is not None
            assert adt_average_h is not None
            assert adt_burst_capacity is not None
            adt_tokens = min(
                float(adt_burst_capacity),
                adt_tokens + (decision_time - adt_last_update_h) / adt_average_h,
            )
            adt_last_update_h = decision_time
        requested = int(priority_function(decision_time))
        if requested == mode:
            no_ops += 1
            decisions.append(
                {
                    "time_h": decision_time,
                    "requested_mode": requested + 1,
                    "outcome": "no_op_reselection",
                    "adt_tokens_before_decision": adt_tokens,
                    "adt_tokens_after_decision": adt_tokens,
                }
            )
        else:
            admissible, diagnostic = evaluate_candidate(
                state,
                requested,
                decision_time,
                initial_level,
                gains,
                certificate,
                tube,
                robust,
                policy,
                last_switch_h,
                adt_tokens,
            )
            if admissible:
                mode = requested
                last_switch_h = decision_time
                executed_times.append(decision_time)
                if uses_adt:
                    assert adt_tokens is not None
                    adt_tokens -= 1.0
                outcome = "executed"
            else:
                rejected += 1
                outcome = "rejected"
            decisions.append(
                {
                    "time_h": decision_time,
                    "requested_mode": requested + 1,
                    "outcome": outcome,
                    "adt_tokens_after_decision": adt_tokens,
                    **diagnostic,
                }
            )
        if uses_adt:
            assert adt_tokens is not None
            adt_token_trace.append(
                {"time_h": decision_time, "tokens_after_decision": adt_tokens}
            )
        modes[-1] = mode

        end_h = min(decision_time + proposal_h, HORIZON_H)
        sample_times = np.arange(decision_time + audit_h, end_h + 0.5 * audit_h, audit_h)
        if sample_times.size == 0 or sample_times[-1] < end_h - 1.0e-12:
            sample_times = np.append(sample_times, end_h)
        sample_times[-1] = end_h

        def right_hand_side(time_h: float, current: np.ndarray) -> np.ndarray:
            demand = pilot.DEMAND + float(delta_function(time_h)) * shift_direction()
            command = pilot.U_STAR + gains[mode] @ (current - pilot.X_STAR)
            return pilot.mfd_rhs(current, command, demand=demand)

        solution = solve_ivp(
            right_hand_side,
            (decision_time, end_h),
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
            state_margin, input_margin = constraint_margins(state, mode, gains)
            times.append(float(time_value))
            states.append(state.copy())
            modes.append(mode)
            deltas.append(float(delta_function(float(time_value))))
            priorities.append(int(priority_function(float(time_value))))
            if state_margin < -1.0e-6 or input_margin < -1.0e-9:
                violation = {
                    "time_h": float(time_value),
                    "state_margin_vehicles": state_margin,
                    "input_margin": input_margin,
                }
                terminated = True
                break
        decision_time = end_h

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    priorities_array = np.asarray(priorities, dtype=int)
    deltas_array = np.asarray(deltas)
    errors = states_array - pilot.X_STAR
    values = np.asarray(
        [
            error @ certificate["matrices"][mode_value] @ error
            for error, mode_value in zip(errors, modes_array)
        ]
    )
    occupancies = values / tube["common_level"]
    floor = float(robust["common_ultimate_level"])
    practical_thresholds = floor + np.exp(
        -float(robust["practical_flow_rate_per_hour"]) * times_array
    ) * (initial_level - floor)
    nominal_thresholds = initial_level * np.exp(
        -runtime.SELECTED_ENVELOPE_RATE * times_array
    )
    commands = np.asarray(
        [pilot.U_STAR + gains[mode_value] @ error for error, mode_value in zip(errors, modes_array)]
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
    state_input_margins = [
        constraint_margins(state_value, int(mode_value), gains)
        for state_value, mode_value in zip(states_array, modes_array)
    ]
    exit_indices = np.flatnonzero(occupancies > 1.0 + 1.0e-9)
    transportation = {
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
    }
    result: dict[str, Any] = {
        "policy": policy,
        "horizon_completed_h": float(times_array[-1]),
        "robust_profile_feasible": robust_profile_feasible,
        "executed_changes": len(executed_times),
        "rejected_changes": rejected,
        "no_op_reselections": no_ops,
        "maximum_tube_occupancy": float(np.max(occupancies)),
        "final_tube_occupancy": float(occupancies[-1]),
        "first_tube_exit_time_h": (
            None if exit_indices.size == 0 else float(times_array[int(exit_indices[0])])
        ),
        "maximum_robust_envelope_ratio": float(np.max(values / practical_thresholds)),
        "maximum_nominal_envelope_ratio": float(np.max(values / nominal_thresholds)),
        "minimum_state_constraint_margin_vehicles": min(item[0] for item in state_input_margins),
        "minimum_input_constraint_margin": min(item[1] for item in state_input_margins),
        "transportation": transportation,
        "physical_violation": violation,
        "decisions": decisions,
    }
    if uses_adt:
        assert adt_average_h is not None
        assert adt_burst_capacity is not None
        count_residual = -math.inf
        audit_points = [0.0, *executed_times, HORIZON_H]
        for left in audit_points:
            for right in audit_points:
                if right + 1.0e-12 < left:
                    continue
                count = sum(left < value <= right + 1.0e-12 for value in executed_times)
                residual = count - (
                    float(adt_burst_capacity) + (right - left) / adt_average_h
                )
                count_residual = max(count_residual, residual)
        result["average_dwell"] = {
            "average_dwell_minutes": adt_average_minutes,
            "burst_capacity": adt_burst_capacity,
            "critical_average_dwell_minutes": 60.0 * float(adt_critical_h),
            "certified_decay_margin_per_hour": adt_decay_margin,
            "maximum_count_bound_residual": count_residual,
            "count_bound_verified": bool(count_residual <= 1.0e-10),
            "minimum_executed_interval_minutes": (
                None
                if len(executed_times) < 2
                else 60.0 * min(np.diff(np.asarray(executed_times)))
            ),
            "token_trace": adt_token_trace,
        }
    if include_trajectory:
        result["trajectory"] = {
            "time_h": times_array,
            "state": states_array,
            "mode": modes_array,
            "priority_mode": priorities_array,
            "demand_shift": deltas_array,
            "tube_occupancy": occupancies,
            "robust_envelope_occupancy": practical_thresholds / tube["common_level"],
            "nominal_envelope_occupancy": nominal_thresholds / tube["common_level"],
        }
    return result


def percent_improvement(candidate: float, comparator: float) -> float:
    return 100.0 * (comparator - candidate) / comparator


def comparison(envelope: dict[str, Any], dwell: dict[str, Any]) -> dict[str, float]:
    env = envelope["transportation"]
    ref = dwell["transportation"]
    return {
        "priority_cost_improvement_percent": percent_improvement(
            env["time_varying_priority_tracking_cost"],
            ref["time_varying_priority_tracking_cost"],
        ),
        "regional_imbalance_improvement_percent": percent_improvement(
            env["regional_imbalance_vehicle_hours"],
            ref["regional_imbalance_vehicle_hours"],
        ),
        "network_vehicle_hours_difference_percent": 100.0
        * (env["vehicle_hours_in_network"] - ref["vehicle_hours_in_network"])
        / ref["vehicle_hours_in_network"],
        "completed_trips_difference_percent": 100.0
        * (env["completed_trips_vehicles"] - ref["completed_trips_vehicles"])
        / ref["completed_trips_vehicles"],
        "priority_match_improvement_percentage_points": 100.0
        * (
            env["fraction_time_executed_mode_matches_priority"]
            - ref["fraction_time_executed_mode_matches_priority"]
        ),
    }


def compact_run(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key
        in {
            "horizon_completed_h",
            "robust_profile_feasible",
            "executed_changes",
            "rejected_changes",
            "no_op_reselections",
            "maximum_tube_occupancy",
            "first_tube_exit_time_h",
            "maximum_robust_envelope_ratio",
            "minimum_state_constraint_margin_vehicles",
            "minimum_input_constraint_margin",
            "transportation",
            "physical_violation",
        }
    }


def timing_audit(
    state: np.ndarray,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    robust: dict[str, Any],
) -> dict[str, float | int]:
    warmup_repeats = 2000
    batches = 9
    repeats_per_batch = 20000

    for _ in range(warmup_repeats):
        evaluate_candidate(
            state,
            1,
            2.0,
            0.8 * tube["common_level"],
            gains,
            certificate,
            tube,
            robust,
            "robust_envelope",
            0.0,
        )

    batch_microseconds: list[float] = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(repeats_per_batch):
            evaluate_candidate(
                state,
                1,
                2.0,
                0.8 * tube["common_level"],
                gains,
                certificate,
                tube,
                robust,
                "robust_envelope",
                0.0,
            )
        elapsed_ns = time.perf_counter_ns() - start
        batch_microseconds.append(elapsed_ns / repeats_per_batch / 1000.0)

    return {
        "warmup_repetitions": warmup_repeats,
        "batches": batches,
        "repetitions_per_batch": repeats_per_batch,
        "total_repetitions": batches * repeats_per_batch,
        "mean_microseconds_per_complete_candidate_check": float(
            np.mean(batch_microseconds)
        ),
        "median_microseconds_per_complete_candidate_check": float(
            np.median(batch_microseconds)
        ),
        "minimum_batch_microseconds_per_complete_candidate_check": float(
            np.min(batch_microseconds)
        ),
        "maximum_batch_microseconds_per_complete_candidate_check": float(
            np.max(batch_microseconds)
        ),
        "batch_microseconds_per_complete_candidate_check": batch_microseconds,
        "decision_period_seconds": PRIMARY_PROPOSAL_MINUTES * 60.0,
    }


def sensitivity_studies(
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> dict[str, Any]:
    def run_pair(**kwargs) -> dict[str, Any]:
        envelope = simulate("robust_envelope", gains, certificate, tube, **kwargs)
        dwell = simulate("robust_dwell", gains, certificate, tube, **kwargs)
        return {
            "robust_envelope": envelope,
            "robust_dwell": dwell,
            "uniform_robust_dwell_feasible": dwell["robust_profile_feasible"],
            "envelope_vs_dwell": (
                comparison(envelope, dwell)
                if dwell["robust_profile_feasible"]
                else None
            ),
        }

    amplitude = {
        str(value): run_pair(amplitude=value)
        for value in (0.0, 0.003, 0.006, 0.009)
    }
    occupancy = {
        str(value): run_pair(initial_occupancy=value)
        for value in (0.4, 0.6, 0.8)
    }
    priority = {}
    for minutes in (30.0, 60.0, 90.0):
        priority[str(minutes)] = run_pair(
            delta_function=lambda time_h, duration=minutes: block_delta(
                time_h, PRIMARY_AMPLITUDE, duration
            ),
            priority_function=lambda time_h, duration=minutes: block_priority(
                time_h, duration
            ),
        )
    proposal = {
        str(minutes): run_pair(proposal_minutes=minutes)
        for minutes in (15.0, 30.0)
    }
    return {
        "one_at_a_time_design": True,
        "demand_shift_amplitude": amplitude,
        "initial_tube_occupancy": occupancy,
        "priority_episode_minutes": priority,
        "proposal_period_minutes": proposal,
    }


def mechanism_studies(
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> dict[str, Any]:
    """Isolate the timing mechanism without retuning plant or controllers.

    At 0.3% redistribution the floor-aware uniform dwell exists, so the
    envelope and dwell policies can be compared as two formally robust
    execution rules.  At the predeclared 0.6% level the common uniform-dwell
    condition is infeasible; only the state-dependent envelope is presented
    as certified, with nominal dwell retained as a diagnostic.
    """

    moderate: dict[str, Any] = {}
    severe: dict[str, Any] = {}
    for minutes in (30.0, 60.0, 90.0):
        priority = lambda time_h, duration=minutes: block_priority(time_h, duration)
        moderate_delta = lambda time_h, duration=minutes: block_delta(
            time_h, 0.003, duration
        )
        envelope = simulate(
            "robust_envelope",
            gains,
            certificate,
            tube,
            amplitude=0.003,
            delta_function=moderate_delta,
            priority_function=priority,
            include_trajectory=True,
        )
        dwell = simulate(
            "robust_dwell",
            gains,
            certificate,
            tube,
            amplitude=0.003,
            delta_function=moderate_delta,
            priority_function=priority,
            include_trajectory=True,
        )
        if not dwell["robust_profile_feasible"]:
            raise RuntimeError("Moderate mechanism case must admit robust dwell")
        moderate[str(minutes)] = {
            "robust_record": robust_record(
                0.003, PRIMARY_PROPOSAL_MINUTES, certificate, tube
            ),
            "robust_envelope": envelope,
            "floor_aware_robust_dwell": dwell,
            "envelope_vs_dwell": comparison(envelope, dwell),
        }

        severe_delta = lambda time_h, duration=minutes: block_delta(
            time_h, PRIMARY_AMPLITUDE, duration
        )
        severe_envelope = simulate(
            "robust_envelope",
            gains,
            certificate,
            tube,
            amplitude=PRIMARY_AMPLITUDE,
            delta_function=severe_delta,
            priority_function=priority,
            include_trajectory=True,
        )
        nominal_dwell = simulate(
            "nominal_dwell",
            gains,
            certificate,
            tube,
            amplitude=PRIMARY_AMPLITUDE,
            delta_function=severe_delta,
            priority_function=priority,
            include_trajectory=True,
        )
        severe[str(minutes)] = {
            "robust_record": robust_record(
                PRIMARY_AMPLITUDE, PRIMARY_PROPOSAL_MINUTES, certificate, tube
            ),
            "robust_envelope": severe_envelope,
            "nominal_dwell_diagnostic": nominal_dwell,
            "envelope_vs_nominal_dwell_diagnostic": comparison(
                severe_envelope, nominal_dwell
            ),
        }
    return {
        "moderate_redistribution_formal_comparison": moderate,
        "primary_redistribution_authority_audit": severe,
    }


def make_primary_figure(runs: dict[str, Any]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), sharex=True)
    trajectory = runs["robust_envelope"]["trajectory"]
    time_h = np.asarray(trajectory["time_h"])
    axes[0, 0].plot(time_h, 100.0 * np.asarray(trajectory["demand_shift"]), color="#4d4d4d")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 0].set_ylabel("Regional demand shift (%)")
    priority = np.asarray(trajectory["priority_mode"]) + 1
    priority_axis = axes[0, 0].twinx()
    priority_axis.step(time_h, priority, where="post", color="#76548f", alpha=0.75)
    priority_axis.set_yticks([1, 2], ["R1", "R2"])
    priority_axis.set_ylabel("Requested priority")

    states = np.asarray(trajectory["state"])
    axes[0, 1].plot(
        time_h,
        100.0 * (states[:, 0] - states[:, 1]),
        color="#35618f",
        label="all admissible policies",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set_ylabel("Regional imbalance (vehicles)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(time_h, trajectory["tube_occupancy"], color="#35618f")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[1, 0].plot(
        time_h,
        trajectory["robust_envelope_occupancy"],
        color="#35618f",
        linestyle=":",
        linewidth=1.2,
        label="robust practical envelope",
    )
    axes[1, 0].set_ylabel("Common-tube occupancy")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05,
        0.82,
        "Sustained-priority audit",
        fontsize=12,
        fontweight="bold",
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].text(
        0.05,
        0.64,
        "All admissible execution rules accept\n"
        "the same three priority changes.\n\n"
        "The state-dependent advantage is tested\n"
        "separately under 30--90 min priority episodes.",
        fontsize=10,
        va="top",
        transform=axes[1, 1].transAxes,
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_mechanism_figure(mechanism: dict[str, Any]) -> None:
    moderate = mechanism["moderate_redistribution_formal_comparison"]
    sixty = moderate["60.0"]
    envelope = sixty["robust_envelope"]["trajectory"]
    dwell = sixty["floor_aware_robust_dwell"]["trajectory"]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=False)

    for label, trajectory, color in (
        ("robust envelope", envelope, "#35618f"),
        ("floor-aware robust dwell", dwell, "#6f9e75"),
    ):
        t = np.asarray(trajectory["time_h"])
        state = np.asarray(trajectory["state"])
        cumulative_imbalance = cumulative_trapezoid(
            100.0 * np.abs(state[:, 0] - state[:, 1]), t, initial=0.0
        )
        axes[0, 0].plot(t, cumulative_imbalance, label=label, color=color)
        axes[0, 1].plot(t, trajectory["tube_occupancy"], label=label, color=color)
    priority = np.asarray(envelope["priority_mode"])
    axes[0, 0].fill_between(
        np.asarray(envelope["time_h"]),
        axes[0, 0].get_ylim()[0],
        axes[0, 0].get_ylim()[1],
        where=priority == 1,
        color="#76548f",
        alpha=0.06,
        step="post",
    )
    axes[0, 0].set_xlabel("Time (h)")
    axes[0, 0].set_ylabel("Cumulative regional imbalance (veh-h)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[0, 1].plot(
        np.asarray(envelope["time_h"]),
        envelope["robust_envelope_occupancy"],
        color="#35618f",
        linestyle=":",
        label="admission envelope",
    )
    axes[0, 1].set_xlabel("Time (h)")
    axes[0, 1].set_ylabel("Common-tube occupancy")
    axes[0, 1].legend(frameon=False, fontsize=8)

    durations = [30.0, 60.0, 90.0]
    priority_gain = [
        moderate[str(value)]["envelope_vs_dwell"]["priority_cost_improvement_percent"]
        for value in durations
    ]
    imbalance_gain = [
        moderate[str(value)]["envelope_vs_dwell"]["regional_imbalance_improvement_percent"]
        for value in durations
    ]
    positions = np.arange(len(durations))
    axes[1, 0].bar(positions - 0.18, priority_gain, 0.36, label="priority cost", color="#35618f")
    axes[1, 0].bar(positions + 0.18, imbalance_gain, 0.36, label="imbalance", color="#6f9e75")
    axes[1, 0].set_xticks(positions, [f"{value:g}" for value in durations])
    axes[1, 0].set_xlabel("Priority episode (min)")
    axes[1, 0].set_ylabel("Envelope improvement over dwell (%)")
    axes[1, 0].legend(frameon=False, fontsize=8)

    moderate_record = moderate["60.0"]["robust_record"]
    severe_record = mechanism["primary_redistribution_authority_audit"]["60.0"]["robust_record"]
    # The governing feasibility ratio is nu*c/ell; recover nu from the
    # reported margin identity ell-nu*c.
    ratios = []
    for record in (moderate_record, severe_record):
        common_level = record["common_ultimate_level"] / record["common_ultimate_tube_occupancy"]
        reset_floor = common_level - record["uniform_dwell_feasibility_margin"]
        ratios.append(reset_floor / common_level)
    axes[1, 1].bar([0, 1], ratios, color=["#6f9e75", "#b24745"])
    axes[1, 1].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[1, 1].set_xticks([0, 1], ["0.3%", "0.6%"])
    axes[1, 1].set_xlabel("Structured demand redistribution")
    axes[1, 1].set_ylabel(r"Uniform-dwell feasibility ratio $\nu c/\ell$")
    axes[1, 1].text(0, ratios[0] + 0.04, "feasible", ha="center", fontsize=9)
    axes[1, 1].text(1, ratios[1] + 0.04, "infeasible", ha="center", fontsize=9)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(MECHANISM_PDF, bbox_inches="tight")
    figure.savefig(MECHANISM_PNG, dpi=220, bbox_inches="tight")
    # Retain the legacy sensitivity names as aliases for downstream LaTeX.
    figure.savefig(SENSITIVITY_PDF, bbox_inches="tight")
    figure.savefig(SENSITIVITY_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    gains, certificate, tube = load_installed()
    robust = robust_record(
        PRIMARY_AMPLITUDE, PRIMARY_PROPOSAL_MINUTES, certificate, tube
    )
    if np.min(pilot.DEMAND - PRIMARY_AMPLITUDE * np.abs(shift_direction())) < -1.0e-12:
        raise RuntimeError("The declared structured demand set contains a negative primitive demand")

    runs = {
        policy: simulate(
            policy,
            gains,
            certificate,
            tube,
            include_trajectory=True,
        )
        for policy in BASE_POLICIES
    }
    sensitivity = sensitivity_studies(gains, certificate, tube)
    mechanism = mechanism_studies(gains, certificate, tube)
    online_timing = timing_audit(
        np.asarray(runs["robust_envelope"]["trajectory"]["state"])[8],
        gains,
        certificate,
        tube,
        robust,
    )
    output = {
        "protocol": "MFD_NONSTATIONARY_STUDY_PROTOCOL.md",
        "scope": "structured demand-shift incident and recovery on the literature-cubic two-region MFD",
        "fixed_elements": (
            "plant, equilibrium, controller pair, Lyapunov metrics, SOS tube, "
            "initial direction, integration tolerances, and policy objectives"
        ),
        "primary": {
            "horizon_h": HORIZON_H,
            "proposal_period_minutes": PRIMARY_PROPOSAL_MINUTES,
            "audit_step_minutes": AUDIT_STEP_MINUTES,
            "maximum_demand_shift_fraction": PRIMARY_AMPLITUDE,
            "initial_tube_occupancy": PRIMARY_INITIAL_OCCUPANCY,
            "robust_record": robust,
            "runs": runs,
            "robust_envelope_vs_nominal_dwell_diagnostic": comparison(
                runs["robust_envelope"], runs["nominal_dwell"]
            ),
        },
        "sensitivity": sensitivity,
        "mechanism_studies": mechanism,
        "online_computation": online_timing,
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
    )
    make_primary_figure(runs)
    make_mechanism_figure(mechanism)

    summary = {
        "robust_record": robust,
        "runs": {
            policy: compact_run(result)
            for policy, result in runs.items()
        },
        "robust_envelope_vs_nominal_dwell_diagnostic": output["primary"][
            "robust_envelope_vs_nominal_dwell_diagnostic"
        ],
        "mechanism_summary": {
            "moderate_redistribution_60_min": {
                "robust_record": mechanism[
                    "moderate_redistribution_formal_comparison"
                ]["60.0"]["robust_record"],
                "envelope": compact_run(mechanism[
                    "moderate_redistribution_formal_comparison"
                ]["60.0"]["robust_envelope"]),
                "dwell": compact_run(mechanism[
                    "moderate_redistribution_formal_comparison"
                ]["60.0"]["floor_aware_robust_dwell"]),
                "comparison": mechanism[
                    "moderate_redistribution_formal_comparison"
                ]["60.0"]["envelope_vs_dwell"],
            },
            "primary_redistribution_60_min": {
                "robust_record": mechanism[
                    "primary_redistribution_authority_audit"
                ]["60.0"]["robust_record"],
                "envelope": compact_run(mechanism[
                    "primary_redistribution_authority_audit"
                ]["60.0"]["robust_envelope"]),
            },
        },
        "online_computation": online_timing,
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
