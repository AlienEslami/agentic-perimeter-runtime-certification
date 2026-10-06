#!/usr/bin/env python3
"""Matched predictive-safety-filter comparator for the cubic-MFD study.

The comparator adapts the model-predictive safety-filter pattern to discrete
controller proposals.  A proposed controller is passed for one proposal
interval only if either installed controller supplies a constraint-compatible
backup trajectory ending in its certified terminal tube.  If no candidate
backup is feasible, the previously stored backup controller is executed.

The main record uses a 30-minute prediction horizon; 15- and
60-minute horizons are retained as sensitivity cases.  This comparator uses
the same nonlinear plant, gains, constraints, initial conditions, proposal
schedules, integration tolerances, and SOS terminal tubes as the paper's
supervisor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import mfd_cubic_pilot as pilot
import mfd_cubic_runtime_pilot as runtime


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS / "mfd_cubic_pilot_results.json"
SOS_JSON = RESULTS / "mfd_cubic_sos_pilot_results.json"
RUNTIME_JSON = RESULTS / "mfd_cubic_runtime_pilot_results.json"
OUTPUT_JSON = RESULTS / "mfd_cubic_predictive_filter_baseline_results.json"
FIGURE_PDF = RESULTS / "mfd_cubic_runtime_with_predictive_baseline.pdf"
FIGURE_PNG = RESULTS / "mfd_cubic_runtime_with_predictive_baseline.png"

HORIZON_GRID_MINUTES = (15.0, 30.0, 60.0)
MAIN_HORIZON_MINUTES = 30.0


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


def load_records() -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any], dict[str, Any]]:
    pilot_record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos_record = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = tuple(
        np.asarray(value, dtype=float)
        for value in pilot_record["controller_search"]["selected_gains"]
    )
    certificate = pilot_record["certificate"]
    certificate["matrices"] = [np.asarray(value, dtype=float) for value in certificate["matrices"]]
    tube = {
        "common_level": float(sos_record["common_level"]),
        "individual_levels": [float(item["level"]) for item in sos_record["sos_certificates"]],
    }
    return gains, certificate, tube


def occupancy(
    state: np.ndarray,
    mode: int,
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> float:
    error = state - pilot.X_STAR
    return float(error @ certificate["matrices"][mode] @ error / tube["common_level"])


def integrate_mode(
    state: np.ndarray,
    mode: int,
    duration_h: float,
    gains: tuple[np.ndarray, np.ndarray],
    sample_interval_h: float = pilot.PROPOSAL_INTERVAL,
) -> tuple[np.ndarray, np.ndarray]:
    if duration_h <= 1.0e-14:
        return np.asarray([0.0]), np.asarray([state.copy()])
    count = max(1, int(math.ceil(duration_h / sample_interval_h)))
    sample_times = np.linspace(0.0, duration_h, count + 1)
    solution = solve_ivp(
        lambda _time, current: pilot.mfd_rhs(
            current, pilot.U_STAR + gains[mode] @ (current - pilot.X_STAR)
        ),
        (0.0, duration_h),
        state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=min(0.0025, duration_h / 10.0),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return sample_times, solution.y.T


def physical_margin_path(
    states: np.ndarray,
    mode: int,
    gains: tuple[np.ndarray, np.ndarray],
) -> float:
    return min(
        float(np.min(pilot.margin_components(state, mode, gains)[1]))
        for state in states
    )


def candidate_backup_record(
    state: np.ndarray,
    candidate_mode: int,
    backup_mode: int,
    horizon_h: float,
    gains: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> dict[str, Any]:
    first_duration = pilot.PROPOSAL_INTERVAL
    _, first_states = integrate_mode(state, candidate_mode, first_duration, gains)
    remaining = max(0.0, horizon_h - first_duration)
    _, backup_states = integrate_mode(first_states[-1], backup_mode, remaining, gains)
    minimum_margin = min(
        physical_margin_path(first_states, candidate_mode, gains),
        physical_margin_path(backup_states, backup_mode, gains),
    )
    terminal_occupancy = occupancy(backup_states[-1], backup_mode, certificate, tube)
    return {
        "candidate_mode": candidate_mode,
        "backup_mode": backup_mode,
        "minimum_predicted_physical_margin": minimum_margin,
        "terminal_tube_occupancy": terminal_occupancy,
        "feasible": minimum_margin >= -1.0e-10 and terminal_occupancy <= 1.0 + 1.0e-10,
    }


def transportation_summary(
    times: np.ndarray,
    states: np.ndarray,
    modes: np.ndarray,
    gains: tuple[np.ndarray, np.ndarray],
    desired_modes: np.ndarray | None,
) -> dict[str, float]:
    commands = np.asarray(
        [pilot.U_STAR + gains[int(mode)] @ (state - pilot.X_STAR) for state, mode in zip(states, modes)]
    )
    rates = [pilot.transportation_rates(state, command) for state, command in zip(states, commands)]
    production = np.asarray([item["network_production_units_per_hour"] for item in rates])
    completions = np.asarray([item["completed_trips_units_per_hour"] for item in rates])
    result = {
        "vehicle_hours_in_network": float(
            pilot.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(states[:, 0] + states[:, 1], times)
        ),
        "excess_vehicle_hours_relative_to_target": float(
            pilot.ACCUMULATION_UNIT_VEHICLES
            * np.trapezoid(states[:, 0] + states[:, 1] - pilot.X_STAR[0] - pilot.X_STAR[1], times)
        ),
        "regional_imbalance_vehicle_hours": float(
            pilot.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(np.abs(states[:, 0] - states[:, 1]), times)
        ),
        "cumulative_network_production_vehicles": float(
            pilot.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(production, times)
        ),
        "completed_trips_vehicles": float(
            pilot.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(completions, times)
        ),
        "quadratic_command_effort": float(
            np.trapezoid(np.sum((commands - pilot.U_STAR) ** 2, axis=1), times)
        ),
    }
    if desired_modes is not None:
        errors = states - pilot.X_STAR
        costs = np.asarray(
            [float(np.sum(pilot.Q_PRIORITY[int(desired)] * error**2)) for error, desired in zip(errors, desired_modes)]
        )
        result["time_varying_priority_tracking_cost"] = float(np.trapezoid(costs, times))
        result["fraction_time_executed_mode_matches_priority"] = float(np.mean(modes == desired_modes))
    return result


def simulate_filter(
    initial_state: np.ndarray,
    gains: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    horizon_minutes: float,
    proposal_mode: Callable[[float, int], int],
    include_trajectory: bool = False,
) -> dict[str, Any]:
    horizon_h = horizon_minutes / 60.0
    state = initial_state.copy()
    mode = 0
    stored_backup = 0
    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    desired = [proposal_mode(0.0, 0)]
    accepted = 0
    rejected = 0
    fallback_changes = 0
    no_ops = 0
    horizon = pilot.SIMULATION_HORIZON
    step = 0
    prediction_records = []
    while times[-1] < horizon - 1.0e-12:
        time = times[-1]
        request = int(proposal_mode(time, step))
        if request == mode:
            selected_mode = mode
            no_ops += 1
        else:
            backups = [
                candidate_backup_record(
                    state, request, backup, horizon_h, gains, certificate, tube
                )
                for backup in (0, 1)
            ]
            feasible = [item for item in backups if item["feasible"]]
            prediction_records.append({"time_h": time, "request": request, "backups": backups})
            if feasible:
                selected = min(feasible, key=lambda item: item["terminal_tube_occupancy"])
                selected_mode = request
                stored_backup = int(selected["backup_mode"])
                accepted += 1
            else:
                selected_mode = stored_backup
                rejected += 1
                if selected_mode != mode:
                    fallback_changes += 1
        end = min(time + pilot.PROPOSAL_INTERVAL, horizon)
        _, segment = integrate_mode(state, selected_mode, end - time, gains)
        state = segment[-1]
        mode = selected_mode
        times.append(end)
        states.append(state.copy())
        modes.append(mode)
        desired.append(proposal_mode(end, step + 1))
        step += 1

    time_array = np.asarray(times)
    state_array = np.asarray(states)
    mode_array = np.asarray(modes, dtype=int)
    desired_array = np.asarray(desired, dtype=int)
    occupancies = np.asarray(
        [occupancy(state_value, int(mode_value), certificate, tube) for state_value, mode_value in zip(state_array, mode_array)]
    )
    margins = [
        float(np.min(pilot.margin_components(state_value, int(mode_value), gains)[1]))
        for state_value, mode_value in zip(state_array, mode_array)
    ]
    exits = np.flatnonzero(occupancies > 1.0 + 1.0e-9)
    changes = int(np.count_nonzero(mode_array[1:] != mode_array[:-1]))
    result = {
        "policy": "predictive_safety_filter",
        "prediction_horizon_minutes": horizon_minutes,
        "terminal_set": "union of the two installed SOS tubes with a held-controller backup",
        "horizon_h": float(time_array[-1]),
        "executed_changes": changes,
        "accepted_requested_changes": accepted,
        "rejected_requested_changes": rejected,
        "fallback_changes": fallback_changes,
        "no_op_reselections": no_ops,
        "maximum_tube_occupancy": float(np.max(occupancies)),
        "first_tube_exit_time_h": None if exits.size == 0 else float(time_array[int(exits[0])]),
        "final_error_vehicles": float(
            pilot.ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(state_array[-1] - pilot.X_STAR)
        ),
        "maximum_error_vehicles": float(
            pilot.ACCUMULATION_UNIT_VEHICLES * np.max(np.linalg.norm(state_array - pilot.X_STAR, axis=1))
        ),
        "minimum_physical_margin": float(min(margins)),
        "transportation": transportation_summary(
            time_array, state_array, mode_array, gains, desired_array
        ),
        "prediction_check_count": len(prediction_records),
    }
    if include_trajectory:
        result["trajectory"] = {
            "time_h": time_array,
            "tube_occupancy": occupancies,
            "error_norm_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(state_array - pilot.X_STAR, axis=1),
            "executed_mode": mode_array,
            "requested_mode": desired_array,
        }
    return result


def main() -> None:
    gains, certificate, tube = load_records()
    closed_loop = tuple(pilot.A + pilot.B @ gain for gain in gains)
    witness_state = pilot.dominant_initial_state(gains, closed_loop, certificate, tube)
    operational_state = runtime.state_on_tube(
        runtime.structured_directions()["region_1_heavy"], 0.8, certificate, tube
    )

    stress_rule = lambda _time, step: step % 2
    operational_rule = lambda time, _step: 1 if int(math.floor((time + 1.0e-10) / 0.25)) % 2 == 0 else 0
    stress = {
        str(minutes): simulate_filter(
            witness_state, gains, certificate, tube, minutes, stress_rule,
            include_trajectory=minutes == MAIN_HORIZON_MINUTES,
        )
        for minutes in HORIZON_GRID_MINUTES
    }
    operational = {
        str(minutes): simulate_filter(
            operational_state, gains, certificate, tube, minutes, operational_rule,
            include_trajectory=minutes == MAIN_HORIZON_MINUTES,
        )
        for minutes in HORIZON_GRID_MINUTES
    }
    output = {
        "method": (
            "one-step proposal pass-through with a held-controller predictive backup "
            "ending in the installed SOS terminal tube"
        ),
        "main_horizon_minutes": MAIN_HORIZON_MINUTES,
        "horizon_sensitivity_minutes": list(HORIZON_GRID_MINUTES),
        "matched_inputs": (
            "nonlinear cubic plant, gains, physical constraints, SOS terminal tubes, "
            "initial states, proposal schedules, and integration tolerances"
        ),
        "stress_runs": stress,
        "operational_priority_runs": operational,
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")

    archived = json.loads(RUNTIME_JSON.read_text(encoding="utf-8"))["witness_runs"]
    labels = {
        "unrestricted_one_minute": "Direct execution",
        "predictive": "Predictive filter",
        "landing_only": "Landing only",
        "time_envelope": "State envelope",
        "dwell_certified": "Fixed dwell",
    }
    colors = {
        "unrestricted_one_minute": "#b24745",
        "predictive": "#76549a",
        "landing_only": "#d28b3c",
        "time_envelope": "#35618f",
        "dwell_certified": "#4a8d68",
    }
    main_predictive = stress[str(MAIN_HORIZON_MINUTES)]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))
    for key in ("unrestricted_one_minute", "landing_only", "time_envelope", "dwell_certified"):
        trajectory = archived[key]["trajectory"]
        axes[0].plot(
            trajectory["time_h"], trajectory["tube_occupancy"],
            label=labels[key], color=colors[key], linewidth=1.35,
        )
    trajectory = main_predictive["trajectory"]
    axes[0].plot(
        trajectory["time_h"], trajectory["tube_occupancy"],
        label=labels["predictive"], color=colors["predictive"], linewidth=1.35,
        linestyle="--",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("time (h)")
    axes[0].set_ylabel("tube occupancy")
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2)
    order = ("unrestricted_one_minute", "predictive", "landing_only", "time_envelope", "dwell_certified")
    changes = [
        main_predictive["executed_changes"] if key == "predictive" else archived[key]["executed_changes"]
        for key in order
    ]
    axes[1].bar(
        [labels[key] for key in order], changes,
        color=[colors[key] for key in order],
    )
    axes[1].set_ylabel("executed controller changes")
    axes[1].tick_params(axis="x", rotation=18, labelsize=7.5)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=240, bbox_inches="tight")
    plt.close(figure)
    compact = {
        "stress": {
            key: {field: value for field, value in run.items() if field != "trajectory"}
            for key, run in stress.items()
        },
        "operational": {
            key: {field: value for field, value in run.items() if field != "trajectory"}
            for key, run in operational.items()
        },
    }
    print(json.dumps(json_ready(compact), indent=2))


if __name__ == "__main__":
    main()
