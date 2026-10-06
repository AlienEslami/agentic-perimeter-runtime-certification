#!/usr/bin/env python3
"""Reproduce the certified longitudinal vehicle-following example.

The supervisor sees the full execution state [s, Delta v].  It filters
controller and desired-gap requests using fixed robust invariant ellipsoids.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, solve_continuous_lyapunov


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
EVALUATION_INTERVAL = 0.5
FINAL_TIME = 50.0
MINIMUM_GAP = 5.0
MAXIMUM_ACCELERATION = 3.0
DISTURBANCE_BOUND = 0.2
YOUNG_EPSILON = 0.5
OBJECTIVE_CLASS_INTERVAL = 5.0
OBJECTIVE_CLASS_HORIZON = 200.0
OBJECTIVE_CLASS_TARGETS = (13.5, 13.6)

GAINS = {
    "conservative": np.array([0.25, 0.90]),
    "responsive": np.array([1.30, 2.40]),
}
LEVELS = {"conservative": 16.0, "responsive": 0.5}
TARGETS = {"long": 15.0, "short": 12.5}


def flow_matrix(mode: str) -> np.ndarray:
    ks, kv = GAINS[mode]
    return np.array([[0.0, 1.0], [-ks, -kv]])


def certificate(mode: str) -> dict:
    matrix = flow_matrix(mode)
    p_matrix = solve_continuous_lyapunov(matrix.T, -np.eye(2))
    residual = matrix.T @ p_matrix + p_matrix @ matrix + np.eye(2)
    assert np.max(np.abs(residual)) < 1.0e-12
    assert np.min(eigvalsh(p_matrix)) > 0.0
    assert np.max(np.real(np.linalg.eigvals(matrix))) < 0.0

    p_inverse = np.linalg.inv(p_matrix)
    disturbance_direction = np.array([0.0, 1.0])
    rate = (1.0 - YOUNG_EPSILON) / np.max(eigvalsh(p_matrix))
    disturbance_envelope = (
        np.linalg.norm(p_matrix @ disturbance_direction) ** 2
        * DISTURBANCE_BOUND**2
        / YOUNG_EPSILON
    )
    ultimate_level = disturbance_envelope / rate
    level = LEVELS[mode]
    gain = GAINS[mode]
    maximum_command = float(np.sqrt(level * gain @ p_inverse @ gain))
    maximum_gap_error = float(
        np.sqrt(level * np.array([1.0, 0.0]) @ p_inverse @ np.array([1.0, 0.0]))
    )
    minimum_certified_gap = min(TARGETS.values()) - maximum_gap_error
    boundary_derivative_bound = -rate * level + disturbance_envelope
    assert ultimate_level < level
    assert maximum_command < MAXIMUM_ACCELERATION
    assert minimum_certified_gap > MINIMUM_GAP
    assert boundary_derivative_bound < 0.0
    return {
        "A": matrix,
        "P": p_matrix,
        "eigenvalues_A": np.linalg.eigvals(matrix),
        "eigenvalues_P": eigvalsh(p_matrix),
        "flow_rate": rate,
        "disturbance_envelope": disturbance_envelope,
        "ultimate_level": ultimate_level,
        "installed_level": level,
        "maximum_command_over_tube": maximum_command,
        "minimum_input_margin_over_tube": MAXIMUM_ACCELERATION - maximum_command,
        "maximum_gap_error_over_tube": maximum_gap_error,
        "minimum_gap_over_tube": minimum_certified_gap,
        "boundary_derivative_bound": boundary_derivative_bound,
    }


CERTIFICATES = {mode: certificate(mode) for mode in GAINS}


def disturbance(time: float) -> float:
    """Two bounded lead-acceleration pulses."""
    if 18.0 <= time < 20.0:
        return -DISTURBANCE_BOUND
    if 33.0 <= time < 35.0:
        return DISTURBANCE_BOUND
    return 0.0


def error(state: np.ndarray, target: float) -> np.ndarray:
    return np.array([state[0] - target, state[1]])


def command(state: np.ndarray, mode: str, target: float) -> float:
    return float(GAINS[mode] @ error(state, target))


def occupancy(state: np.ndarray, mode: str, target: float) -> float:
    current_error = error(state, target)
    record = CERTIFICATES[mode]
    return float(
        current_error @ record["P"] @ current_error / record["installed_level"]
    )


def candidate_record(state: np.ndarray, mode: str, target: float) -> dict:
    current_error = error(state, target)
    current_command = command(state, mode, target)
    current_occupancy = occupancy(state, mode, target)
    return {
        "mode": mode,
        "target": target,
        "error": current_error,
        "command": current_command,
        "input_margin": MAXIMUM_ACCELERATION - abs(current_command),
        "gap_margin": state[0] - MINIMUM_GAP,
        "tube_occupancy": current_occupancy,
        "admissible": bool(
            current_occupancy <= 1.0 + 1.0e-12
            and abs(current_command) <= MAXIMUM_ACCELERATION + 1.0e-12
            and state[0] >= MINIMUM_GAP - 1.0e-12
        ),
    }


def dynamics(time: float, state: np.ndarray, mode: str, target: float) -> np.ndarray:
    return np.array(
        [
            state[1],
            -command(state, mode, target) + disturbance(time),
        ]
    )


def largest_admissible_target(
    state: np.ndarray, current_target: float, requested_target: float, mode: str
) -> float:
    """Largest monotone target step whose successor lies in the active tube."""
    if candidate_record(state, mode, requested_target)["admissible"]:
        return requested_target
    assert candidate_record(state, mode, current_target)["admissible"]
    lower, upper = current_target, requested_target
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if candidate_record(state, mode, midpoint)["admissible"]:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def append_sample(
    storage: dict[str, list],
    time: float,
    state: np.ndarray,
    mode: str,
    target: float,
) -> None:
    storage["time"].append(float(time))
    storage["state"].append(state.copy())
    storage["mode"].append(mode)
    storage["target"].append(float(target))
    storage["command"].append(command(state, mode, target))
    storage["occupancy"].append(occupancy(state, mode, target))
    storage["disturbance"].append(disturbance(time))


def simulate(policy: str) -> dict:
    state = np.array([15.0, 0.0])
    mode = "conservative"
    target = TARGETS["long"]
    storage = {
        "time": [],
        "state": [],
        "mode": [],
        "target": [],
        "command": [],
        "occupancy": [],
        "disturbance": [],
    }
    events: list[dict] = []
    append_sample(storage, 0.0, state, mode, target)

    number_of_intervals = int(round(FINAL_TIME / EVALUATION_INTERVAL))
    for step in range(number_of_intervals):
        time = step * EVALUATION_INTERVAL
        changed = False

        if policy == "unrestricted":
            if abs(time - 5.0) < 1.0e-12:
                proposal = candidate_record(
                    state, "responsive", TARGETS["short"]
                )
                mode, target = "responsive", TARGETS["short"]
                events.append(
                    {
                        "time": time,
                        "type": "combined request executed",
                        "candidate": proposal,
                    }
                )
                changed = True
            elif abs(time - 30.0) < 1.0e-12:
                proposal = candidate_record(state, mode, TARGETS["long"])
                target = TARGETS["long"]
                events.append(
                    {
                        "time": time,
                        "type": "target request executed",
                        "candidate": proposal,
                    }
                )
                changed = True
        elif policy == "supervised":
            if abs(time - 5.0) < 1.0e-12:
                combined = candidate_record(
                    state, "responsive", TARGETS["short"]
                )
                repair = candidate_record(
                    state, "conservative", TARGETS["short"]
                )
                assert not combined["admissible"]
                assert repair["admissible"]
                target = TARGETS["short"]
                events.append(
                    {
                        "time": time,
                        "type": "combined request rejected; target-only repair accepted",
                        "rejected_candidate": combined,
                        "accepted_candidate": repair,
                    }
                )
                changed = True
            elif abs(time - 10.0) < 1.0e-12:
                responsive = candidate_record(state, "responsive", target)
                assert responsive["admissible"]
                mode = "responsive"
                events.append(
                    {
                        "time": time,
                        "type": "responsive controller accepted",
                        "accepted_candidate": responsive,
                    }
                )
                changed = True
            elif time >= 30.0 and target < TARGETS["long"] - 1.0e-12:
                direct = candidate_record(state, mode, TARGETS["long"])
                next_target = largest_admissible_target(
                    state, target, TARGETS["long"], mode
                )
                accepted = candidate_record(state, mode, next_target)
                assert accepted["admissible"]
                old_target = target
                target = next_target
                events.append(
                    {
                        "time": time,
                        "type": "target repair step",
                        "requested_target": TARGETS["long"],
                        "previous_target": old_target,
                        "accepted_target": target,
                        "direct_candidate": direct,
                        "accepted_candidate": accepted,
                    }
                )
                changed = target > old_target + 1.0e-12
        else:
            raise ValueError(policy)

        if changed:
            append_sample(storage, time, state, mode, target)

        interval_end = time + EVALUATION_INTERVAL
        evaluation_times = np.linspace(time, interval_end, 51)[1:]
        solution = solve_ivp(
            lambda t, x: dynamics(t, x, mode, target),
            (time, interval_end),
            state,
            method="DOP853",
            t_eval=evaluation_times,
            rtol=1.0e-11,
            atol=1.0e-13,
            max_step=0.005,
        )
        assert solution.success
        for sample_time, sample_state in zip(solution.t, solution.y.T):
            append_sample(storage, sample_time, sample_state, mode, target)
        state = solution.y[:, -1]

    result = {
        key: np.asarray(value)
        for key, value in storage.items()
        if key not in {"mode"}
    }
    result["mode"] = np.asarray(storage["mode"], dtype="U16")
    result["events"] = events
    result["minimum_gap"] = float(np.min(result["state"][:, 0]))
    result["maximum_absolute_command"] = float(np.max(np.abs(result["command"])))
    result["minimum_input_margin"] = (
        MAXIMUM_ACCELERATION - result["maximum_absolute_command"]
    )
    result["maximum_tube_occupancy"] = float(np.max(result["occupancy"]))
    return result


def transition_audit() -> dict:
    p_c = CERTIFICATES["conservative"]["P"]
    p_r = CERTIFICATES["responsive"]["P"]
    conservative_to_responsive = float(np.max(eigvalsh(p_r, p_c)))
    responsive_to_conservative = float(np.max(eigvalsh(p_c, p_r)))
    return {
        "conservative_to_responsive_factor": conservative_to_responsive,
        "responsive_to_conservative_factor": responsive_to_conservative,
        "bidirectional_envelope": max(
            conservative_to_responsive, responsive_to_conservative
        ),
        "target_jump_rule": "e_plus = e_minus + [d_minus-d_plus, 0]",
    }


def objective_class_robust_audit() -> dict:
    """Instantiate the robust MDT and classwise ADT bounds.

    The policy alternates two nearby targets every five seconds under the fixed
    responsive controller and is defined for an infinite continuation.  The
    returned trajectory is a finite numerical prefix of that policy.  Target
    updates reuse the same quadratic certificate, so their directed
    multiplicative factor is one; the target translation is covered by a
    bounded-domain additive transition record.
    """

    mode = "responsive"
    record = CERTIFICATES[mode]
    p_matrix = record["P"]
    level = record["installed_level"]
    flow_rate = record["flow_rate"]
    disturbance_envelope = record["disturbance_envelope"]
    target_increment = abs(OBJECTIVE_CLASS_TARGETS[1] - OBJECTIVE_CLASS_TARGETS[0])
    reset = np.array([target_increment, 0.0])
    reset_metric = float(reset @ p_matrix @ reset)
    kappa_bound = 2.0 * math.sqrt(level * reset_metric) + reset_metric

    beta = math.exp(-flow_rate * OBJECTIVE_CLASS_INTERVAL)
    transition_factor = 1.0
    contraction = transition_factor * beta
    disturbance_floor = disturbance_envelope / flow_rate
    robust_pre_radius = disturbance_floor + beta * kappa_bound / (1.0 - contraction)
    robust_all_radius = robust_pre_radius + kappa_bound
    nominal_pre_radius = beta * kappa_bound / (1.0 - contraction)
    nominal_post_update_bound = nominal_pre_radius + kappa_bound
    assert contraction < 1.0
    assert robust_all_radius < level

    def persistent_disturbance(time: float) -> float:
        return DISTURBANCE_BOUND * math.sin(2.0 * math.pi * time / 7.0)

    def objective_dynamics(time: float, state: np.ndarray, target: float) -> np.ndarray:
        current_error = error(state, target)
        current_command = float(GAINS[mode] @ current_error)
        return np.array(
            [state[1], -current_command + persistent_disturbance(time)]
        )

    state = np.array([OBJECTIVE_CLASS_TARGETS[0], 0.0])
    target = OBJECTIVE_CLASS_TARGETS[0]
    times = [0.0]
    states = [state.copy()]
    targets = [target]
    certificate_values = [0.0]
    pre_event_values: list[float] = []
    post_event_values: list[float] = []
    event_times: list[float] = []
    available_budget = 1.0
    minimum_pre_event_budget = available_budget
    refill_rate = 1.0 / OBJECTIVE_CLASS_INTERVAL
    number_of_intervals = int(round(OBJECTIVE_CLASS_HORIZON / OBJECTIVE_CLASS_INTERVAL))

    for interval_index in range(number_of_intervals):
        interval_start = interval_index * OBJECTIVE_CLASS_INTERVAL
        interval_end = (interval_index + 1) * OBJECTIVE_CLASS_INTERVAL
        available_budget = min(
            1.0,
            available_budget + refill_rate * OBJECTIVE_CLASS_INTERVAL,
        )
        solution = solve_ivp(
            lambda time, value: objective_dynamics(time, value, target),
            (interval_start, interval_end),
            state,
            method="DOP853",
            t_eval=np.linspace(interval_start, interval_end, 501)[1:],
            rtol=1.0e-11,
            atol=1.0e-13,
            max_step=0.01,
        )
        assert solution.success
        for sample_time, sample_state in zip(solution.t, solution.y.T):
            sample_error = error(sample_state, target)
            sample_value = float(sample_error @ p_matrix @ sample_error)
            times.append(float(sample_time))
            states.append(sample_state.copy())
            targets.append(target)
            certificate_values.append(sample_value)
        state = solution.y[:, -1]
        pre_error = error(state, target)
        pre_value = float(pre_error @ p_matrix @ pre_error)

        minimum_pre_event_budget = min(minimum_pre_event_budget, available_budget)
        assert available_budget >= 1.0 - 1.0e-12
        available_budget -= 1.0
        target = OBJECTIVE_CLASS_TARGETS[(interval_index + 1) % 2]
        post_error = error(state, target)
        post_value = float(post_error @ p_matrix @ post_error)
        assert post_value <= pre_value + kappa_bound + 1.0e-10
        assert post_value <= level + 1.0e-10
        event_times.append(interval_end)
        pre_event_values.append(pre_value)
        post_event_values.append(post_value)
        times.append(interval_end)
        states.append(state.copy())
        targets.append(target)
        certificate_values.append(post_value)

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    targets_array = np.asarray(targets)
    values_array = np.asarray(certificate_values)
    commands_array = np.array(
        [command(state_value, mode, target_value) for state_value, target_value in zip(states_array, targets_array)]
    )
    event_times_array = np.asarray(event_times)
    assert np.min(np.diff(event_times_array)) >= OBJECTIVE_CLASS_INTERVAL - 1.0e-12
    assert minimum_pre_event_budget >= 1.0 - 1.0e-12

    realized_pre_maximum = float(np.max(pre_event_values))
    realized_all_maximum = float(np.max(values_array))
    assert realized_pre_maximum <= robust_pre_radius + 1.0e-10
    assert realized_all_maximum <= robust_all_radius + 1.0e-10
    assert np.max(np.abs(commands_array)) <= MAXIMUM_ACCELERATION + 1.0e-10
    assert np.min(states_array[:, 0]) >= MINIMUM_GAP - 1.0e-10

    return {
        "profile": {
            "description": (
                "indefinitely extendable alternating target class under the fixed responsive controller"
            ),
            "finite_prefix_horizon": OBJECTIVE_CLASS_HORIZON,
            "minimum_and_average_update_interval": OBJECTIVE_CLASS_INTERVAL,
            "chatter_bound": 1.0,
            "available_update_budget_capacity": 1.0,
            "available_update_budget_refill_rate": refill_rate,
            "minimum_pre_event_available_budget": minimum_pre_event_budget,
            "targets": OBJECTIVE_CLASS_TARGETS,
            "target_increment": target_increment,
            "directed_transition_factor": transition_factor,
            "bounded_domain_kappa": kappa_bound,
        },
        "corollary_2_adt": {
            "effective_rate": flow_rate - math.log(transition_factor) / OBJECTIVE_CLASS_INTERVAL,
            "event_count_bound": "N(t,s) <= 1 + (t-s)/5",
        },
        "theorem_2_nominal_mdt": {
            "pre_update_radius": nominal_pre_radius,
            "post_update_bound": nominal_post_update_bound,
            "interpretation": "zero-disturbance specialization of the same nonzero-offset recurring objective class",
        },
        "theorem_3_robust_mdt": {
            "flow_rate": flow_rate,
            "disturbance_envelope": disturbance_envelope,
            "disturbance_floor": disturbance_floor,
            "beta": beta,
            "contraction_factor": contraction,
            "pre_update_radius": robust_pre_radius,
            "all_time_radius": robust_all_radius,
        },
        "realized_prefix": {
            "number_of_events": len(event_times),
            "maximum_pre_update_value": realized_pre_maximum,
            "maximum_all_time_value": realized_all_maximum,
            "maximum_tube_occupancy": realized_all_maximum / level,
            "minimum_gap": float(np.min(states_array[:, 0])),
            "minimum_input_margin": float(
                MAXIMUM_ACCELERATION - np.max(np.abs(commands_array))
            ),
        },
        "trajectory": {
            "time": times_array,
            "state": states_array,
            "target": targets_array,
            "certificate_value": values_array,
            "command": commands_array,
            "event_time": event_times_array,
            "pre_event_value": np.asarray(pre_event_values),
            "post_event_value": np.asarray(post_event_values),
        },
    }


def plot_results(unrestricted: dict, supervised: dict) -> None:
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    figure, axis = plt.subplots(figsize=(3.5, 2.55))
    axis.plot(
        unrestricted["time"],
        unrestricted["state"][:, 0],
        linewidth=1.6,
        label="unrestricted execution",
    )
    axis.plot(
        supervised["time"],
        supervised["state"][:, 0],
        linewidth=1.8,
        label="runtime-certified execution",
    )
    requested_target = np.where(
        supervised["time"] < 5.0,
        TARGETS["long"],
        np.where(supervised["time"] < 30.0, TARGETS["short"], TARGETS["long"]),
    )
    axis.step(
        supervised["time"],
        requested_target,
        where="post",
        color="#777777",
        linestyle=":",
        linewidth=1.4,
        label="requested gap",
    )
    axis.step(
        supervised["time"],
        supervised["target"],
        where="post",
        color="#2CA02C",
        linestyle="--",
        linewidth=1.5,
        label="certified gap",
    )
    axis.axhline(
        MINIMUM_GAP, color="#C44E52", linestyle="-.", linewidth=1.2, label="minimum gap"
    )
    axis.set_xlabel("time (s)")
    axis.set_ylabel("bumper-to-bumper gap (m)")
    axis.set_xlim(0.0, FINAL_TIME)
    axis.legend(frameon=False, ncol=2, fontsize=6.2)
    figure.tight_layout()
    figure.savefig(ROOT / "vehicle_gap_tracking.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.5, 2.55))
    for result, label in (
        (unrestricted, "unrestricted execution"),
        (supervised, "runtime-certified execution"),
    ):
        axis.plot(
            result["time"],
            MAXIMUM_ACCELERATION - np.abs(result["command"]),
            linewidth=1.8,
            label=label,
        )
    axis.axhline(0.0, color="#C44E52", linestyle=":", linewidth=1.4)
    axis.set_xlabel("time (s)")
    axis.set_ylabel(r"acceleration margin (m/s$^2$)")
    axis.set_xlim(0.0, FINAL_TIME)
    axis.legend(frameon=False, fontsize=6.8)
    figure.tight_layout()
    figure.savefig(ROOT / "vehicle_acceleration_margin.png", dpi=220)
    plt.close(figure)


def json_ready(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [
                {"real": float(item.real), "imag": float(item.imag)}
                for item in value
            ]
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
    unrestricted = simulate("unrestricted")
    supervised = simulate("supervised")
    transitions = transition_audit()
    objective_class = objective_class_robust_audit()

    assert unrestricted["minimum_input_margin"] < 0.0
    assert supervised["minimum_input_margin"] > 0.0
    assert supervised["minimum_gap"] > MINIMUM_GAP
    assert supervised["maximum_tube_occupancy"] <= 1.0 + 3.0e-9
    final_governor_event = [
        event
        for event in supervised["events"]
        if event["type"] == "target repair step"
    ][-1]
    assert abs(final_governor_event["accepted_target"] - TARGETS["long"]) < 1.0e-10
    plot_results(unrestricted, supervised)

    summary = {
        "unrestricted": {
            key: unrestricted[key]
            for key in (
                "minimum_gap",
                "maximum_absolute_command",
                "minimum_input_margin",
                "maximum_tube_occupancy",
                "events",
            )
        },
        "supervised": {
            key: supervised[key]
            for key in (
                "minimum_gap",
                "maximum_absolute_command",
                "minimum_input_margin",
                "maximum_tube_occupancy",
                "events",
            )
        },
        "final_certified_target_time": final_governor_event["time"],
    }
    output = {
        "model": {
            "state_order": ["s", "Delta v"],
            "disturbance_bound": DISTURBANCE_BOUND,
            "minimum_gap": MINIMUM_GAP,
            "maximum_acceleration": MAXIMUM_ACCELERATION,
            "evaluation_interval": EVALUATION_INTERVAL,
            "gains": GAINS,
            "targets": TARGETS,
            "disturbance_profile": [
                {"interval": [18.0, 20.0], "value": -DISTURBANCE_BOUND},
                {"interval": [33.0, 35.0], "value": DISTURBANCE_BOUND},
            ],
        },
        "certificates": CERTIFICATES,
        "transitions": transitions,
        "simulation": summary,
        "objective_class_robust_audit": {
            key: value
            for key, value in objective_class.items()
            if key != "trajectory"
        },
    }
    with open(
        RESULTS_DIR / "vehicle_following_results.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(json_ready(output), handle, indent=2)
    np.savez_compressed(
        RESULTS_DIR / "vehicle_following_trajectories.npz",
        unrestricted_time=unrestricted["time"],
        unrestricted_state=unrestricted["state"],
        unrestricted_mode=unrestricted["mode"],
        unrestricted_target=unrestricted["target"],
        unrestricted_command=unrestricted["command"],
        unrestricted_occupancy=unrestricted["occupancy"],
        supervised_time=supervised["time"],
        supervised_state=supervised["state"],
        supervised_mode=supervised["mode"],
        supervised_target=supervised["target"],
        supervised_command=supervised["command"],
        supervised_occupancy=supervised["occupancy"],
        objective_class_time=objective_class["trajectory"]["time"],
        objective_class_state=objective_class["trajectory"]["state"],
        objective_class_target=objective_class["trajectory"]["target"],
        objective_class_certificate_value=objective_class["trajectory"]["certificate_value"],
        objective_class_command=objective_class["trajectory"]["command"],
        objective_class_event_time=objective_class["trajectory"]["event_time"],
        objective_class_pre_event_value=objective_class["trajectory"]["pre_event_value"],
        objective_class_post_event_value=objective_class["trajectory"]["post_event_value"],
    )

    print("Certificate levels:", {key: value["installed_level"] for key, value in CERTIFICATES.items()})
    print("Transition factors:", transitions)
    print("Unrestricted minimum input margin:", unrestricted["minimum_input_margin"])
    print("Supervised minimum gap:", supervised["minimum_gap"])
    print("Supervised minimum input margin:", supervised["minimum_input_margin"])
    print("Supervised maximum tube occupancy:", supervised["maximum_tube_occupancy"])
    print("Final certified target time:", final_governor_event["time"])
    print(
        "Objective-class robust radii:",
        objective_class["theorem_3_robust_mdt"]["pre_update_radius"],
        objective_class["theorem_3_robust_mdt"]["all_time_radius"],
    )
    print(
        "Objective-class realized maxima:",
        objective_class["realized_prefix"]["maximum_pre_update_value"],
        objective_class["realized_prefix"]["maximum_all_time_value"],
    )


if __name__ == "__main__":
    main()
