#!/usr/bin/env python3
"""Matched memory-poisoning audit for the certified MFD action channel.

The experiment makes the agent state explicit.  A trusted memory requests the
Region-2-priority controller (mode 0).  During a declared attack interval, a
memory-poisoning map overwrites the active objective with an alternating false
priority.  The same resulting proposal stream is sent either directly to the
plant or through the installed state-envelope supervisor.

The study evaluates containment of the certified set; leaving that set is not
interpreted as proof of a physical collision or network failure.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

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
OUTPUT_JSON = RESULTS / "mfd_cubic_agent_attack_results.json"
FIGURE_PDF = RESULTS / "mfd_cubic_agent_attack.pdf"
FIGURE_PNG = RESULTS / "mfd_cubic_agent_attack.png"

DECISION_INTERVAL_H = 1.0 / 60.0
HORIZON_H = 6.0
ATTACK_START_H = 0.0
ATTACK_END_H = 4.0
ENVELOPE_RATE_PER_HOUR = 1.5


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


def load_installed() -> tuple[tuple[np.ndarray, ...], dict[str, Any], dict[str, Any]]:
    plant = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = tuple(
        np.asarray(value, dtype=float)
        for value in plant["controller_search"]["selected_gains"]
    )
    certificate = plant["certificate"]
    certificate["matrices"] = [
        np.asarray(value, dtype=float) for value in certificate["matrices"]
    ]
    tube = {
        "common_level": float(sos["common_level"]),
        "individual_levels": [float(item["level"]) for item in sos["sos_certificates"]],
    }
    return gains, certificate, tube


def agent_state(time_h: float, step: int, last_audit: str) -> dict[str, Any]:
    """Return trusted and attack-modified memory plus the resulting action."""
    trusted = {
        "operator_objective": "prioritize_region_2",
        "requested_mode": 0,
        "last_audit_outcome": last_audit,
    }
    attacked = ATTACK_START_H <= time_h < ATTACK_END_H
    modified = dict(trusted)
    if attacked:
        modified.update(
            {
                "operator_objective": (
                    "prioritize_region_2" if step % 2 == 0 else "prioritize_region_1"
                ),
                "requested_mode": step % 2,
                "memory_integrity": "poisoned",
            }
        )
    else:
        modified["memory_integrity"] = "trusted"
    proposal = int(modified["requested_mode"])
    return {
        "trusted_memory": trusted,
        "effective_memory": modified,
        "attack_active": attacked,
        "proposal_mode": proposal,
    }


def occupancy(
    state: np.ndarray,
    mode: int,
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> float:
    error = state - pilot.X_STAR
    return float(
        error
        @ certificate["matrices"][mode]
        @ error
        / tube["common_level"]
    )


def run(
    policy: str,
    initial_state: np.ndarray,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> dict[str, Any]:
    if policy not in {"unrestricted", "certified_envelope"}:
        raise ValueError(policy)

    state = np.asarray(initial_state, dtype=float).copy()
    mode = 0
    initial_occupancy = occupancy(state, mode, certificate, tube)
    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    occupancies = [initial_occupancy]
    proposals = []
    accepted_times: list[float] = []
    rejected_times: list[float] = []
    rejection_reasons: Counter[str] = Counter()
    no_ops = 0
    last_audit = "initial"

    step = 0
    while times[-1] < HORIZON_H - 1.0e-12:
        time_h = times[-1]
        agent = agent_state(time_h, step, last_audit)
        requested = int(agent["proposal_mode"])
        candidate_occupancy = occupancy(state, requested, certificate, tube)
        margin_names, margins = pilot.margin_components(state, requested, gains)
        minimum_margin = float(np.min(margins))
        envelope_level = initial_occupancy * math.exp(-ENVELOPE_RATE_PER_HOUR * time_h)
        envelope_ratio = candidate_occupancy / max(envelope_level, 1.0e-300)

        outcome = "no_op"
        reason = None
        if requested == mode:
            no_ops += 1
        elif policy == "unrestricted":
            mode = requested
            accepted_times.append(time_h)
            outcome = "executed"
        else:
            admissible = True
            if minimum_margin < 0.0:
                admissible = False
                reason = str(margin_names[int(np.argmin(margins))])
            elif candidate_occupancy > 1.0 + 1.0e-10:
                admissible = False
                reason = "successor_tube"
            elif candidate_occupancy > envelope_level + 1.0e-10:
                admissible = False
                reason = "contraction_envelope"
            if admissible:
                mode = requested
                accepted_times.append(time_h)
                outcome = "executed"
            else:
                rejected_times.append(time_h)
                rejection_reasons[str(reason)] += 1
                outcome = "rejected_hold"

        proposals.append(
            {
                "time_h": time_h,
                **agent,
                "pre_decision_mode": int(modes[-1]),
                "executed_mode": mode,
                "candidate_tube_occupancy": candidate_occupancy,
                "admissible_envelope_level": envelope_level,
                "candidate_to_envelope_ratio": envelope_ratio,
                "minimum_candidate_physical_margin": minimum_margin,
                "outcome": outcome,
                "rejection_reason": reason,
            }
        )
        last_audit = outcome

        end = min(time_h + DECISION_INTERVAL_H, HORIZON_H)
        solution = solve_ivp(
            lambda _time, current: pilot.mfd_rhs(
                current,
                pilot.U_STAR + gains[mode] @ (current - pilot.X_STAR),
            ),
            (time_h, end),
            state,
            method="DOP853",
            t_eval=[end],
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=0.0025,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1]
        times.append(end)
        states.append(state.copy())
        modes.append(mode)
        occupancies.append(occupancy(state, mode, certificate, tube))
        step += 1

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    occupancies_array = np.asarray(occupancies)
    commands = np.asarray(
        [
            pilot.U_STAR + gains[int(active)] @ (sample - pilot.X_STAR)
            for sample, active in zip(states_array, modes_array)
        ]
    )
    rates = [
        pilot.transportation_rates(sample, command)
        for sample, command in zip(states_array, commands)
    ]
    completions = np.asarray(
        [record["completed_trips_units_per_hour"] for record in rates]
    )
    physical_margins = [
        float(np.min(pilot.margin_components(sample, int(active), gains)[1]))
        for sample, active in zip(states_array, modes_array)
    ]
    exit_indices = np.flatnonzero(occupancies_array > 1.0 + 1.0e-9)
    attack_change_proposals = [
        event
        for event in proposals
        if event["attack_active"] and event["proposal_mode"] != event["pre_decision_mode"]
    ]
    rejected_attack_proposals = [
        event for event in attack_change_proposals if event["outcome"] == "rejected_hold"
    ]
    outside_envelope = [
        event
        for event in attack_change_proposals
        if event["candidate_to_envelope_ratio"] > 1.0 + 1.0e-9
    ]

    return {
        "policy": policy,
        "executed_changes": len(accepted_times),
        "rejected_changes": len(rejected_times),
        "no_op_reselections": no_ops,
        "attack_change_proposals": len(attack_change_proposals),
        "attack_proposals_outside_envelope": len(outside_envelope),
        "rejected_attack_change_proposals": len(rejected_attack_proposals),
        "rejection_reason_counts": dict(rejection_reasons),
        "maximum_tube_occupancy": float(np.max(occupancies_array)),
        "final_tube_occupancy": float(occupancies_array[-1]),
        "first_tube_exit_time_h": (
            None if exit_indices.size == 0 else float(times_array[int(exit_indices[0])])
        ),
        "minimum_physical_margin": float(np.min(physical_margins)),
        "transportation": {
            "vehicle_hours_in_network": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(states_array[:, 0] + states_array[:, 1], times_array)
            ),
            "regional_imbalance_vehicle_hours": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(np.abs(states_array[:, 0] - states_array[:, 1]), times_array)
            ),
            "completed_trips_vehicles": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(completions, times_array)
            ),
        },
        "events": proposals,
        "trajectory": {
            "time_h": times_array,
            "executed_mode": modes_array,
            "tube_occupancy": occupancies_array,
            "regional_imbalance_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES
            * (states_array[:, 0] - states_array[:, 1]),
        },
    }


def make_figure(payload: dict[str, Any]) -> None:
    unrestricted = payload["runs"]["unrestricted"]
    certified = payload["runs"]["certified_envelope"]
    event_times = np.asarray([item["time_h"] for item in certified["events"]])
    proposed_modes = np.asarray([item["proposal_mode"] for item in certified["events"]])
    ratios = np.asarray([item["candidate_to_envelope_ratio"] for item in certified["events"]])
    accepted = np.asarray([item["outcome"] == "executed" for item in certified["events"]])
    nonidentity = np.asarray(
        [item["proposal_mode"] != item["pre_decision_mode"] for item in certified["events"]]
    )

    figure, axes = plt.subplots(3, 1, figsize=(8.8, 7.4), sharex=True)
    for axis in axes:
        axis.axvspan(ATTACK_START_H, ATTACK_END_H, color="#f3c969", alpha=0.28)
        axis.grid(alpha=0.22)

    axes[0].step(event_times, proposed_modes, where="post", color="#b24745", label="proposal from effective memory")
    axes[0].step(
        certified["trajectory"]["time_h"],
        certified["trajectory"]["executed_mode"],
        where="post",
        color="#35618f",
        linewidth=1.6,
        label="certified executed mode",
    )
    axes[0].set_yticks([0, 1], labels=["Mode 1", "Mode 2"])
    axes[0].set_ylabel("agent action")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    for key, color, label in (
        ("unrestricted", "#b24745", "unrestricted attacked execution"),
        ("certified_envelope", "#35618f", "certified action channel"),
    ):
        trajectory = payload["runs"][key]["trajectory"]
        axes[1].plot(
            trajectory["time_h"],
            trajectory["tube_occupancy"],
            color=color,
            linewidth=1.5,
            label=label,
        )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("tube occupancy")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    accepted_mask = nonidentity & accepted
    rejected_mask = nonidentity & ~accepted
    axes[2].scatter(event_times[accepted_mask], ratios[accepted_mask], s=15, color="#4a8d68", label="executed change")
    axes[2].scatter(event_times[rejected_mask], ratios[rejected_mask], s=15, color="#b24745", marker="x", label="rejected; hold")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("candidate / envelope")
    axes[2].set_xlabel("time (h)")
    axes[2].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if not PILOT_JSON.exists() or not SOS_JSON.exists():
        raise SystemExit("Run the cubic pilot and SOS scripts first")
    gains, certificate, tube = load_installed()
    initial_state = runtime.state_on_tube(
        runtime.structured_directions()["network_heavy"],
        0.8,
        certificate,
        tube,
    )
    runs = {
        name: run(name, initial_state, gains, certificate, tube)
        for name in ("unrestricted", "certified_envelope")
    }
    payload = {
        "experiment": "agent-memory poisoning and certified action channel",
        "agent_state": "X_i^A=(s_i^-,h_i^A,omega_i^A)",
        "attack_map": "effective memory replaces the trusted regional objective during the declared interval",
        "proposal_rule": "the effective objective selects the corresponding installed controller mode",
        "decision_interval_minutes": DECISION_INTERVAL_H * 60.0,
        "attack_interval_h": [ATTACK_START_H, ATTACK_END_H],
        "envelope_rate_per_hour": ENVELOPE_RATE_PER_HOUR,
        "initial_tube_occupancy": 0.8,
        "initial_state": initial_state,
        "runs": runs,
        "audit": {
            "matched_proposal_stream": True,
            "certified_tube_contained": runs["certified_envelope"]["maximum_tube_occupancy"] <= 1.0 + 1.0e-8,
            "certified_physical_constraints_satisfied": runs["certified_envelope"]["minimum_physical_margin"] >= -1.0e-9,
            "unrestricted_left_certified_tube": runs["unrestricted"]["first_tube_exit_time_h"] is not None,
            "unsafe_effect_claim": "departure from the installed certified set",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
    make_figure(payload)
    summary = {
        "attack_interval_h": payload["attack_interval_h"],
        "unrestricted": {key: value for key, value in runs["unrestricted"].items() if key not in {"events", "trajectory"}},
        "certified_envelope": {key: value for key, value in runs["certified_envelope"].items() if key not in {"events", "trajectory"}},
        "audit": payload["audit"],
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
