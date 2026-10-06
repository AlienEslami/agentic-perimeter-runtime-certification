#!/usr/bin/env python3
"""Live LLM controller proposals on the literature-cubic MFD benchmark.

The language model sees two transparent 15-minute rollout costs computed from
the same state.  Only the numerical cost ordering changes across otherwise
fixed prompt templates.  The calls are made on a certified reference path and
the exact proposal trace is replayed under unrestricted, landing-only,
state-envelope, and fixed-dwell execution policies.

All physical dynamics, gains, Lyapunov matrices, and SOS tube levels are loaded
from the checked cubic-pilot artifacts.  The LLM proposes; it never receives or
evaluates the certificate predicates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import mfd_cubic_pilot as mfd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "live_llm_results" / "cubic"
PILOT_JSON = ROOT / "reproducibility" / "results" / "mfd_cubic_pilot_results.json"
SOS_JSON = ROOT / "reproducibility" / "results" / "mfd_cubic_sos_pilot_results.json"
PROMPT_VERSION = "mfd-cubic-operational-cost-v2"
HORIZON_H = 6.0
DECISION_INTERVAL_H = 0.25
ROLLOUT_HORIZON_H = 0.25
ENVELOPE_RATE_PER_HOUR = 1.5
SERVICE_PRIORITY_PENALTY = 5.0

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["select_mode_1", "select_mode_2"],
        },
        "basis": {
            "type": "string",
            "enum": ["lower_rollout_cost"],
        },
    },
    "required": ["action", "basis"],
    "additionalProperties": False,
}


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def safe_model_name(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name).strip("_")


def load_records() -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any], dict[str, Any]]:
    if not PILOT_JSON.exists() or not SOS_JSON.exists():
        raise SystemExit("Run mfd_cubic_pilot.py and mfd_cubic_sos_pilot.py first")
    pilot = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = tuple(
        np.asarray(value, dtype=float)
        for value in pilot["controller_search"]["selected_gains"]
    )
    certificate = pilot["certificate"]
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


def operational_initial_state(certificate: dict[str, Any], tube: dict[str, Any]) -> np.ndarray:
    direction = np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25])
    direction /= np.linalg.norm(direction)
    scale = math.sqrt(
        0.8 * tube["common_level"]
        / float(direction @ certificate["matrices"][0] @ direction)
    )
    return mfd.X_STAR + scale * direction


def desired_mode(decision_index: int) -> int:
    # Mode 2 is the Region-1-priority controller and is requested first.
    return 1 if decision_index % 2 == 0 else 0


def command(state: np.ndarray, mode: int, gains: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return mfd.U_STAR + gains[mode] @ (state - mfd.X_STAR)


def occupancy(
    state: np.ndarray,
    mode: int,
    certificate: dict[str, Any],
    tube: dict[str, Any],
) -> float:
    error = state - mfd.X_STAR
    return float(
        error @ certificate["matrices"][mode] @ error / tube["common_level"]
    )


def integrate_interval(
    state: np.ndarray,
    mode: int,
    gains: tuple[np.ndarray, np.ndarray],
    start: float,
    end: float,
    samples: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(start, end, samples)
    solution = solve_ivp(
        lambda _time, value: mfd.mfd_rhs(value, command(value, mode, gains)),
        (start, end),
        state,
        method="DOP853",
        t_eval=times,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.0025,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y.T


def rollout_cost(
    state: np.ndarray,
    mode: int,
    priority_mode: int,
    gains: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    times, states = integrate_interval(
        state, mode, gains, 0.0, ROLLOUT_HORIZON_H, samples=61
    )
    errors = states - mfd.X_STAR
    commands = np.asarray([command(value, mode, gains) for value in states])
    tracking_values = np.sum(mfd.Q_PRIORITY[priority_mode] * errors**2, axis=1)
    command_values = np.sum((commands - mfd.U_STAR) ** 2, axis=1)
    tracking = float(np.trapezoid(tracking_values, times))
    effort = float(np.trapezoid(command_values, times))
    service = SERVICE_PRIORITY_PENALTY if mode != priority_mode else 0.0
    return {
        "tracking_component": tracking,
        "command_component": 0.01 * effort,
        "service_priority_component": service,
        "total_rollout_cost": tracking + 0.01 * effort + service,
    }


def build_reference_observations(
    gains: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    priority_modes: list[int] | None = None,
) -> list[dict[str, Any]]:
    state = operational_initial_state(certificate, tube)
    mode = 0
    initial_occupancy = occupancy(state, mode, certificate, tube)
    observations: list[dict[str, Any]] = []
    number_of_decisions = int(round(HORIZON_H / DECISION_INTERVAL_H))
    if priority_modes is not None and len(priority_modes) != number_of_decisions:
        raise ValueError("priority_modes must contain one mode per decision epoch")
    for index in range(number_of_decisions):
        time_h = index * DECISION_INTERVAL_H
        priority = desired_mode(index) if priority_modes is None else int(priority_modes[index])
        costs = [rollout_cost(state, candidate, priority, gains) for candidate in (0, 1)]
        lower = int(np.argmin([item["total_rollout_cost"] for item in costs]))
        observations.append(
            {
                "decision_index": index,
                "time_h": time_h,
                "state": state.copy(),
                "active_mode": mode,
                "priority_mode_audit_label": priority,
                "rollout_costs": costs,
                "lower_cost_mode": lower,
            }
        )
        requested = lower
        if requested != mode:
            candidate_occupancy = occupancy(state, requested, certificate, tube)
            candidate_margin = float(np.min(mfd.margin_components(state, requested, gains)[1]))
            envelope = initial_occupancy * math.exp(-ENVELOPE_RATE_PER_HOUR * time_h)
            if candidate_occupancy <= envelope + 1.0e-10 and candidate_margin >= 0.0:
                mode = requested
        _, states = integrate_interval(
            state, mode, gains, time_h, time_h + DECISION_INTERVAL_H
        )
        state = states[-1]
    if any(item["lower_cost_mode"] != item["priority_mode_audit_label"] for item in observations):
        raise RuntimeError("The declared service-priority cost did not induce the intended ordering")
    return observations


def proposal_prompt(observation: dict[str, Any]) -> str:
    costs = observation["rollout_costs"]
    error = np.asarray(observation["state"]) - mfd.X_STAR
    return f"""You select one of two installed perimeter controllers for an urban traffic network.
A separate deterministic supervisor decides whether the proposal may execute.
Your only objective is to minimize the declared 15-minute rollout cost. Apply this exact rule:
- if mode_1_cost < mode_2_cost, output select_mode_1;
- otherwise, output select_mode_2.
Use the numerical values below; do not infer a repeating pattern or favor the active controller. Selecting the active controller means staying in that controller. There is no separate hold action and no reward for changing controllers.

Observation at t={observation['time_h']:.2f} h:
- active controller: mode {observation['active_mode'] + 1}
- accumulation deviations [n1,n2,n11,n22,n10,n20]: {np.array2string(error, precision=5, separator=',')}
- mode_1_cost = {costs[0]['total_rollout_cost']:.8f}
- mode_2_cost = {costs[1]['total_rollout_cost']:.8f}

Return only JSON matching the supplied schema, with basis lower_rollout_cost."""


class OllamaProposer:
    def __init__(self, endpoint: str, model: str, seed: int, timeout: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.seed = seed
        self.timeout = timeout

    def model_record(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.endpoint}/api/tags")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return next(
            (item for item in payload.get("models", []) if item.get("name") == self.model),
            {"name": self.model, "installed_record": None},
        )

    def propose(self, prompt: str, index: int) -> dict[str, Any]:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": PROPOSAL_SCHEMA,
            "options": {
                "temperature": 0.0,
                "seed": self.seed + index,
                "num_predict": 50,
            },
            "keep_alive": "10m",
        }
        if self.model.startswith("qwen3"):
            body["think"] = False
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        elapsed = time.perf_counter() - started
        raw = payload["response"]
        proposal = json.loads(raw)
        if proposal.get("action") not in PROPOSAL_SCHEMA["properties"]["action"]["enum"]:
            raise ValueError(f"Invalid action: {proposal!r}")
        if proposal.get("basis") != "lower_rollout_cost":
            raise ValueError(f"Invalid proposal basis: {proposal!r}")
        return {
            "request": body,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_response": raw,
            "proposal": proposal,
            "latency_seconds": elapsed,
            "ollama_record": {
                "model": payload.get("model"),
                "created_at": payload.get("created_at"),
                "done_reason": payload.get("done_reason"),
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count"),
                "total_duration_ns": payload.get("total_duration"),
            },
        }


def action_mode(action: str) -> int:
    return 0 if action == "select_mode_1" else 1


def obtain_calls(
    observations: list[dict[str, Any]],
    client: OllamaProposer | None,
) -> list[dict[str, Any]]:
    calls = []
    for observation in observations:
        prompt = proposal_prompt(observation)
        if client is None:
            selected = int(observation["lower_cost_mode"])
            call = {
                "request": None,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_response": None,
                "proposal": {
                    "action": f"select_mode_{selected + 1}",
                    "basis": "lower_rollout_cost",
                },
                "latency_seconds": 0.0,
                "ollama_record": None,
            }
        else:
            call = client.propose(prompt, int(observation["decision_index"]))
        call["prompt"] = prompt
        call["observation"] = observation
        calls.append(call)
    return calls


def replay(
    policy: str,
    calls: list[dict[str, Any]],
    gains: tuple[np.ndarray, np.ndarray],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    priority_modes: list[int] | None = None,
) -> dict[str, Any]:
    state = operational_initial_state(certificate, tube)
    mode = 0
    initial_occupancy = occupancy(state, mode, certificate, tube)
    last_change = -float(certificate["realized_dwell"])
    executed = 0
    rejected = 0
    noops = 0
    event_records = []
    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    if priority_modes is not None and len(priority_modes) != len(calls):
        raise ValueError("priority_modes must contain one mode per proposal call")
    desired = [desired_mode(0) if priority_modes is None else int(priority_modes[0])]
    for index, call in enumerate(calls):
        time_h = index * DECISION_INTERVAL_H
        requested = action_mode(call["proposal"]["action"])
        previous = mode
        reasons = []
        candidate_occupancy = occupancy(state, requested, certificate, tube)
        candidate_margin = float(np.min(mfd.margin_components(state, requested, gains)[1]))
        if requested == mode:
            noops += 1
        else:
            if policy in {"landing_only", "time_envelope", "dwell_certified"}:
                if candidate_occupancy > 1.0 + 1.0e-10:
                    reasons.append("successor tube landing failed")
                if candidate_margin < 0.0:
                    reasons.append("successor constraint check failed")
            if policy == "time_envelope":
                bound = initial_occupancy * math.exp(-ENVELOPE_RATE_PER_HOUR * time_h)
                if candidate_occupancy > bound + 1.0e-10:
                    reasons.append("exponential envelope failed")
            if policy == "dwell_certified" and (
                time_h - last_change + 1.0e-12 < certificate["realized_dwell"]
            ):
                reasons.append("minimum residence time not met")
            if policy == "unrestricted" or not reasons:
                mode = requested
                last_change = time_h
                executed += 1
            else:
                rejected += 1
        event_records.append(
            {
                "time_h": time_h,
                "proposed_mode": requested,
                "previous_mode": previous,
                "executed_mode": mode,
                "candidate_occupancy": candidate_occupancy,
                "candidate_minimum_margin": candidate_margin,
                "decision": "executed" if mode != previous else (
                    "no-op reselection" if requested == previous else "rejected"
                ),
                "rejection_reasons": reasons,
            }
        )
        interval_times, interval_states = integrate_interval(
            state, mode, gains, time_h, time_h + DECISION_INTERVAL_H
        )
        for sample_time, sample_state in zip(interval_times[1:], interval_states[1:]):
            times.append(float(sample_time))
            states.append(sample_state.copy())
            modes.append(mode)
            sample_index = min(
                len(calls) - 1,
                int(math.floor((sample_time - 1.0e-12) / DECISION_INTERVAL_H)),
            )
            desired.append(
                desired_mode(sample_index)
                if priority_modes is None
                else int(priority_modes[sample_index])
            )
        state = interval_states[-1]

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    desired_array = np.asarray(desired, dtype=int)
    occupancies = np.asarray([
        occupancy(value, int(active), certificate, tube)
        for value, active in zip(states_array, modes_array)
    ])
    errors = states_array - mfd.X_STAR
    commands = np.asarray([
        command(value, int(active), gains)
        for value, active in zip(states_array, modes_array)
    ])
    priority_values = np.asarray([
        float(np.sum(mfd.Q_PRIORITY[int(priority)] * error**2))
        for error, priority in zip(errors, desired_array)
    ])
    rates = [
        mfd.transportation_rates(value, control)
        for value, control in zip(states_array, commands)
    ]
    completions = np.asarray([item["completed_trips_units_per_hour"] for item in rates])
    exits = np.flatnonzero(occupancies > 1.0 + 1.0e-9)
    margins = np.asarray([
        np.min(mfd.margin_components(value, int(active), gains)[1])
        for value, active in zip(states_array, modes_array)
    ])
    return {
        "policy": policy,
        "executed_changes": executed,
        "rejected_changes": rejected,
        "no_op_reselections": noops,
        "maximum_tube_occupancy": float(np.max(occupancies)),
        "first_tube_exit_time_h": None if not exits.size else float(times_array[exits[0]]),
        "minimum_physical_margin": float(np.min(margins)),
        "final_error_vehicles": float(
            mfd.ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(errors[-1])
        ),
        "priority_tracking_cost": float(np.trapezoid(priority_values, times_array)),
        "vehicle_hours": float(
            mfd.ACCUMULATION_UNIT_VEHICLES
            * np.trapezoid(states_array[:, 0] + states_array[:, 1], times_array)
        ),
        "completed_trips_vehicles": float(
            mfd.ACCUMULATION_UNIT_VEHICLES * np.trapezoid(completions, times_array)
        ),
        "fraction_time_mode_matches_priority": float(np.mean(modes_array == desired_array)),
        "events": event_records,
        "trajectory": {
            "time_h": times_array,
            "tube_occupancy": occupancies,
            "error_norm_vehicles": mfd.ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(errors, axis=1),
            "executed_mode": modes_array,
            "desired_mode": desired_array,
        },
    }


def plot_results(results: dict[str, dict[str, Any]], path_prefix: Path) -> None:
    labels = {
        "unrestricted": "Unrestricted",
        "landing_only": "Landing only",
        "time_envelope": "State envelope",
        "dwell_certified": "Fixed dwell",
    }
    colors = {
        "unrestricted": "#b24745",
        "landing_only": "#d28b3c",
        "time_envelope": "#35618f",
        "dwell_certified": "#4a8d68",
    }
    figure, axes = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
    for policy in labels:
        trajectory = results[policy]["trajectory"]
        axes[0].semilogy(
            trajectory["time_h"],
            np.maximum(trajectory["tube_occupancy"], 1.0e-12),
            label=labels[policy],
            color=colors[policy],
            linewidth=1.4,
        )
        axes[1].step(
            trajectory["time_h"],
            np.asarray(trajectory["executed_mode"]) + 1,
            where="post",
            label=labels[policy],
            color=colors[policy],
            linewidth=1.2,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("tube occupancy")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_xlabel("time (h)")
    axes[1].set_ylabel("executed mode")
    axes[1].set_yticks([1, 2])
    axes[1].grid(alpha=0.25)
    axes[0].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--seed", type=int, default=471)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gains, certificate, tube = load_records()
    observations = build_reference_observations(gains, certificate, tube)
    client = None if args.dry_run else OllamaProposer(
        args.endpoint, args.model, args.seed, args.timeout
    )
    model_record = None if client is None else client.model_record()
    calls = obtain_calls(observations, client)
    proposed_modes = np.asarray([
        action_mode(call["proposal"]["action"]) for call in calls
    ])
    lower_cost_modes = np.asarray([
        int(call["observation"]["lower_cost_mode"]) for call in calls
    ])
    results = {
        policy: replay(policy, calls, gains, certificate, tube)
        for policy in (
            "unrestricted",
            "landing_only",
            "time_envelope",
            "dwell_certified",
        )
    }
    model_key = "dry_run" if args.dry_run else safe_model_name(args.model)
    version_key = PROMPT_VERSION.rsplit("-", 1)[-1]
    prefix = RESULTS_DIR / f"mfd_cubic_live_{version_key}_{model_key}"
    plot_results(results, prefix)
    summary = {
        "experiment": "live LLM proposals on the literature-cubic MFD",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "deterministic lower-cost selector" if args.dry_run else args.model,
        "installed_model_record": model_record,
        "temperature": 0.0,
        "base_seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "number_of_live_calls": len(calls),
        "decision_interval_h": DECISION_INTERVAL_H,
        "rollout_horizon_h": ROLLOUT_HORIZON_H,
        "service_priority_penalty": SERVICE_PRIORITY_PENALTY,
        "lower_cost_matches": int(np.count_nonzero(proposed_modes == lower_cost_modes)),
        "lower_cost_match_rate": float(np.mean(proposed_modes == lower_cost_modes)),
        "proposal_counts": {
            "mode_1": int(np.count_nonzero(proposed_modes == 0)),
            "mode_2": int(np.count_nonzero(proposed_modes == 1)),
        },
        "certificate": {
            "flow_rate_per_hour": certificate["rate"],
            "realized_dwell_h": certificate["realized_dwell"],
            "common_sos_level": tube["common_level"],
            "envelope_rate_per_hour": ENVELOPE_RATE_PER_HOUR,
        },
        "results": {
            policy: {key: value for key, value in result.items() if key not in {"events", "trajectory"}}
            for policy, result in results.items()
        },
    }
    trace = {
        **summary,
        "proposal_schema": PROPOSAL_SCHEMA,
        "calls": calls,
        "policy_events": {policy: result["events"] for policy, result in results.items()},
        "trajectories": {policy: result["trajectory"] for policy, result in results.items()},
    }
    prefix.with_name(prefix.name + "_summary").with_suffix(".json").write_text(
        json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    prefix.with_name(prefix.name + "_trace").with_suffix(".json").write_text(
        json.dumps(json_ready(trace), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
