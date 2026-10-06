#!/usr/bin/env python3
"""Live economic LLM proposal audit for the two-region MFD example.

At each six-minute proposal epoch, a rolling demand forecast and the measured
state are used to evaluate a transparent 30-minute operating-cost proxy for
each installed fixed-gain controller.  A locally hosted language model selects
one of the two controllers.  Reselecting the active controller means staying;
there is no separate ``hold`` label.  The model does not receive the dwell-time
certificate, Lyapunov matrices, tube occupancies, constraint-test results, or a
switching penalty.

Calls are made online on the runtime-certified reference trajectory.  The exact
proposal trace is then replayed with unconditional execution.  The checked-in
deterministic alternating witness is retained as a separate reference.  This is
an auditable example of rational but myopic controller selection under forecast
variation, not a statistical benchmark of language-model quality or traffic
performance.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import mfd_fixed_gain_validation as mfd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "live_llm_results"
BASELINE_RESULTS = ROOT / "reproducibility" / "results"
PROMPT_VERSION = "mfd-economic-proposer-v3"
LIVE_HORIZON = 12.0  # h
FORECAST_HORIZON = 0.5  # h
FORECAST_OSCILLATION_CENTER = 0.05
FORECAST_OSCILLATION_AMPLITUDE = 0.20
FORECAST_NOISE_STD = 0.015
FORECAST_NOISE_AR = 0.25

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["select_mode_1", "select_mode_2"],
        },
        "basis": {
            "type": "string",
            "enum": [
                "lower_predicted_operating_cost",
            ],
        },
    },
    "required": ["action", "basis"],
    "additionalProperties": False,
}


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


class OllamaMFDProposer:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float,
        seed: int,
        timeout: float,
        schema: dict,
        think: bool | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.schema = schema
        self.think = think

    def model_record(self) -> dict:
        request = urllib.request.Request(f"{self.endpoint}/api/tags")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        installed = [
            item for item in payload.get("models", [])
            if item.get("name") == self.model
        ]
        return installed[0] if installed else {
            "name": self.model,
            "installed_record": None,
        }

    def propose(self, prompt: str, decision_index: int) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": self.schema,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed + decision_index,
                "num_predict": 60,
            },
            "keep_alive": "10m",
        }
        if self.think is not None:
            body["think"] = self.think
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
        if proposal.get("action") not in self.schema["properties"]["action"]["enum"]:
            raise ValueError(f"invalid action in model response: {proposal!r}")
        if proposal.get("basis") not in self.schema["properties"]["basis"]["enum"]:
            raise ValueError(f"invalid basis in model response: {proposal!r}")
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
                "load_duration_ns": payload.get("load_duration"),
            },
        }


def rolling_forecast_errors(number_of_steps: int, seed: int) -> np.ndarray:
    """Construct a seeded oscillatory regional forecast-revision stress input.

    The common scalar revision oscillates around five percent with a 20 percent
    amplitude and small autocorrelated noise.  Region 2 receives the opposite
    revision.  This represents an over-correcting rolling forecast: it is a
    declared constructed stress input for persistent rolling-forecast
    over-correction.
    """
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, FORECAST_NOISE_STD, number_of_steps)
    innovation_scale = np.sqrt(1.0 - FORECAST_NOISE_AR**2)
    for step in range(1, number_of_steps):
        noise[step] = (
            FORECAST_NOISE_AR * noise[step - 1]
            + innovation_scale * noise[step]
        )
    oscillation = FORECAST_OSCILLATION_CENTER + (
        FORECAST_OSCILLATION_AMPLITUDE
        * (-1.0) ** np.arange(number_of_steps)
    )
    regional_one = oscillation + noise
    return np.column_stack((regional_one, -regional_one))


def trapezoid(values: np.ndarray, times: np.ndarray) -> float:
    return float(np.trapezoid(values, times))


def controller_economic_forecast(
    decision_time: float,
    state: np.ndarray,
    mode: int,
    regional_errors: np.ndarray,
) -> dict:
    """Return the declared constant-mode 30-minute cost calculation."""
    demand = mfd.DEMAND.copy()
    demand[:3] *= 1.0 + regional_errors[0]
    demand[3:] *= 1.0 + regional_errors[1]
    evaluation_times = np.linspace(
        decision_time, decision_time + FORECAST_HORIZON, 51
    )
    solution = solve_ivp(
        lambda current_time, current_state: mfd.mfd_rhs(
            current_state,
            mfd.U_STAR + mfd.GAINS[mode] @ (current_state - mfd.X_STAR),
            demand=demand,
        ),
        (decision_time, decision_time + FORECAST_HORIZON),
        state,
        method="DOP853",
        t_eval=evaluation_times,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.01,
    )
    assert solution.success
    states = solution.y.T
    commands = np.asarray([
        mfd.U_STAR + mfd.GAINS[mode] @ (sample - mfd.X_STAR)
        for sample in states
    ])
    accumulation_hours = trapezoid(
        states[:, 0] + states[:, 1], evaluation_times
    )
    imbalance_hours = trapezoid(
        np.abs(states[:, 0] - states[:, 1]), evaluation_times
    )
    command_effort = trapezoid(
        np.sum((commands - mfd.U_STAR) ** 2, axis=1), evaluation_times
    )
    completed_units = trapezoid(
        np.asarray([
            mfd.transportation_rates(sample, command)["completed_trips"]
            for sample, command in zip(states, commands)
        ]),
        evaluation_times,
    )
    congestion_cost = 2000.0 * accumulation_hours
    imbalance_cost = 4000.0 * imbalance_hours
    command_cost = 20000.0 * command_effort
    completion_credit = 5000.0 * completed_units
    return {
        "mode": mode,
        "forecast_demand": demand,
        "accumulation_hours": accumulation_hours,
        "imbalance_hours": imbalance_hours,
        "command_effort": command_effort,
        "completed_accumulation_units": completed_units,
        "congestion_cost_dollars": congestion_cost,
        "imbalance_cost_dollars": imbalance_cost,
        "command_cost_dollars": command_cost,
        "completion_credit_dollars": completion_credit,
        "net_operating_cost_dollars": (
            congestion_cost + imbalance_cost + command_cost - completion_credit
        ),
    }


def economic_observation(
    decision_time: float,
    state: np.ndarray,
    regional_errors: np.ndarray,
) -> dict:
    forecasts = [
        controller_economic_forecast(
            decision_time, state, mode, regional_errors
        )
        for mode in (0, 1)
    ]
    lower_cost_mode = int(np.argmin([
        item["net_operating_cost_dollars"] for item in forecasts
    ]))
    return {
        "forecast_horizon_h": FORECAST_HORIZON,
        "regional_forecast_errors": regional_errors,
        "controller_forecasts": forecasts,
        "lower_cost_mode": lower_cost_mode,
        "cost_difference_mode_2_minus_mode_1_dollars": (
            forecasts[1]["net_operating_cost_dollars"]
            - forecasts[0]["net_operating_cost_dollars"]
        ),
    }


def proposal_prompt(
    decision_time: float,
    state: np.ndarray,
    current_mode: int,
    economics: dict,
) -> str:
    error = state - mfd.X_STAR
    current_command = mfd.U_STAR + mfd.GAINS[current_mode] @ error
    forecasts = economics["controller_forecasts"]
    return f"""You are the operational controller-selection agent for a two-region traffic network.
A separate deterministic runtime supervisor--not you--decides whether execution is certified.
Your sole decision objective is to minimize the predicted 30-minute net operating cost. The two controllers have already been evaluated from the same measured state and demand forecast. Compare only the two final dollar costs and select the controller with the smaller numerical value. Do not treat the current mode as a preference and do not infer a different regional priority. Selecting the active controller means stay; selecting the other controller requests a switch. There is no reward for switching itself.

Observation at t={decision_time:.1f} h:
- current controller: mode {current_mode + 1}
- accumulation deviations [n1,n2,n11,n22,n10,n20]: {np.array2string(error, precision=6, separator=',')}
- current perimeter commands [u12,u21,u02,u20]: {np.array2string(current_command, precision=6, separator=',')}
- rolling demand forecast errors [region1,region2]: {np.array2string(economics['regional_forecast_errors'], precision=6, separator=',')}
- predicted net operating cost with mode 1: ${forecasts[0]['net_operating_cost_dollars']:.2f}
- predicted net operating cost with mode 2: ${forecasts[1]['net_operating_cost_dollars']:.2f}

Choose the mode attached to the smaller dollar cost. Return only JSON matching the supplied schema."""


def action_mode(action: str) -> int | None:
    if action == "select_mode_1":
        return 0
    if action == "select_mode_2":
        return 1
    return None


def tube_occupancy(
    state: np.ndarray,
    mode: int,
    certificate: dict,
    tube: dict,
) -> float:
    error = state - mfd.X_STAR
    return float(
        error @ certificate["matrices"][mode] @ error / tube["common_level"]
    )


def append_sample(storage: dict[str, list], time_value, state, mode, certificate, tube):
    names, margins = mfd.margin_components(state, mode)
    command = mfd.U_STAR + mfd.GAINS[mode] @ (state - mfd.X_STAR)
    storage["time"].append(float(time_value))
    storage["state"].append(state.copy())
    storage["mode"].append(int(mode))
    storage["minimum_margin"].append(float(np.min(margins)))
    storage["active_constraint"].append(names[int(np.argmin(margins))])
    storage["occupancy"].append(tube_occupancy(state, mode, certificate, tube))
    storage["command"].append(command)


def integrate_interval(storage, state, mode, start_time, certificate, tube):
    interval_end = start_time + mfd.PROPOSAL_INTERVAL
    evaluation_times = np.linspace(start_time, interval_end, 21)[1:]
    solution = solve_ivp(
        lambda current_time, current_state: mfd.closed_loop_rhs(
            current_time, current_state, mode
        ),
        (start_time, interval_end),
        state,
        method="DOP853",
        t_eval=evaluation_times,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.0025,
    )
    assert solution.success
    for sample_time, sample_state in zip(solution.t, solution.y.T):
        append_sample(
            storage, sample_time, sample_state, mode, certificate, tube
        )
    return solution.y[:, -1]


def finalize_simulation(storage, events, policy, accepted_times):
    result = {
        "policy": policy,
        "time": np.asarray(storage["time"]),
        "state": np.asarray(storage["state"]),
        "mode": np.asarray(storage["mode"], dtype=int),
        "minimum_margin": np.asarray(storage["minimum_margin"]),
        "active_constraint": np.asarray(storage["active_constraint"], dtype="U40"),
        "occupancy": np.asarray(storage["occupancy"]),
        "command": np.asarray(storage["command"]),
        "events": events,
        "accepted_times": accepted_times,
    }
    error_norm = np.linalg.norm(result["state"] - mfd.X_STAR, axis=1)
    result["error_norm"] = error_norm
    above = np.flatnonzero(result["occupancy"] > 1.0 + 1.0e-9)
    negative = np.flatnonzero(result["minimum_margin"] < -1.0e-10)
    result["first_tube_exit_time"] = (
        float(result["time"][above[0]]) if above.size else None
    )
    result["first_constraint_violation_time"] = (
        float(result["time"][negative[0]]) if negative.size else None
    )
    result["maximum_tube_occupancy"] = float(np.max(result["occupancy"]))
    result["minimum_constraint_margin"] = float(np.min(result["minimum_margin"]))
    result["maximum_error_norm"] = float(np.max(error_norm))
    result["final_error_norm"] = float(error_norm[-1])
    result["executed_switches"] = int(np.count_nonzero(np.diff(result["mode"])))
    return result


def live_certified_run(
    client: OllamaMFDProposer,
    certificate: dict,
    tube: dict,
    forecast_errors: np.ndarray,
) -> tuple[dict, list[dict]]:
    state = mfd.dominant_initial_state(certificate, tube)
    mode = 0
    last_switch_step = 0
    accepted_times: list[float] = []
    events: list[dict] = []
    calls: list[dict] = []
    storage = {key: [] for key in (
        "time", "state", "mode", "minimum_margin", "active_constraint",
        "occupancy", "command",
    )}
    append_sample(storage, 0.0, state, mode, certificate, tube)
    number_of_steps = int(round(LIVE_HORIZON / mfd.PROPOSAL_INTERVAL))

    for step in range(number_of_steps):
        current_time = step * mfd.PROPOSAL_INTERVAL
        economics = economic_observation(
            current_time, state, forecast_errors[step]
        )
        prompt = proposal_prompt(current_time, state, mode, economics)
        call = client.propose(prompt, step)
        call.update({
            "decision_index": step,
            "decision_time": current_time,
            "prompt_version": PROMPT_VERSION,
            "observation": {
                "state": state.copy(),
                "current_mode": mode,
                "economics": economics,
            },
        })
        requested_mode = action_mode(call["proposal"]["action"])
        previous_mode = mode
        elapsed = (step - last_switch_step) * mfd.PROPOSAL_INTERVAL
        decision = "hold proposed"
        rejection_reasons: list[str] = []
        candidate = None

        if requested_mode is not None:
            names, margins = mfd.margin_components(state, requested_mode)
            candidate = {
                "mode": requested_mode,
                "minimum_constraint_margin": float(np.min(margins)),
                "active_constraint": names[int(np.argmin(margins))],
                "tube_occupancy": tube_occupancy(
                    state, requested_mode, certificate, tube
                ),
            }
            if requested_mode == mode:
                decision = "requested mode already active"
            else:
                if elapsed + 1.0e-12 < certificate["realized_dwell"]:
                    rejection_reasons.append("minimum residence time not met")
                if candidate["tube_occupancy"] > 1.0 + 1.0e-10:
                    rejection_reasons.append("successor tube landing failed")
                if candidate["minimum_constraint_margin"] < -1.0e-10:
                    rejection_reasons.append("successor constraint check failed")
                if rejection_reasons:
                    decision = "requested change rejected; active mode retained"
                else:
                    mode = requested_mode
                    last_switch_step = step
                    accepted_times.append(current_time)
                    decision = "proposal accepted"

        events.append({
            "time": current_time,
            "economic_lower_cost_mode": economics["lower_cost_mode"],
            "cost_difference_mode_2_minus_mode_1_dollars": economics[
                "cost_difference_mode_2_minus_mode_1_dollars"
            ],
            "proposal": call["proposal"],
            "previous_mode": previous_mode,
            "executed_mode": mode,
            "elapsed_since_last_switch": elapsed,
            "candidate": candidate,
            "decision": decision,
            "rejection_reasons": rejection_reasons,
        })
        calls.append(call)
        append_sample(storage, current_time, state, mode, certificate, tube)
        state = integrate_interval(
            storage, state, mode, current_time, certificate, tube
        )

    return finalize_simulation(
        storage,
        events,
        "runtime-certified online LLM execution",
        accepted_times,
    ), calls


def unrestricted_replay(calls, certificate, tube):
    state = mfd.dominant_initial_state(certificate, tube)
    mode = 0
    accepted_times: list[float] = []
    events: list[dict] = []
    storage = {key: [] for key in (
        "time", "state", "mode", "minimum_margin", "active_constraint",
        "occupancy", "command",
    )}
    append_sample(storage, 0.0, state, mode, certificate, tube)
    for step, call in enumerate(calls):
        current_time = step * mfd.PROPOSAL_INTERVAL
        requested_mode = action_mode(call["proposal"]["action"])
        previous_mode = mode
        if requested_mode is not None:
            mode = requested_mode
        if mode != previous_mode:
            accepted_times.append(current_time)
        names, margins = mfd.margin_components(state, mode)
        events.append({
            "time": current_time,
            "source": "exact live-proposal trace replay",
            "proposal": call["proposal"],
            "previous_mode": previous_mode,
            "executed_mode": mode,
            "candidate": {
                "minimum_constraint_margin": float(np.min(margins)),
                "active_constraint": names[int(np.argmin(margins))],
                "tube_occupancy": tube_occupancy(state, mode, certificate, tube),
            },
        })
        append_sample(storage, current_time, state, mode, certificate, tube)
        state = integrate_interval(
            storage, state, mode, current_time, certificate, tube
        )
    return finalize_simulation(
        storage,
        events,
        "unrestricted matched LLM-trace replay",
        accepted_times,
    )


def deterministic_reference() -> tuple[dict, dict[str, np.ndarray]]:
    with (BASELINE_RESULTS / "mfd_fixed_gain_results.json").open(
        encoding="utf-8"
    ) as stream:
        record = json.load(stream)
    archive = np.load(BASELINE_RESULTS / "mfd_fixed_gain_trajectories.npz")
    times = archive["unconditional_times"]
    occupancies = archive["unconditional_occupancies"]
    first_exit = np.flatnonzero(occupancies > 1.0 + 1.0e-9)
    summary = {
        **record["simulation"]["unconditional"],
        "first_tube_exit_time": (
            float(times[first_exit[0]]) if first_exit.size else None
        ),
        "description": "checked deterministic six-minute alternating witness",
    }
    prefix = times <= LIVE_HORIZON + 1.0e-12
    arrays = {
        "time": times[prefix],
        "state": archive["unconditional_states"][prefix],
        "mode": archive["unconditional_modes"][prefix],
        "occupancy": occupancies[prefix],
    }
    arrays["error_norm"] = np.linalg.norm(arrays["state"] - mfd.X_STAR, axis=1)
    return summary, arrays


def simulation_summary(simulation: dict) -> dict:
    return {
        "policy": simulation["policy"],
        "horizon_h": float(simulation["time"][-1]),
        "executed_switches": simulation["executed_switches"],
        "accepted_switch_times_h": simulation["accepted_times"],
        "first_tube_exit_time_h": simulation["first_tube_exit_time"],
        "first_constraint_violation_time_h": simulation[
            "first_constraint_violation_time"
        ],
        "maximum_tube_occupancy": simulation["maximum_tube_occupancy"],
        "minimum_constraint_margin": simulation["minimum_constraint_margin"],
        "maximum_error_norm": simulation["maximum_error_norm"],
        "final_error_norm": simulation["final_error_norm"],
    }


def write_event_csv(calls, certified, unrestricted, output_prefix):
    with (RESULTS_DIR / f"{output_prefix}_event_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "time_h", "current_mode", "lower_cost_mode",
            "mode_1_cost_dollars", "mode_2_cost_dollars",
            "cost_difference_mode_2_minus_mode_1_dollars", "proposed_action",
            "proposal_basis", "certified_decision", "certified_executed_mode",
            "certified_candidate_occupancy", "certified_candidate_margin",
            "unrestricted_executed_mode", "unrestricted_occupancy", "latency_s",
        ])
        writer.writeheader()
        for call, certified_event, unrestricted_event in zip(
            calls, certified["events"], unrestricted["events"]
        ):
            candidate = certified_event["candidate"] or {}
            economics = call["observation"]["economics"]
            forecasts = economics["controller_forecasts"]
            writer.writerow({
                "time_h": call["decision_time"],
                "current_mode": call["observation"]["current_mode"] + 1,
                "lower_cost_mode": economics["lower_cost_mode"] + 1,
                "mode_1_cost_dollars": forecasts[0][
                    "net_operating_cost_dollars"
                ],
                "mode_2_cost_dollars": forecasts[1][
                    "net_operating_cost_dollars"
                ],
                "cost_difference_mode_2_minus_mode_1_dollars": economics[
                    "cost_difference_mode_2_minus_mode_1_dollars"
                ],
                "proposed_action": call["proposal"]["action"],
                "proposal_basis": call["proposal"]["basis"],
                "certified_decision": certified_event["decision"],
                "certified_executed_mode": certified_event["executed_mode"] + 1,
                "certified_candidate_occupancy": candidate.get("tube_occupancy"),
                "certified_candidate_margin": candidate.get(
                    "minimum_constraint_margin"
                ),
                "unrestricted_executed_mode": unrestricted_event["executed_mode"] + 1,
                "unrestricted_occupancy": unrestricted_event["candidate"][
                    "tube_occupancy"
                ],
                "latency_s": call["latency_seconds"],
            })


def write_trajectories(certified, unrestricted, deterministic, output_prefix):
    arrays = {}
    for prefix, simulation in (
        ("runtime_certified", certified),
        ("unrestricted", unrestricted),
    ):
        for key in (
            "time", "state", "mode", "minimum_margin", "occupancy", "command",
            "error_norm",
        ):
            arrays[f"{prefix}_{key}"] = simulation[key]
    for key, value in deterministic.items():
        arrays[f"deterministic_{key}"] = value
    np.savez_compressed(RESULTS_DIR / f"{output_prefix}_trajectories.npz", **arrays)


def plot_comparison(certified, unrestricted, deterministic, output_prefix):
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    figure, axes = plt.subplots(3, 1, figsize=(7.15, 5.6), sharex=True)
    axes[0].semilogy(
        unrestricted["time"], unrestricted["error_norm"],
        color="#C44E52", linewidth=1.4, label="unrestricted LLM-trace replay",
    )
    axes[0].semilogy(
        certified["time"], certified["error_norm"],
        color="#2C7FB8", linewidth=1.5, label="runtime-certified LLM execution",
    )
    axes[0].semilogy(
        deterministic["time"], deterministic["error_norm"],
        color="#555555", linestyle="--", linewidth=1.0,
        label="deterministic alternating reference",
    )
    axes[0].set_ylabel("physical error norm")
    axes[0].legend(frameon=False, ncol=3, fontsize=6.7)

    proposal_modes = np.array([
        action_mode(event["proposal"]["action"])
        if action_mode(event["proposal"]["action"]) is not None
        else event["previous_mode"]
        for event in certified["events"]
    ])
    proposal_times = np.array([event["time"] for event in certified["events"]])
    axes[1].scatter(
        proposal_times, proposal_modes + 1,
        color="#888888", s=7, marker="x", label="LLM proposal",
    )
    axes[1].step(
        certified["time"], certified["mode"] + 1,
        where="post", color="#2C7FB8", linewidth=1.5,
        label="certified executed mode",
    )
    axes[1].set_yticks([1, 2], ["mode 1", "mode 2"])
    axes[1].set_ylim(0.75, 2.25)
    axes[1].set_ylabel("proposal / execution")
    axes[1].legend(frameon=False, ncol=2, fontsize=7)

    axes[2].semilogy(
        unrestricted["time"], np.maximum(unrestricted["occupancy"], 1.0e-6),
        color="#C44E52", linewidth=1.4, label="unrestricted LLM-trace replay",
    )
    axes[2].semilogy(
        certified["time"], np.maximum(certified["occupancy"], 1.0e-6),
        color="#2C7FB8", linewidth=1.5, label="runtime-certified LLM execution",
    )
    axes[2].semilogy(
        deterministic["time"], np.maximum(deterministic["occupancy"], 1.0e-6),
        color="#555555", linestyle="--", linewidth=1.0,
        label="deterministic alternating reference",
    )
    axes[2].axhline(
        1.0, color="#222222", linestyle=":", linewidth=1.0,
        label="certified boundary",
    )
    axes[2].set_ylabel("active-tube occupancy")
    axes[2].set_xlabel("time (h)")
    axes[2].set_xlim(0.0, LIVE_HORIZON)
    axes[2].legend(frameon=False, ncol=4, fontsize=6.5)
    for accepted_time in certified["accepted_times"]:
        for axis in axes:
            axis.axvline(
                accepted_time, color="#2CA02C", linewidth=0.8,
                linestyle="-.", alpha=0.75,
            )
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / f"{output_prefix}_comparison.png", dpi=240)
    figure.savefig(RESULTS_DIR / f"{output_prefix}_comparison.pdf")
    plt.close(figure)


def plot_compact(certified, unrestricted, calls, output_prefix):
    """Single-column manuscript rendering of the live comparison."""
    plt.rcParams.update({"font.size": 7, "axes.grid": True, "grid.alpha": 0.25})
    figure, axes = plt.subplots(3, 1, figsize=(3.45, 4.7), sharex=True)
    axes[0].semilogy(
        unrestricted["time"], unrestricted["error_norm"],
        color="#C44E52", linewidth=1.25, label="unrestricted replay",
    )
    axes[0].semilogy(
        certified["time"], certified["error_norm"],
        color="#2C7FB8", linewidth=1.35, label="certified execution",
    )
    axes[0].set_ylabel("error norm")
    axes[0].legend(frameon=False, fontsize=5.8, ncol=2, loc="lower left")

    proposal_modes = np.array([
        action_mode(call["proposal"]["action"])
        if action_mode(call["proposal"]["action"]) is not None
        else call["observation"]["current_mode"]
        for call in calls
    ])
    proposal_times = np.array([call["decision_time"] for call in calls])
    axes[1].scatter(
        proposal_times, proposal_modes + 1,
        color="#888888", s=5, marker="x", label="LLM proposal",
    )
    axes[1].step(
        certified["time"], certified["mode"] + 1,
        where="post", color="#2C7FB8", linewidth=1.25,
        label="executed mode",
    )
    axes[1].set_yticks([1, 2], ["mode 1", "mode 2"])
    axes[1].set_ylim(0.75, 2.25)
    axes[1].set_ylabel("proposal / execution")
    axes[1].legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center")

    axes[2].semilogy(
        unrestricted["time"], np.maximum(unrestricted["occupancy"], 1.0e-6),
        color="#C44E52", linewidth=1.25, label="unrestricted replay",
    )
    axes[2].semilogy(
        certified["time"], np.maximum(certified["occupancy"], 1.0e-6),
        color="#2C7FB8", linewidth=1.35, label="certified execution",
    )
    axes[2].axhline(
        1.0, color="#222222", linestyle=":", linewidth=0.9,
        label="tube boundary",
    )
    axes[2].set_ylabel("tube occupancy")
    axes[2].set_xlabel("time (h)")
    axes[2].set_xlim(0.0, LIVE_HORIZON)
    axes[2].text(
        11.7, 1.18, "tube boundary", ha="right", va="bottom", fontsize=5.5
    )
    for accepted_time in certified["accepted_times"]:
        for axis in axes:
            axis.axvline(
                accepted_time, color="#2CA02C", linewidth=0.7,
                linestyle="-.", alpha=0.75,
            )
    figure.tight_layout(pad=0.6)
    figure.savefig(RESULTS_DIR / f"{output_prefix}_compact.png", dpi=260)
    figure.savefig(RESULTS_DIR / f"{output_prefix}_compact.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--forecast-seed", type=int, default=20260812)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    certificate = mfd.independent_certificate()
    tube = mfd.tube_record(certificate)
    output_prefix = "live_llm_mfd_economic"
    client = OllamaMFDProposer(
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        timeout=args.timeout,
        schema=PROPOSAL_SCHEMA,
    )
    model_record = client.model_record()
    number_of_steps = int(round(LIVE_HORIZON / mfd.PROPOSAL_INTERVAL))
    forecast_errors = rolling_forecast_errors(
        number_of_steps, args.forecast_seed
    )
    certified, calls = live_certified_run(
        client, certificate, tube, forecast_errors
    )
    unrestricted = unrestricted_replay(calls, certificate, tube)
    deterministic_summary, deterministic_arrays = deterministic_reference()

    economically_preferred = np.array([
        call["observation"]["economics"]["lower_cost_mode"] for call in calls
    ])
    proposed = np.array([
        action_mode(call["proposal"]["action"]) for call in calls
    ], dtype=object)
    economic_matches = int(sum(
        proposed_mode == preferred_mode
        for proposed_mode, preferred_mode in zip(proposed, economically_preferred)
    ))
    no_op_reselections = int(sum(
        proposed_mode == call["observation"]["current_mode"]
        for proposed_mode, call in zip(proposed, calls)
    ))
    requested_changes = len(calls) - no_op_reselections
    executed_changes = certified["executed_switches"]
    rejected_changes = requested_changes - executed_changes
    plot_comparison(
        certified, unrestricted, deterministic_arrays, output_prefix
    )
    plot_compact(certified, unrestricted, calls, output_prefix)
    write_event_csv(calls, certified, unrestricted, output_prefix)
    write_trajectories(
        certified, unrestricted, deterministic_arrays, output_prefix
    )

    summary = {
        "experiment": (
            "live myopic-economic MFD controller proposals with certified online "
            "execution, matched unrestricted replay, and deterministic reference"
        ),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "temperature": args.temperature,
        "base_seed": args.seed,
        "forecast_seed": args.forecast_seed,
        "prompt_version": PROMPT_VERSION,
        "live_horizon_h": LIVE_HORIZON,
        "forecast_horizon_h": FORECAST_HORIZON,
        "forecast_oscillation_center": FORECAST_OSCILLATION_CENTER,
        "forecast_oscillation_amplitude": FORECAST_OSCILLATION_AMPLITUDE,
        "forecast_noise_standard_deviation": FORECAST_NOISE_STD,
        "forecast_noise_ar_coefficient": FORECAST_NOISE_AR,
        "proposal_interval_h": mfd.PROPOSAL_INTERVAL,
        "number_of_live_calls": len(calls),
        "lower_cost_mode_matches": economic_matches,
        "lower_cost_mode_match_rate": economic_matches / len(calls),
        "economically_preferred_mode_counts": {
            "mode_1": int(np.count_nonzero(economically_preferred == 0)),
            "mode_2": int(np.count_nonzero(economically_preferred == 1)),
        },
        "proposal_counts": {
            "mode_1": int(np.count_nonzero(proposed == 0)),
            "mode_2": int(np.count_nonzero(proposed == 1)),
        },
        "certified_online_proposal_accounting": {
            "no_op_active_mode_reselections": no_op_reselections,
            "requested_mode_changes": requested_changes,
            "executed_mode_changes": executed_changes,
            "rejected_mode_changes": rejected_changes,
        },
        "certificate": {
            "rate": certificate["rate"],
            "transition_factor": certificate["transition_factor"],
            "analytical_dwell_time_h": certificate["dwell_time"],
            "realized_dwell_time_h": certificate["realized_dwell"],
            "sampled_contraction": certificate["sampled_contraction"],
            "common_tube_level": tube["common_level"],
        },
        "summaries": {
            "unrestricted_live_trace_replay": simulation_summary(unrestricted),
            "runtime_certified_live_execution": simulation_summary(certified),
            "deterministic_alternating_reference": deterministic_summary,
        },
        "method_note": (
            "No mode-selection cue is supplied. A declared oscillatory rolling-"
            "forecast stress input changes the two transparent 30-minute cost "
            "estimates; the agent selects a mode under a myopic no-switching-"
            "penalty objective. Calls "
            "are online on the certified reference path; the unrestricted branch "
            "is an exact matched-trace replay, not a second adaptive LLM trajectory."
        ),
    }
    trace = {
        **summary,
        "installed_model_record": model_record,
        "proposal_schema": PROPOSAL_SCHEMA,
        "calls": calls,
        "runtime_certified_events": certified["events"],
        "unrestricted_events": unrestricted["events"],
    }
    with (RESULTS_DIR / f"{output_prefix}_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_ready(summary), stream, indent=2)
    with (RESULTS_DIR / f"{output_prefix}_trace.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_ready(trace), stream, indent=2)
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
