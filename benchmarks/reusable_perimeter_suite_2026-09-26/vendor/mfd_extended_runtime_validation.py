#!/usr/bin/env python3
"""Extended runtime-supervision comparisons for the application-centered paper.

The script consumes the checked live MFD proposal trace and evaluates alternative
execution authorities on the same proposal labels.  It also supports a genuinely
live unrestricted branch and a greedy, certificate-aware adversarial proposer.
Every policy uses the same nonlinear plant, initial condition, proposal cadence,
and installed controller library.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import live_llm_mfd_validation as live
import mfd_fixed_gain_validation as mfd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
LIVE_RESULTS_DIR = ROOT / "live_llm_results"
TRACE_PATH = LIVE_RESULTS_DIR / "live_llm_mfd_economic_trace.json"
PREDICTIVE_HORIZON = 0.5  # h


def load_checked_calls() -> list[dict]:
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    calls = payload["calls"]
    if len(calls) != 120:
        raise RuntimeError(f"expected 120 checked calls, found {len(calls)}")
    return calls


def storage_template() -> dict[str, list]:
    return {key: [] for key in (
        "time", "state", "mode", "minimum_margin", "active_constraint",
        "occupancy", "command",
    )}


def economic_improvement(call: dict, current_mode: int, requested_mode: int) -> float:
    forecasts = call["observation"]["economics"]["controller_forecasts"]
    costs = [float(record["net_operating_cost_dollars"]) for record in forecasts]
    return costs[current_mode] - costs[requested_mode]


def physical_prediction(
    state: np.ndarray,
    mode: int,
    start_time: float,
    certificate: dict | None = None,
    tube: dict | None = None,
) -> dict:
    sample_times = np.linspace(start_time, start_time + PREDICTIVE_HORIZON, 51)
    solution = solve_ivp(
        lambda time_value, state_value: mfd.closed_loop_rhs(
            time_value, state_value, mode
        ),
        (sample_times[0], sample_times[-1]),
        state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.01,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    minimum_margin = math.inf
    for sample in solution.y.T:
        _, margins = mfd.margin_components(sample, mode)
        minimum_margin = min(minimum_margin, float(np.min(margins)))
    result = {
        "minimum_predicted_physical_margin": minimum_margin,
        "terminal_state": solution.y[:, -1],
    }
    if certificate is not None and tube is not None:
        result["terminal_tube_occupancy"] = live.tube_occupancy(
            solution.y[:, -1], mode, certificate, tube
        )
    return result


def transportation_summary(simulation: dict) -> dict:
    times = simulation["time"]
    states = simulation["state"]
    commands = simulation["command"]
    rates = [mfd.transportation_rates(state, command) for state, command in zip(states, commands)]
    production = np.asarray([record["network_production"] for record in rates])
    completions = np.asarray([record["completed_trips"] for record in rates])
    accumulations = states[:, 0] + states[:, 1]
    return {
        "vehicle_hours_in_network": float(100.0 * np.trapezoid(accumulations, times)),
        "completed_trips_vehicles": float(100.0 * np.trapezoid(completions, times)),
        "cumulative_network_production_vehicles": float(100.0 * np.trapezoid(production, times)),
        "peak_total_accumulation_vehicles": float(100.0 * np.max(accumulations)),
        "mean_total_accumulation_vehicles": float(100.0 * np.trapezoid(accumulations, times) / (times[-1] - times[0])),
    }


def summarize(simulation: dict) -> dict:
    decisions = Counter(event.get("decision", "") for event in simulation["events"])
    reasons = Counter(
        reason
        for event in simulation["events"]
        for reason in event.get("rejection_reasons", [])
    )
    return {
        **live.simulation_summary(simulation),
        "decision_counts": dict(decisions),
        "rejection_reason_counts": dict(reasons),
        "transportation": transportation_summary(simulation),
    }


def replay_policy(
    calls: list[dict],
    certificate: dict,
    tube: dict,
    policy: str,
    hysteresis_dollars: float = 0.0,
    disabled_predicates: frozenset[str] = frozenset(),
) -> dict:
    state = mfd.dominant_initial_state(certificate, tube)
    mode = 0
    last_switch_step = 0
    accepted_times: list[float] = []
    events: list[dict] = []
    storage = storage_template()
    live.append_sample(storage, 0.0, state, mode, certificate, tube)

    for step, call in enumerate(calls):
        current_time = step * mfd.PROPOSAL_INTERVAL
        requested_mode = live.action_mode(call["proposal"]["action"])
        previous_mode = mode
        elapsed = (step - last_switch_step) * mfd.PROPOSAL_INTERVAL
        reasons: list[str] = []
        candidate: dict | None = None
        decision = "requested mode already active"

        if requested_mode is not None and requested_mode != mode:
            names, margins = mfd.margin_components(state, requested_mode)
            candidate = {
                "mode": requested_mode,
                "tube_occupancy": live.tube_occupancy(state, requested_mode, certificate, tube),
                "minimum_constraint_margin": float(np.min(margins)),
                "active_constraint": names[int(np.argmin(margins))],
            }
            accept = True
            if policy in {"timer_only", "certified_ablation"}:
                if "timing" not in disabled_predicates and elapsed + 1.0e-12 < certificate["realized_dwell"]:
                    reasons.append("minimum residence time not met")
            elif policy == "hysteresis":
                improvement = economic_improvement(call, mode, requested_mode)
                candidate["economic_improvement_dollars"] = improvement
                if improvement < hysteresis_dollars:
                    reasons.append("economic hysteresis band not exceeded")
            elif policy in {"predictive_filter", "predictive_backup_filter"}:
                prediction = physical_prediction(
                    state, requested_mode, current_time, certificate, tube
                )
                candidate.update(prediction)
                if prediction["minimum_predicted_physical_margin"] < -1.0e-10:
                    reasons.append("finite-horizon physical prediction failed")
                if (
                    policy == "predictive_backup_filter"
                    and prediction["terminal_tube_occupancy"] > 1.0 + 1.0e-10
                ):
                    reasons.append("certified terminal set not reached")
            elif policy == "unrestricted":
                pass
            else:
                raise ValueError(policy)

            if policy == "certified_ablation":
                if "landing" not in disabled_predicates and candidate["tube_occupancy"] > 1.0 + 1.0e-10:
                    reasons.append("successor tube landing failed")
                if "constraints" not in disabled_predicates and candidate["minimum_constraint_margin"] < -1.0e-10:
                    reasons.append("successor constraint check failed")

            accept = not reasons
            if accept:
                mode = requested_mode
                last_switch_step = step
                accepted_times.append(current_time)
                decision = "proposal accepted"
            else:
                decision = "requested change rejected; active mode retained"

        events.append({
            "time": current_time,
            "proposal": call["proposal"],
            "previous_mode": previous_mode,
            "executed_mode": mode,
            "elapsed_since_last_switch": elapsed,
            "candidate": candidate,
            "decision": decision,
            "rejection_reasons": reasons,
        })
        live.append_sample(storage, current_time, state, mode, certificate, tube)
        state = live.integrate_interval(storage, state, mode, current_time, certificate, tube)

    label = policy if policy != "hysteresis" else f"hysteresis_{hysteresis_dollars:g}_dollars"
    return live.finalize_simulation(storage, events, label, accepted_times)


def adversarial_run(certificate: dict, tube: dict, certified: bool, horizon: float = 40.0) -> dict:
    """Greedy well-formed proposer maximising immediate post-transition V."""
    state = mfd.dominant_initial_state(certificate, tube)
    mode = 0
    last_switch_step = 0
    accepted_times: list[float] = []
    events: list[dict] = []
    storage = storage_template()
    live.append_sample(storage, 0.0, state, mode, certificate, tube)
    steps = int(round(horizon / mfd.PROPOSAL_INTERVAL))

    for step in range(steps):
        current_time = step * mfd.PROPOSAL_INTERVAL
        lookahead = []
        for candidate_mode in (0, 1):
            prediction = physical_prediction(state, candidate_mode, current_time)
            # This is the plan's stated adversary: score the inherited physical
            # state in each successor certificate immediately after transition.
            score = live.tube_occupancy(state, candidate_mode, certificate, tube)
            lookahead.append((score, candidate_mode, prediction))
        _, requested_mode, prediction = max(lookahead, key=lambda item: item[0])
        previous_mode = mode
        elapsed = (step - last_switch_step) * mfd.PROPOSAL_INTERVAL
        names, margins = mfd.margin_components(state, requested_mode)
        candidate_occupancy = live.tube_occupancy(state, requested_mode, certificate, tube)
        reasons: list[str] = []
        if certified and requested_mode != mode:
            if elapsed + 1.0e-12 < certificate["realized_dwell"]:
                reasons.append("minimum residence time not met")
            if candidate_occupancy > 1.0 + 1.0e-10:
                reasons.append("successor tube landing failed")
            if float(np.min(margins)) < -1.0e-10:
                reasons.append("successor constraint check failed")
        if requested_mode != mode and not reasons:
            mode = requested_mode
            last_switch_step = step
            accepted_times.append(current_time)
        decision = (
            "proposal accepted" if requested_mode != previous_mode and not reasons
            else "requested change rejected; active mode retained" if reasons
            else "requested mode already active"
        )
        events.append({
            "time": current_time,
            "proposal": {"action": f"select_mode_{requested_mode + 1}", "basis": "greedy adversarial look-ahead"},
            "adversarial_scores": {f"mode_{item[1] + 1}": item[0] for item in lookahead},
            "previous_mode": previous_mode,
            "executed_mode": mode,
            "elapsed_since_last_switch": elapsed,
            "candidate": {
                "mode": requested_mode,
                "tube_occupancy": candidate_occupancy,
                "minimum_constraint_margin": float(np.min(margins)),
                "minimum_predicted_physical_margin": prediction["minimum_predicted_physical_margin"],
            },
            "decision": decision,
            "rejection_reasons": reasons,
        })
        live.append_sample(storage, current_time, state, mode, certificate, tube)
        state = live.integrate_interval(storage, state, mode, current_time, certificate, tube)

    label = "adversarial_certified" if certified else "adversarial_unrestricted"
    return live.finalize_simulation(storage, events, label, accepted_times)


def live_unrestricted_run(client, certificate, tube, forecast_errors: np.ndarray) -> tuple[dict, list[dict]]:
    state = mfd.dominant_initial_state(certificate, tube)
    mode = 0
    accepted_times: list[float] = []
    events: list[dict] = []
    calls: list[dict] = []
    storage = storage_template()
    live.append_sample(storage, 0.0, state, mode, certificate, tube)
    for step, regional_errors in enumerate(forecast_errors):
        current_time = step * mfd.PROPOSAL_INTERVAL
        economics = live.economic_observation(current_time, state, regional_errors)
        prompt = live.proposal_prompt(current_time, state, mode, economics)
        call = client.propose(prompt, step)
        call.update({
            "decision_index": step,
            "decision_time": current_time,
            "prompt_version": live.PROMPT_VERSION,
            "observation": {"state": state.copy(), "current_mode": mode, "economics": economics},
        })
        requested_mode = live.action_mode(call["proposal"]["action"])
        previous_mode = mode
        if requested_mode is not None:
            mode = requested_mode
        if mode != previous_mode:
            accepted_times.append(current_time)
        names, margins = mfd.margin_components(state, mode)
        events.append({
            "time": current_time,
            "proposal": call["proposal"],
            "previous_mode": previous_mode,
            "executed_mode": mode,
            "candidate": {
                "tube_occupancy": live.tube_occupancy(state, mode, certificate, tube),
                "minimum_constraint_margin": float(np.min(margins)),
                "active_constraint": names[int(np.argmin(margins))],
            },
            "decision": "unrestricted proposal executed",
            "rejection_reasons": [],
        })
        calls.append(call)
        live.append_sample(storage, current_time, state, mode, certificate, tube)
        state = live.integrate_interval(storage, state, mode, current_time, certificate, tube)
    return live.finalize_simulation(storage, events, "genuinely_live_unrestricted", accepted_times), calls


def architecture_ablation_record() -> dict:
    architecture = json.loads(
        (RESULTS_DIR / "mfd_architecture_results.json").read_text(encoding="utf-8")
    )
    requests = architecture["simulation"]["adaptation_requests"]
    return {
        "timing_is_nonbinding": all(
            request["time"] - 1.0 >= 5.9 - 1.0e-12 for request in requests
        ),
        "landing_enabled": [
            {"time": item["time"], "accepted": item["accepted"], "robust_landing_ratio": item["robust_landing_ratio"]}
            for item in requests
        ],
        "landing_disabled": [
            {"time": item["time"], "accepted": True, "formal_successor_tube_violation": item["robust_landing_ratio"] > 1.0}
            for item in requests
        ],
        "interpretation": (
            "At 8 h, disabling the landing predicate admits an action whose robust "
            "successor-set ratio exceeds one. The resulting loss is the formal "
            "successor-tube guarantee."
        ),
    }


def plot_policy_comparison(results: dict[str, dict]) -> None:
    selected = [
        "unrestricted", "timer_only", "hysteresis_250", "predictive_filter",
        "predictive_backup_filter",
        "certified", "adversarial_certified", "adversarial_unrestricted",
    ]
    labels = [
        "Unrestricted", "Timer", "Hyst. $250", "Physical\nscreen",
        "Terminal\nfilter", "Certificate", "Greedy\ncert.", "Greedy\nunrest.",
    ]
    exits = [float(np.max(results[name]["occupancy"])) for name in selected]
    switches = [results[name]["executed_switches"] for name in selected]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    axes[0].bar(labels, exits, color="#35618f")
    axes[0].axhline(1.0, color="#b24745", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("maximum tube occupancy")
    axes[1].bar(labels, switches, color="#4a8d68")
    axes[1].set_ylabel("executed configuration changes")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=18, labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "mfd_runtime_policy_comparison.pdf", bbox_inches="tight")
    figure.savefig(RESULTS_DIR / "mfd_runtime_policy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_rejection_reasons(summary: dict) -> None:
    policies = ["timer_only", "certified"]
    reasons = sorted({
        reason for policy in policies
        for reason in summary[policy]["rejection_reason_counts"]
    })
    figure, axis = plt.subplots(figsize=(5.5, 3.0))
    bottom = np.zeros(len(policies))
    for reason in reasons:
        values = np.asarray([summary[policy]["rejection_reason_counts"].get(reason, 0) for policy in policies])
        axis.bar(policies, values, bottom=bottom, label=reason)
        bottom += values
    axis.set_ylabel("rejected change requests")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "mfd_rejection_reasons.pdf", bbox_inches="tight")
    figure.savefig(RESULTS_DIR / "mfd_rejection_reasons.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_transport_csv(summary: dict) -> None:
    order = [
        "unrestricted", "timer_only", "hysteresis_100", "hysteresis_250",
        "hysteresis_500", "predictive_filter", "predictive_backup_filter", "certified",
        "adversarial_certified", "adversarial_unrestricted",
    ]
    with (RESULTS_DIR / "mfd_transport_policy_table.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "policy", "horizon_h", "vehicle_hours_in_network", "completed_trips_vehicles",
            "cumulative_network_production_vehicles", "peak_total_accumulation_vehicles",
            "constraint_violation_time_h", "maximum_tube_occupancy", "configuration_changes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in order:
            item = summary[name]
            transport = item["transportation"]
            writer.writerow({
                "policy": name,
                "horizon_h": item["horizon_h"],
                "vehicle_hours_in_network": transport["vehicle_hours_in_network"],
                "completed_trips_vehicles": transport["completed_trips_vehicles"],
                "cumulative_network_production_vehicles": transport["cumulative_network_production_vehicles"],
                "peak_total_accumulation_vehicles": transport["peak_total_accumulation_vehicles"],
                "constraint_violation_time_h": item["first_constraint_violation_time_h"],
                "maximum_tube_occupancy": item["maximum_tube_occupancy"],
                "configuration_changes": item["executed_switches"],
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-live-unrestricted", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=471)
    parser.add_argument("--forecast-seed", type=int, default=20260812)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    certificate = mfd.independent_certificate()
    tube = mfd.tube_record(certificate)
    calls = load_checked_calls()
    checked_certified = replay_policy(calls, certificate, tube, "certified_ablation")
    results = {
        "unrestricted": replay_policy(calls, certificate, tube, "unrestricted"),
        "timer_only": replay_policy(calls, certificate, tube, "timer_only"),
        "hysteresis_0": replay_policy(calls, certificate, tube, "hysteresis", 0.0),
        "hysteresis_100": replay_policy(calls, certificate, tube, "hysteresis", 100.0),
        "hysteresis_250": replay_policy(calls, certificate, tube, "hysteresis", 250.0),
        "hysteresis_500": replay_policy(calls, certificate, tube, "hysteresis", 500.0),
        "predictive_filter": replay_policy(calls, certificate, tube, "predictive_filter"),
        "predictive_backup_filter": replay_policy(
            calls, certificate, tube, "predictive_backup_filter"
        ),
        "certified": checked_certified,
        "ablation_no_timing": replay_policy(
            calls, certificate, tube, "certified_ablation", disabled_predicates=frozenset({"timing"})
        ),
        "ablation_no_landing": replay_policy(
            calls, certificate, tube, "certified_ablation", disabled_predicates=frozenset({"landing"})
        ),
        "ablation_no_constraints": replay_policy(
            calls, certificate, tube, "certified_ablation", disabled_predicates=frozenset({"constraints"})
        ),
        "adversarial_certified": adversarial_run(certificate, tube, certified=True),
        "adversarial_unrestricted": adversarial_run(certificate, tube, certified=False),
    }

    live_calls = None
    if args.run_live_unrestricted:
        client = live.OllamaMFDProposer(
            args.endpoint, args.model, args.temperature, args.seed, args.timeout, live.PROPOSAL_SCHEMA
        )
        forecast = live.rolling_forecast_errors(
            int(round(live.LIVE_HORIZON / mfd.PROPOSAL_INTERVAL)), args.forecast_seed
        )
        live_result, live_calls = live_unrestricted_run(client, certificate, tube, forecast)
        results["live_unrestricted"] = live_result

    summaries = {name: summarize(result) for name, result in results.items()}
    output = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checked_trace": str(TRACE_PATH.relative_to(ROOT)),
        "predictive_filter_horizon_h": PREDICTIVE_HORIZON,
        "certificate": {
            "rate": certificate["rate"],
            "transition_factor": certificate["transition_factor"],
            "realized_dwell_h": certificate["realized_dwell"],
            "tube_level": tube["common_level"],
        },
        "policies": summaries,
        "partial_observation_ablation": architecture_ablation_record(),
        "domain_predicate_ablation": {
            "status": "inert for the installed same-domain fixed-gain and P-to-PI records",
            "reason": "all exercised predecessor information sets are declared inside their transition domains",
        },
        "live_unrestricted_call_count": None if live_calls is None else len(live_calls),
    }
    (RESULTS_DIR / "mfd_extended_runtime_results.json").write_text(
        json.dumps(live.json_ready(output), indent=2), encoding="utf-8"
    )
    if live_calls is not None:
        (LIVE_RESULTS_DIR / "live_llm_mfd_unrestricted_trace.json").write_text(
            json.dumps(live.json_ready({"calls": live_calls, "summary": summaries["live_unrestricted"]}), indent=2),
            encoding="utf-8",
        )
    write_transport_csv(summaries)
    plot_policy_comparison(results)
    plot_rejection_reasons(summaries)
    print(json.dumps(live.json_ready(output), indent=2))


if __name__ == "__main__":
    main()
