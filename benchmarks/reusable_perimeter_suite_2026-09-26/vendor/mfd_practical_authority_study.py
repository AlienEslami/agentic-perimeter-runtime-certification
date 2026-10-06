#!/usr/bin/env python3
"""Practical-authority study for the application-centered MFD manuscript.

This study tests two focused improvements without changing the plant model:

1. install the already verified common SOS tube instead of the conservative
   coefficient-sum tube used by the baseline experiment; and
2. replace the uniform residence timer by state-dependent certified profiles.

The landing profile preserves the installed tube at every accepted transition.
The event-contraction profile additionally requires the post-transition
Lyapunov value to contract by a fixed factor relative to the previous accepted
transition.  On the six-minute proposal grid, this gives a non-Zeno sequence;
if infinitely many changes execute, their event values converge geometrically,
and if changes terminate, held-mode flow convergence applies.

The script also searches for an interpretable initial regional-imbalance state
inside the SOS tube and evaluates adaptive economic proposals from that state.
It writes a separate result artifact so that the publication baseline remains
unchanged until the new record has been reviewed.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import live_llm_mfd_validation as live
import mfd_extended_runtime_validation as extended
import mfd_fixed_gain_validation as mfd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
SYNTHESIS_PATH = RESULTS_DIR / "mfd_certificate_synthesis_results.json"
OUTPUT_PATH = RESULTS_DIR / "mfd_practical_authority_results.json"
FIGURE_PATH = RESULTS_DIR / "mfd_practical_authority_comparison.pdf"
FIGURE_PNG_PATH = RESULTS_DIR / "mfd_practical_authority_comparison.png"
TRAJECTORY_FIGURE_PATH = RESULTS_DIR / "mfd_practical_authority_trajectories.pdf"
TRAJECTORY_FIGURE_PNG_PATH = RESULTS_DIR / "mfd_practical_authority_trajectories.png"
HYSTERESIS_FIGURE_PATH = RESULTS_DIR / "mfd_hysteresis_sensitivity.pdf"
HYSTERESIS_FIGURE_PNG_PATH = RESULTS_DIR / "mfd_hysteresis_sensitivity.png"


def sos_tube_record(certificate: dict) -> dict:
    """Return the common fixed-degree SOS tube already audited in synthesis."""
    synthesis = json.loads(SYNTHESIS_PATH.read_text(encoding="utf-8"))
    records = synthesis["sos_certificates"]
    if not all(record["feasible"] for record in records):
        raise RuntimeError("the checked SOS records are not all feasible")
    tube = mfd.tube_record(certificate)
    tube.update({
        "installed_certificate": "checked fixed-degree SOS certificate",
        "individual_levels": [float(record["level"]) for record in records],
        "common_level": float(min(record["level"] for record in records)),
        "solver": synthesis["solver"],
        "maximum_coefficient_residual": float(max(
            record["maximum_coefficient_residual"] for record in records
        )),
        "minimum_gram_eigenvalue": float(min(
            record["minimum_gram_eigenvalue"] for record in records
        )),
    })
    return tube


def initial_state_on_tube(
    direction: np.ndarray,
    occupancy: float,
    certificate: dict,
    tube: dict,
    mode: int = 0,
) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    matrix = certificate["matrices"][mode]
    scale = math.sqrt(
        occupancy * tube["common_level"] / float(direction @ matrix @ direction)
    )
    return mfd.X_STAR + scale * direction


def policy_simulation(
    proposal_source,
    initial_state: np.ndarray,
    certificate: dict,
    tube: dict,
    policy: str,
    horizon: float = live.LIVE_HORIZON,
    event_contraction: float = 0.99,
    envelope_rate: float = 0.01,
    hysteresis_dollars: float = 0.0,
) -> dict:
    """Simulate a static trace or adaptive proposal callback.

    ``proposal_source`` is either a checked list of calls or a callback with
    signature ``(step, time, state, mode) -> (requested_mode, metadata)``.
    """
    state = np.asarray(initial_state, dtype=float).copy()
    mode = 0
    last_switch_step = 0
    last_event_occupancy = live.tube_occupancy(
        state, mode, certificate, tube
    )
    initial_occupancy = last_event_occupancy
    accepted_times: list[float] = []
    events: list[dict] = []
    storage = extended.storage_template()
    live.append_sample(storage, 0.0, state, mode, certificate, tube)
    steps = int(round(horizon / mfd.PROPOSAL_INTERVAL))

    for step in range(steps):
        current_time = step * mfd.PROPOSAL_INTERVAL
        if callable(proposal_source):
            requested_mode, proposal_metadata = proposal_source(
                step, current_time, state, mode
            )
        else:
            call = proposal_source[step]
            requested_mode = live.action_mode(call["proposal"]["action"])
            proposal_metadata = {
                "proposal": call["proposal"],
                "source": "checked live LLM trace",
            }

        previous_mode = mode
        elapsed = (step - last_switch_step) * mfd.PROPOSAL_INTERVAL
        candidate = None
        reasons: list[str] = []
        decision = "requested mode already active"

        if requested_mode is not None and requested_mode != mode:
            names, margins = mfd.margin_components(state, requested_mode)
            candidate_occupancy = live.tube_occupancy(
                state, requested_mode, certificate, tube
            )
            candidate = {
                "mode": int(requested_mode),
                "tube_occupancy": candidate_occupancy,
                "minimum_constraint_margin": float(np.min(margins)),
                "active_constraint": names[int(np.argmin(margins))],
            }

            if policy == "unrestricted":
                pass
            elif policy == "fixed_dwell":
                if elapsed + 1.0e-12 < certificate["realized_dwell"]:
                    reasons.append("minimum residence time not met")
                if candidate_occupancy > 1.0 + 1.0e-10:
                    reasons.append("successor tube landing failed")
                if candidate["minimum_constraint_margin"] < -1.0e-10:
                    reasons.append("successor constraint check failed")
            elif policy == "landing":
                if candidate_occupancy > 1.0 + 1.0e-10:
                    reasons.append("successor tube landing failed")
                if candidate["minimum_constraint_margin"] < -1.0e-10:
                    reasons.append("successor constraint check failed")
            elif policy == "event_contraction":
                threshold = event_contraction * last_event_occupancy
                candidate["event_contraction_threshold"] = threshold
                if candidate_occupancy > threshold + 1.0e-10:
                    reasons.append("event-contraction budget not met")
                if candidate_occupancy > 1.0 + 1.0e-10:
                    reasons.append("successor tube landing failed")
                if candidate["minimum_constraint_margin"] < -1.0e-10:
                    reasons.append("successor constraint check failed")
            elif policy == "time_envelope":
                if not 0.0 < envelope_rate <= certificate["rate"]:
                    raise ValueError("envelope rate must lie in (0, flow rate]")
                threshold = initial_occupancy * math.exp(-envelope_rate * current_time)
                candidate["time_envelope_threshold"] = threshold
                if candidate_occupancy > threshold + 1.0e-10:
                    reasons.append("time-contraction envelope not met")
                if candidate_occupancy > 1.0 + 1.0e-10:
                    reasons.append("successor tube landing failed")
                if candidate["minimum_constraint_margin"] < -1.0e-10:
                    reasons.append("successor constraint check failed")
            elif policy == "hysteresis":
                improvement = float(proposal_metadata["economic_improvement_dollars"])
                candidate["economic_improvement_dollars"] = improvement
                if improvement < hysteresis_dollars:
                    reasons.append("economic hysteresis band not exceeded")
            else:
                raise ValueError(policy)

            if reasons:
                decision = "requested change rejected; active mode retained"
            else:
                mode = requested_mode
                last_switch_step = step
                last_event_occupancy = candidate_occupancy
                accepted_times.append(current_time)
                decision = "proposal accepted"

        events.append({
            "time": current_time,
            "previous_mode": previous_mode,
            "executed_mode": mode,
            "elapsed_since_last_switch": elapsed,
            "last_event_occupancy": last_event_occupancy,
            "candidate": candidate,
            "decision": decision,
            "rejection_reasons": reasons,
            **proposal_metadata,
        })
        live.append_sample(storage, current_time, state, mode, certificate, tube)
        state = live.integrate_interval(
            storage, state, mode, current_time, certificate, tube
        )

    return live.finalize_simulation(
        storage, events, policy, accepted_times
    )


def trajectory_operating_cost(simulation: dict) -> float:
    states = simulation["state"]
    commands = simulation["command"]
    times = simulation["time"]
    completion = np.asarray([
        mfd.transportation_rates(state, command)["completed_trips"]
        for state, command in zip(states, commands)
    ])
    integrand = (
        2000.0 * (states[:, 0] + states[:, 1])
        + 4000.0 * np.abs(states[:, 0] - states[:, 1])
        + 20000.0 * np.sum((commands - mfd.U_STAR) ** 2, axis=1)
        - 5000.0 * completion
    )
    return float(np.trapezoid(integrand, times))


def result_summary(simulation: dict) -> dict:
    summary = extended.summarize(simulation)
    summary["operating_cost_proxy_dollars"] = trajectory_operating_cost(simulation)
    return summary


def time_envelope_audit(simulation: dict, rate: float) -> dict:
    envelope = simulation["occupancy"][0] * np.exp(-rate * simulation["time"])
    ratios = simulation["occupancy"] / envelope
    return {
        "rate": rate,
        "maximum_realized_to_envelope_ratio": float(np.max(ratios)),
        "final_realized_to_envelope_ratio": float(ratios[-1]),
    }


def economic_proposer(regional_errors: np.ndarray | None = None):
    errors = np.zeros(2) if regional_errors is None else np.asarray(regional_errors)

    def propose(step: int, current_time: float, state: np.ndarray, mode: int):
        economics = live.economic_observation(current_time, state, errors)
        requested_mode = int(economics["lower_cost_mode"])
        costs = [
            float(item["net_operating_cost_dollars"])
            for item in economics["controller_forecasts"]
        ]
        return requested_mode, {
            "proposal": {
                "action": f"select_mode_{requested_mode + 1}",
                "basis": "lower_predicted_operating_cost",
            },
            "source": "deterministic lower-cost oracle",
            "cost_difference_mode_2_minus_mode_1_dollars": float(
                economics["cost_difference_mode_2_minus_mode_1_dollars"]
            ),
            "economic_improvement_dollars": float(costs[mode] - costs[requested_mode]),
        }

    return propose


def structured_directions() -> dict[str, np.ndarray]:
    """Physically interpretable perturbations of regional accumulation."""
    return {
        "region_1_heavy": np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]),
        "region_2_heavy": np.array([-1.0, 1.0, -0.5, 0.5, -0.25, 0.25]),
        "network_heavy": np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25]),
        "opposed_destination_mix": np.array([0.0, 0.0, 1.0, -1.0, -0.5, 0.5]),
    }


def select_operational_initial_state(certificate: dict, tube: dict) -> dict:
    """Choose the structured state with the largest fixed-mode cost spread."""
    candidates = []
    for name, direction in structured_directions().items():
        for sign in (1.0, -1.0):
            state = initial_state_on_tube(
                sign * direction, 0.80, certificate, tube
            )
            margin_by_mode = [
                float(np.min(mfd.margin_components(state, mode)[1]))
                for mode in (0, 1)
            ]
            if min(margin_by_mode) < 0.0:
                continue
            fixed = {}
            for mode in (0, 1):
                fixed[mode] = policy_simulation(
                    lambda _step, _time, _state, _active, selected=mode: (
                        selected,
                        {
                            "proposal": {"action": f"select_mode_{selected + 1}"},
                            "source": "fixed-mode comparison",
                        },
                    ),
                    state,
                    certificate,
                    tube,
                    "unrestricted",
                )
            costs = [trajectory_operating_cost(fixed[mode]) for mode in (0, 1)]
            candidates.append({
                "name": name if sign > 0 else f"{name}_reversed",
                "state": state,
                "initial_error_vehicles": 100.0 * (state - mfd.X_STAR),
                "initial_error_norm_vehicles": 100.0 * float(np.linalg.norm(state - mfd.X_STAR)),
                "initial_mode_occupancies": [
                    live.tube_occupancy(state, mode, certificate, tube)
                    for mode in (0, 1)
                ],
                "minimum_initial_margins": margin_by_mode,
                "fixed_mode_costs": costs,
                "fixed_mode_cost_spread": abs(costs[1] - costs[0]),
                "preferred_fixed_mode": int(np.argmin(costs)),
            })
    if not candidates:
        raise RuntimeError("no feasible structured initial states")
    return max(candidates, key=lambda record: record["fixed_mode_cost_spread"])


def plot_comparison(records: dict[str, dict]) -> None:
    order = ["unrestricted", "landing", "time_envelope_0.2", "hysteresis_17", "fixed_dwell"]
    labels = ["Unrestricted", "Landing", "Time envelope", "Hyst. $17", "Fixed dwell"]
    colors = ["#b24745", "#d28b3c", "#35618f", "#7d68a6", "#4a8d68"]
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    axes[0].bar(labels, [records[key]["maximum_tube_occupancy"] for key in order], color=colors)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("maximum tube occupancy")
    axes[1].bar(labels, [records[key]["executed_switches"] for key in order], color=colors)
    axes[1].set_ylabel("executed changes")
    baseline = records["fixed_dwell"]["operating_cost_proxy_dollars"]
    axes[2].bar(
        labels,
        [records[key]["operating_cost_proxy_dollars"] - baseline for key in order],
        color=colors,
    )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("cost difference from fixed dwell ($)")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=18, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")
    figure.savefig(FIGURE_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(simulations: dict[str, dict]) -> None:
    selected = ["unrestricted", "landing", "time_envelope_0.2", "hysteresis_17", "fixed_dwell"]
    labels = ["Unrestricted", "Landing", "Time envelope", "Hysteresis $17", "Fixed dwell"]
    colors = ["#b24745", "#d28b3c", "#35618f", "#7d68a6", "#4a8d68"]
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 5.5), sharex=True)
    for key, label, color in zip(selected, labels, colors):
        simulation = simulations[key]
        axes[0].plot(simulation["time"], simulation["occupancy"], label=label, color=color)
        axes[1].plot(
            simulation["time"],
            100.0 * simulation["error_norm"],
            label=label,
            color=color,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("tube occupancy")
    axes[1].set_ylabel("tracking-error norm (vehicles)")
    axes[1].set_xlabel("time (h)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(TRAJECTORY_FIGURE_PATH, bbox_inches="tight")
    figure.savefig(TRAJECTORY_FIGURE_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_hysteresis_sensitivity(sweep: dict[str, dict], transfer: list[dict]) -> None:
    thresholds = np.asarray([float(value) for value in sweep])
    changes = np.asarray([sweep[value]["executed_switches"] for value in sweep])
    occupancies = np.asarray([sweep[value]["maximum_tube_occupancy"] for value in sweep])
    label_map = {
        "region_1_heavy": "R1+",
        "region_1_heavy_reversed": "R1-",
        "region_2_heavy": "R2+",
        "region_2_heavy_reversed": "R2-",
        "network_heavy": "N+",
        "network_heavy_reversed": "N-",
        "opposed_destination_mix": "Mix+",
        "opposed_destination_mix_reversed": "Mix-",
    }
    names = [label_map[record["name"]] for record in transfer]
    hysteresis_occupancy = [record["hysteresis_17"]["maximum_tube_occupancy"] for record in transfer]
    online_occupancy = [record["time_envelope_0.2"]["maximum_tube_occupancy"] for record in transfer]

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    twin = axes[0].twinx()
    axes[0].plot(thresholds, changes, marker="o", color="#35618f", label="changes")
    twin.plot(thresholds, occupancies, marker="s", color="#b24745", label="occupancy")
    twin.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    twin.set_yscale("log")
    axes[0].set_xlabel("hysteresis band ($)")
    axes[0].set_ylabel("executed changes", color="#35618f")
    twin.set_ylabel("maximum tube occupancy", color="#b24745")
    axes[0].grid(alpha=0.25)

    locations = np.arange(len(names))
    width = 0.38
    axes[1].bar(locations - width / 2.0, hysteresis_occupancy, width, label="Hysteresis $17", color="#7d68a6")
    axes[1].bar(locations + width / 2.0, online_occupancy, width, label="Time envelope", color="#35618f")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xticks(locations, names, fontsize=8)
    axes[1].set_ylabel("maximum tube occupancy")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(HYSTERESIS_FIGURE_PATH, bbox_inches="tight")
    figure.savefig(HYSTERESIS_FIGURE_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    certificate = mfd.independent_certificate()
    analytic_tube = mfd.tube_record(certificate)
    sos_tube = sos_tube_record(certificate)
    checked_calls = extended.load_checked_calls()

    baseline_state = mfd.dominant_initial_state(certificate, sos_tube)
    contraction_factors = (0.999, 0.99, 0.95, 0.90)
    trace_runs = {
        "unrestricted": policy_simulation(
            checked_calls, baseline_state, certificate, sos_tube, "unrestricted"
        ),
        "landing": policy_simulation(
            checked_calls, baseline_state, certificate, sos_tube, "landing"
        ),
        "fixed_dwell": policy_simulation(
            checked_calls, baseline_state, certificate, sos_tube, "fixed_dwell"
        ),
    }
    for factor in contraction_factors:
        trace_runs[f"event_{factor:g}"] = policy_simulation(
            checked_calls,
            baseline_state,
            certificate,
            sos_tube,
            "event_contraction",
            event_contraction=factor,
        )
    trace_runs["time_envelope_0.2"] = policy_simulation(
        checked_calls,
        baseline_state,
        certificate,
        sos_tube,
        "time_envelope",
        envelope_rate=0.2,
    )

    operational_direction = structured_directions()["region_1_heavy"]
    operational_state = initial_state_on_tube(
        operational_direction, 0.80, certificate, sos_tube
    )
    selected_initial = {
        "name": "region_1_heavy",
        "description": (
            "symmetric regional imbalance: region 1 starts 24.03 vehicles above "
            "and region 2 starts 24.03 vehicles below their nominal accumulations"
        ),
        "state": operational_state,
        "initial_error_vehicles": 100.0 * (operational_state - mfd.X_STAR),
        "initial_error_norm_vehicles": 100.0 * float(
            np.linalg.norm(operational_state - mfd.X_STAR)
        ),
        "initial_mode_occupancies": [
            live.tube_occupancy(operational_state, mode, certificate, sos_tube)
            for mode in (0, 1)
        ],
        "minimum_initial_margins": [
            float(np.min(mfd.margin_components(operational_state, mode)[1]))
            for mode in (0, 1)
        ],
    }
    proposer = economic_proposer()
    operational_runs = {
        "unrestricted": policy_simulation(
            proposer, operational_state, certificate, sos_tube, "unrestricted", horizon=24.0
        ),
        "landing": policy_simulation(
            proposer, operational_state, certificate, sos_tube, "landing", horizon=24.0
        ),
        "fixed_dwell": policy_simulation(
            proposer, operational_state, certificate, sos_tube, "fixed_dwell", horizon=24.0
        ),
        "event_0.999": policy_simulation(
            proposer,
            operational_state,
            certificate,
            sos_tube,
            "event_contraction",
            horizon=24.0,
            event_contraction=0.999,
        ),
        "time_envelope_0.2": policy_simulation(
            proposer,
            operational_state,
            certificate,
            sos_tube,
            "time_envelope",
            horizon=24.0,
            envelope_rate=0.2,
        ),
        "hysteresis_17": policy_simulation(
            proposer,
            operational_state,
            certificate,
            sos_tube,
            "hysteresis",
            horizon=24.0,
            hysteresis_dollars=17.0,
        ),
        "hysteresis_100": policy_simulation(
            proposer,
            operational_state,
            certificate,
            sos_tube,
            "hysteresis",
            horizon=24.0,
            hysteresis_dollars=100.0,
        ),
        "hysteresis_250": policy_simulation(
            proposer,
            operational_state,
            certificate,
            sos_tube,
            "hysteresis",
            horizon=24.0,
            hysteresis_dollars=250.0,
        ),
    }

    trace_summaries = {
        name: result_summary(simulation) for name, simulation in trace_runs.items()
    }
    operational_summaries = {
        name: result_summary(simulation) for name, simulation in operational_runs.items()
    }

    hysteresis_thresholds = list(range(0, 21)) + [50, 100]
    hysteresis_sweep = {}
    for threshold in hysteresis_thresholds:
        simulation = policy_simulation(
            proposer,
            operational_state,
            certificate,
            sos_tube,
            "hysteresis",
            horizon=24.0,
            hysteresis_dollars=float(threshold),
        )
        hysteresis_sweep[str(threshold)] = result_summary(simulation)

    transfer_grid = []
    for name, direction in structured_directions().items():
        for sign in (1.0, -1.0):
            scenario_name = name if sign > 0.0 else f"{name}_reversed"
            state = initial_state_on_tube(sign * direction, 0.80, certificate, sos_tube)
            hysteresis_simulation = policy_simulation(
                proposer,
                state,
                certificate,
                sos_tube,
                "hysteresis",
                horizon=24.0,
                hysteresis_dollars=17.0,
            )
            online_simulation = policy_simulation(
                proposer,
                state,
                certificate,
                sos_tube,
                "time_envelope",
                horizon=24.0,
                envelope_rate=0.2,
            )
            transfer_grid.append({
                "name": scenario_name,
                "initial_error_norm_vehicles": 100.0 * float(np.linalg.norm(state - mfd.X_STAR)),
                "hysteresis_17": result_summary(hysteresis_simulation),
                "time_envelope_0.2": result_summary(online_simulation),
            })

    assert operational_summaries["unrestricted"]["first_constraint_violation_time_h"] == 22.1
    assert operational_summaries["landing"]["first_tube_exit_time_h"] is None
    assert operational_summaries["time_envelope_0.2"]["first_tube_exit_time_h"] is None
    assert operational_summaries["time_envelope_0.2"]["executed_switches"] > operational_summaries["fixed_dwell"]["executed_switches"]
    assert hysteresis_sweep["16"]["first_tube_exit_time_h"] is not None
    assert hysteresis_sweep["17"]["first_tube_exit_time_h"] is None
    assert hysteresis_sweep["18"]["executed_switches"] == 0
    assert all(record["time_envelope_0.2"]["first_tube_exit_time_h"] is None for record in transfer_grid)
    assert any(record["hysteresis_17"]["first_constraint_violation_time_h"] is not None for record in transfer_grid)
    trace_envelope_audit = time_envelope_audit(trace_runs["time_envelope_0.2"], 0.2)
    operational_envelope_audit = time_envelope_audit(
        operational_runs["time_envelope_0.2"], 0.2
    )
    assert trace_envelope_audit["maximum_realized_to_envelope_ratio"] <= 1.0 + 1.0e-9
    assert operational_envelope_audit["maximum_realized_to_envelope_ratio"] <= 1.0 + 1.0e-9
    output = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "state-dependent execution and SOS-tube practicality audit",
        "certificate": {
            "flow_rate": certificate["rate"],
            "uniform_dwell_h": certificate["realized_dwell"],
            "transition_factor": certificate["transition_factor"],
            "analytic_common_level": analytic_tube["common_level"],
            "sos_common_level": sos_tube["common_level"],
            "sos_to_analytic_level_ratio": (
                sos_tube["common_level"] / analytic_tube["common_level"]
            ),
            "sos_solver": sos_tube["solver"],
            "sos_maximum_coefficient_residual": sos_tube["maximum_coefficient_residual"],
            "sos_minimum_gram_eigenvalue": sos_tube["minimum_gram_eigenvalue"],
        },
        "checked_llm_trace_on_sos_tube": trace_summaries,
        "operational_initial_state": selected_initial,
        "adaptive_economic_scenario": operational_summaries,
        "time_envelope_audits": {
            "checked_llm_trace": trace_envelope_audit,
            "adaptive_economic_scenario": operational_envelope_audit,
        },
        "hysteresis_threshold_sweep": hysteresis_sweep,
        "structured_initial_condition_transfer": transfer_grid,
        "interpretation_guardrail": (
            "The adaptive economic case is a constructed nonlinear MFD experiment "
            "designed to isolate execution-authority effects. Policy comparisons "
            "are retained in full."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(live.json_ready(output), indent=2), encoding="utf-8"
    )
    plot_comparison(operational_summaries)
    plot_trajectories(operational_runs)
    plot_hysteresis_sensitivity(hysteresis_sweep, transfer_grid)
    print(json.dumps(live.json_ready(output), indent=2))


if __name__ == "__main__":
    main()
