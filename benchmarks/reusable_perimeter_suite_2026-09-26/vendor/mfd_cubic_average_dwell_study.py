#!/usr/bin/env python3
"""Average-dwell token-budget study on the checked cubic-MFD benchmark.

The policy combines three independently auditable predicates:

1. successor landing and physical constraints;
2. the robust state-dependent envelope; and
3. a token bucket that enforces the classical average-dwell count.

The token bucket permits a declared finite burst while preserving a positive
long-run Lyapunov decay margin.  No controller, plant, tube, or theorem
constant is retuned in this study.
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

import mfd_cubic_nonstationary_study as study


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_average_dwell_results.json"
FIGURE_PDF = RESULTS_DIR / "mfd_cubic_average_dwell.pdf"
FIGURE_PNG = RESULTS_DIR / "mfd_cubic_average_dwell.png"

MODERATE_AMPLITUDE = 0.003
SEVERE_AMPLITUDE = 0.006
EPISODE_MINUTES = (30.0, 60.0, 90.0)
BURST_CAPACITIES = (1, 2, 3)


def json_ready(value: Any) -> Any:
    return study.json_ready(value)


def compact(result: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "policy",
        "horizon_completed_h",
        "robust_profile_feasible",
        "executed_changes",
        "rejected_changes",
        "no_op_reselections",
        "maximum_tube_occupancy",
        "final_tube_occupancy",
        "first_tube_exit_time_h",
        "minimum_state_constraint_margin_vehicles",
        "minimum_input_constraint_margin",
        "transportation",
        "physical_violation",
        "average_dwell",
    }
    record = {key: value for key, value in result.items() if key in keys}
    if "average_dwell" in record:
        record["average_dwell"] = {
            key: value
            for key, value in record["average_dwell"].items()
            if key != "token_trace"
        }
    return record


def safe(result: dict[str, Any]) -> bool:
    average = result.get("average_dwell", {})
    return bool(
        result["robust_profile_feasible"]
        and result["physical_violation"] is None
        and result["maximum_tube_occupancy"] <= 1.0 + 1.0e-9
        and result["first_tube_exit_time_h"] is None
        and average.get("count_bound_verified", False)
        and average.get("certified_decay_margin_per_hour", -1.0) > 0.0
    )


def run_case(
    episode_minutes: float,
    amplitude: float,
    average_minutes: float,
    burst_capacity: int,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    include_trajectory: bool = False,
) -> dict[str, Any]:
    priority = lambda time_h: study.block_priority(time_h, episode_minutes)
    delta = lambda time_h: study.block_delta(time_h, amplitude, episode_minutes)
    return study.simulate(
        "robust_adt_envelope",
        gains,
        certificate,
        tube,
        amplitude=amplitude,
        proposal_minutes=study.PRIMARY_PROPOSAL_MINUTES,
        delta_function=delta,
        priority_function=priority,
        include_trajectory=include_trajectory,
        adt_average_minutes=average_minutes,
        adt_burst_capacity=burst_capacity,
    )


def comparator_case(
    policy: str,
    episode_minutes: float,
    amplitude: float,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    include_trajectory: bool = False,
) -> dict[str, Any]:
    priority = lambda time_h: study.block_priority(time_h, episode_minutes)
    delta = lambda time_h: study.block_delta(time_h, amplitude, episode_minutes)
    return study.simulate(
        policy,
        gains,
        certificate,
        tube,
        amplitude=amplitude,
        proposal_minutes=study.PRIMARY_PROPOSAL_MINUTES,
        delta_function=delta,
        priority_function=priority,
        include_trajectory=include_trajectory,
    )


def make_figure(selected: dict[str, Any], comparisons: dict[str, Any]) -> None:
    trajectory = selected["trajectory"]
    time_h = np.asarray(trajectory["time_h"], dtype=float)
    mode = np.asarray(trajectory["mode"], dtype=int) + 1
    priority = np.asarray(trajectory["priority_mode"], dtype=int) + 1
    occupancy = np.asarray(trajectory["tube_occupancy"], dtype=float)
    tokens = selected["average_dwell"]["token_trace"]
    token_time = np.asarray([item["time_h"] for item in tokens], dtype=float)
    token_value = np.asarray([item["tokens_after_decision"] for item in tokens], dtype=float)

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 6.9))
    axes[0, 0].step(time_h, priority, where="post", color="#8a6bb1", linewidth=2.0, label="requested priority")
    axes[0, 0].step(time_h, mode, where="post", color="#245a85", linewidth=1.7, label="executed mode")
    axes[0, 0].set_yticks([1, 2])
    axes[0, 0].set_ylabel("Mode")
    axes[0, 0].set_title("Burst-capable executed schedule")
    axes[0, 0].legend(frameon=False, loc="upper right")

    axes[0, 1].step(token_time, token_value, where="post", color="#c2762b", linewidth=2.0)
    axes[0, 1].axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0, 1].set_ylabel("Available update tokens")
    axes[0, 1].set_title("Average-dwell token budget")

    axes[1, 0].plot(time_h, occupancy, color="#2d7c5b", linewidth=2.0)
    axes[1, 0].axhline(1.0, color="#9b2c2c", linestyle="--", linewidth=1.2, label="tube boundary")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("Tube occupancy")
    axes[1, 0].set_title("Certified state containment")
    axes[1, 0].legend(frameon=False)

    labels = ["30", "60", "90"]
    adt_gain = [comparisons[value]["adt_vs_dwell"]["priority_cost_improvement_percent"] for value in labels]
    envelope_gain = [comparisons[value]["envelope_vs_dwell"]["priority_cost_improvement_percent"] for value in labels]
    positions = np.arange(len(labels))
    axes[1, 1].bar(positions - 0.18, adt_gain, 0.36, color="#245a85", label="ADT + envelope")
    axes[1, 1].bar(positions + 0.18, envelope_gain, 0.36, color="#77a88d", label="envelope only")
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_xlabel("Priority episode (min)")
    axes[1, 1].set_ylabel("Priority-cost improvement vs robust dwell (%)")
    axes[1, 1].set_title("Operator-facing benefit")
    axes[1, 1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    gains, certificate, tube = study.load_installed()
    robust = study.robust_record(
        MODERATE_AMPLITUDE,
        study.PRIMARY_PROPOSAL_MINUTES,
        certificate,
        tube,
    )
    critical_minutes = 60.0 * math.log(
        float(certificate["transition_factor"])
    ) / float(robust["practical_flow_rate_per_hour"])
    selected_average_minutes = 15.0 * math.ceil(
        (critical_minutes + 1.0e-9) / 15.0
    )
    selected_burst_capacity = 2
    grid_average_minutes = tuple(
        sorted({selected_average_minutes, 120.0, 135.0})
    )

    grid: dict[str, Any] = {}
    for amplitude in (MODERATE_AMPLITUDE, SEVERE_AMPLITUDE):
        amplitude_record: dict[str, Any] = {}
        for episode in EPISODE_MINUTES:
            candidates: dict[str, Any] = {}
            for average in grid_average_minutes:
                for capacity in BURST_CAPACITIES:
                    key = f"tau_{average:.0f}_N0_{capacity}"
                    result = run_case(
                        episode,
                        amplitude,
                        average,
                        capacity,
                        gains,
                        certificate,
                        tube,
                    )
                    candidates[key] = {"safe": safe(result), "run": compact(result)}
            amplitude_record[str(int(episode))] = candidates
        grid[f"amplitude_{amplitude:.3f}"] = amplitude_record

    comparisons: dict[str, Any] = {}
    selected_full: dict[str, Any] | None = None
    for episode in EPISODE_MINUTES:
        adt = run_case(
            episode,
            MODERATE_AMPLITUDE,
            selected_average_minutes,
            selected_burst_capacity,
            gains,
            certificate,
            tube,
            include_trajectory=episode == 60.0,
        )
        dwell = comparator_case(
            "robust_dwell",
            episode,
            MODERATE_AMPLITUDE,
            gains,
            certificate,
            tube,
        )
        envelope = comparator_case(
            "robust_envelope",
            episode,
            MODERATE_AMPLITUDE,
            gains,
            certificate,
            tube,
        )
        if not safe(adt):
            raise RuntimeError(f"selected ADT profile failed at {episode:g} min")
        comparisons[str(int(episode))] = {
            "adt": compact(adt),
            "robust_dwell": compact(dwell),
            "robust_envelope": compact(envelope),
            "adt_vs_dwell": study.comparison(adt, dwell),
            "envelope_vs_dwell": study.comparison(envelope, dwell),
        }
        if episode == 60.0:
            selected_full = adt

    assert selected_full is not None
    output = {
        "study": "average-dwell token budget combined with the robust state envelope",
        "fixed_elements": (
            "literature-cubic plant, controller pair, Lyapunov metrics, SOS tube, "
            "demand profiles, proposal clock, landing checks, and constraints"
        ),
        "selection_rule": (
            "N0=2 declares a two-change burst; tau_a is the first 15-minute "
            "proposal-grid value strictly above log(nu_bar)/r"
        ),
        "robust_average_dwell_threshold_minutes": critical_minutes,
        "selected": {
            "average_dwell_minutes": selected_average_minutes,
            "burst_capacity": selected_burst_capacity,
        },
        "grid": grid,
        "moderate_comparisons": comparisons,
        "selected_60_min_trajectory": selected_full,
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(selected_full, comparisons)

    summary = {
        "robust_average_dwell_threshold_minutes": critical_minutes,
        "selected": output["selected"],
        "comparisons": {
            key: {
                "adt": value["adt"],
                "adt_vs_dwell": value["adt_vs_dwell"],
            }
            for key, value in comparisons.items()
        },
        "all_grid_points_safe": all(
            candidate["safe"]
            for amplitude in grid.values()
            for episode in amplitude.values()
            for candidate in episode.values()
        ),
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
