#!/usr/bin/env python3
"""Matched runtime trajectories inside the cubic-MFD SOS tube."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mfd_cubic_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
SOS_JSON = RESULTS_DIR / "mfd_cubic_sos_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_runtime_pilot_results.json"
FIGURE_PDF = RESULTS_DIR / "mfd_cubic_runtime_pilot.pdf"
FIGURE_PNG = RESULTS_DIR / "mfd_cubic_runtime_pilot.png"
ENVELOPE_RATES = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
SELECTED_ENVELOPE_RATE = 1.5


def state_on_tube(
    direction: np.ndarray,
    occupancy: float,
    certificate: dict[str, Any],
    tube: dict[str, Any],
    mode: int = 0,
) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    matrix = certificate["matrices"][mode]
    scale = math.sqrt(
        occupancy * tube["common_level"] / float(direction @ matrix @ direction)
    )
    return pilot.X_STAR + scale * direction


def structured_directions() -> dict[str, np.ndarray]:
    return {
        "region_1_heavy": np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25]),
        "region_2_heavy": np.array([-1.0, 1.0, -0.5, 0.5, -0.25, 0.25]),
        "network_heavy": np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25]),
        "opposed_destination_mix": np.array([0.0, 0.0, 1.0, -1.0, -0.5, 0.5]),
    }


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


def main() -> None:
    if not PILOT_JSON.exists() or not SOS_JSON.exists():
        raise SystemExit("Run the cubic model and SOS pilot scripts first")
    record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = tuple(np.asarray(value, dtype=float) for value in record["controller_search"]["selected_gains"])
    closed_loop = tuple(pilot.A + pilot.B @ gain for gain in gains)
    certificate = record["certificate"]
    certificate["matrices"] = [np.asarray(value, dtype=float) for value in certificate["matrices"]]
    tube = {
        "individual_levels": [float(item["level"]) for item in sos["sos_certificates"]],
        "common_level": float(sos["common_level"]),
        "certificate": "degree-six SOS at the physical constraint boundary",
    }
    policies = (
        "held_1",
        "held_2",
        "unrestricted_one_minute",
        "unrestricted_peak",
        "landing_only",
        "time_envelope",
        "dwell_certified",
    )
    witness_state = pilot.dominant_initial_state(gains, closed_loop, certificate, tube)
    envelope_sweep = {
        str(rate): pilot.simulate(
            "time_envelope",
            gains,
            closed_loop,
            certificate,
            tube,
            envelope_rate=rate,
            initial_state=witness_state,
        )
        for rate in ENVELOPE_RATES
    }
    eligible_rates = [
        rate
        for rate in ENVELOPE_RATES
        if envelope_sweep[str(rate)]["executed_changes"] >= 100
    ]
    selected_by_rule = min(
        eligible_rates,
        key=lambda rate: envelope_sweep[str(rate)]["transportation"][
            "excess_vehicle_hours_relative_to_target"
        ],
    )
    if selected_by_rule != SELECTED_ENVELOPE_RATE:
        raise RuntimeError(
            "The archived envelope-rate selection no longer follows its declared rule: "
            f"expected {SELECTED_ENVELOPE_RATE}, obtained {selected_by_rule}"
        )
    witness_runs = {
        policy: pilot.simulate(
            policy,
            gains,
            closed_loop,
            certificate,
            tube,
            envelope_rate=SELECTED_ENVELOPE_RATE,
            initial_state=witness_state,
            include_trajectory=True,
        )
        for policy in policies
    }

    operational_state = state_on_tube(
        structured_directions()["region_1_heavy"], 0.8, certificate, tube
    )
    operational_policies = (
        "held_1",
        "held_2",
        "unrestricted_one_minute",
        "landing_only",
        "time_envelope",
        "dwell_certified",
    )
    operational_runs = {
        policy: pilot.simulate(
            policy,
            gains,
            closed_loop,
            certificate,
            tube,
            envelope_rate=SELECTED_ENVELOPE_RATE,
            initial_state=operational_state,
            proposal_period_minutes=15.0,
            include_trajectory=True,
        )
        for policy in operational_policies
    }

    transfer = []
    for name, direction in structured_directions().items():
        for sign in (1.0, -1.0):
            scenario = name if sign > 0.0 else f"{name}_reversed"
            state = state_on_tube(sign * direction, 0.8, certificate, tube)
            runs = {
                policy: pilot.simulate(
                    policy,
                    gains,
                    closed_loop,
                    certificate,
                    tube,
                    envelope_rate=SELECTED_ENVELOPE_RATE,
                    initial_state=state,
                )
                for policy in (
                    "unrestricted_one_minute",
                    "landing_only",
                    "time_envelope",
                    "dwell_certified",
                )
            }
            transfer.append(
                {
                    "scenario": scenario,
                    "initial_error_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES * (state - pilot.X_STAR),
                    "initial_error_norm_vehicles": (
                        pilot.ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(state - pilot.X_STAR)
                    ),
                    "minimum_initial_margins": [
                        float(np.min(pilot.margin_components(state, mode, gains)[1]))
                        for mode in (0, 1)
                    ],
                    "runs": runs,
                }
            )

    output = {
        "scope": "matched one-minute proposal trajectories using the cubic-MFD SOS tube",
        "proposal_process": (
            "the same alternating mode request is supplied to every execution profile; "
            "it represents a fast proposer whose preferred region-priority objective flips each minute"
        ),
        "envelope_rate_per_hour": SELECTED_ENVELOPE_RATE,
        "envelope_rate_selection_rule": (
            "minimize excess vehicle-hours on the declared rate grid subject "
            "to at least 100 executed changes over six hours"
        ),
        "envelope_rate_sweep": envelope_sweep,
        "sos_common_level": tube["common_level"],
        "witness_initial_state": witness_state,
        "witness_initial_error_norm_vehicles": (
            pilot.ACCUMULATION_UNIT_VEHICLES * np.linalg.norm(witness_state - pilot.X_STAR)
        ),
        "witness_runs": witness_runs,
        "operational_priority_scenario": {
            "description": (
                "a 229.45-vehicle regional imbalance with the economically prioritized "
                "region alternating every 15 minutes under otherwise nominal MFD physics"
            ),
            "proposal_period_minutes": 15.0,
            "initial_state": operational_state,
            "initial_error_norm_vehicles": (
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.linalg.norm(operational_state - pilot.X_STAR)
            ),
            "runs": operational_runs,
        },
        "structured_transfer": transfer,
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")

    labels = {
        "unrestricted_one_minute": "Unrestricted, 1-min",
        "landing_only": "Landing only",
        "time_envelope": "State envelope",
        "dwell_certified": "Fixed dwell",
    }
    colors = {
        "unrestricted_one_minute": "#b24745",
        "landing_only": "#d28b3c",
        "time_envelope": "#35618f",
        "dwell_certified": "#4a8d68",
    }
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    selected = tuple(labels)
    for key in selected:
        trajectory = witness_runs[key]["trajectory"]
        axes[0].plot(
            trajectory["time_h"],
            trajectory["tube_occupancy"],
            label=labels[key],
            color=colors[key],
            linewidth=1.5,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("time (h)")
    axes[0].set_ylabel("tube occupancy")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(
        [labels[key] for key in selected],
        [witness_runs[key]["executed_changes"] for key in selected],
        color=[colors[key] for key in selected],
    )
    axes[1].set_ylabel("executed controller changes")
    for axis in axes[1:]:
        axis.tick_params(axis="x", rotation=18, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "sos_common_level": tube["common_level"],
        "witness_initial_error_norm_vehicles": output["witness_initial_error_norm_vehicles"],
        # Keep trajectories in the archived JSON, but avoid flooding a normal
        # terminal run with thousands of time samples.
        "witness_runs": {
            key: {field: value for field, value in run.items() if field != "trajectory"}
            for key, run in witness_runs.items()
        },
        "operational_priority_runs": {
            key: {field: value for field, value in run.items() if field != "trajectory"}
            for key, run in operational_runs.items()
        },
        "structured_transfer": [
            {
                "scenario": item["scenario"],
                "initial_error_norm_vehicles": item["initial_error_norm_vehicles"],
                "maximum_occupancies": {
                    key: value["maximum_tube_occupancy"] for key, value in item["runs"].items()
                },
            }
            for item in transfer
        ],
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
