#!/usr/bin/env python3
"""Persistent regional-priority study for the cubic-MFD runtime filter.

The study keeps the nominal MFD physics, installed controllers, certificate,
initial condition, and 15-minute decision clock fixed.  Only the duration of
the transportation operator's regional service priority changes.  This makes
the comparison identify the operational consequence of execution authority:
the state-envelope rule may follow a changed priority immediately when its
landing and decay tests pass, whereas the fixed-dwell rule must wait.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mfd_cubic_pilot as pilot
import mfd_cubic_runtime_pilot as runtime


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
SOS_JSON = RESULTS_DIR / "mfd_cubic_sos_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_priority_persistence_results.json"
FIGURE_PDF = RESULTS_DIR / "mfd_cubic_priority_persistence.pdf"
FIGURE_PNG = RESULTS_DIR / "mfd_cubic_priority_persistence.png"
DECISION_PERIOD_MINUTES = 15.0


def repeated_blocks(blocks: int) -> list[int]:
    """Six-hour alternating priority schedule with ``blocks`` 15-min blocks."""
    horizon_blocks = int(round(6.0 * 60.0 / DECISION_PERIOD_MINUTES))
    return [1 if (index // blocks) % 2 == 0 else 0 for index in range(horizon_blocks)]


PROFILES: dict[str, dict[str, Any]] = {
    "30_min_episodes": {
        "label": "30-min episodes",
        "description": "regional service priority alternates every 30 minutes",
        "schedule": repeated_blocks(2),
    },
    "60_min_episodes": {
        "label": "60-min episodes",
        "description": "regional service priority alternates every 60 minutes",
        "schedule": repeated_blocks(4),
    },
    "90_min_episodes": {
        "label": "90-min episodes",
        "description": "regional service priority alternates every 90 minutes",
        "schedule": repeated_blocks(6),
    },
    "incident_recovery": {
        "label": "Incident/recovery",
        "description": (
            "an irregular incident-management sequence with regional priorities "
            "lasting 75, 105, 90, and 90 minutes"
        ),
        "schedule": [1] * 5 + [0] * 7 + [1] * 6 + [0] * 6,
    },
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


def percent_improvement(candidate: float, comparator: float) -> float:
    return 100.0 * (comparator - candidate) / comparator


def percent_difference(candidate: float, comparator: float) -> float:
    return 100.0 * (candidate - comparator) / comparator


def main() -> None:
    record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = tuple(
        np.asarray(value, dtype=float)
        for value in record["controller_search"]["selected_gains"]
    )
    closed_loop = tuple(pilot.A + pilot.B @ gain for gain in gains)
    certificate = record["certificate"]
    certificate["matrices"] = [
        np.asarray(value, dtype=float) for value in certificate["matrices"]
    ]
    tube = {
        "individual_levels": [
            float(item["level"]) for item in sos["sos_certificates"]
        ],
        "common_level": float(sos["common_level"]),
        "certificate": "residual-validated degree-six SOS certificate",
    }
    initial_state = runtime.state_on_tube(
        runtime.structured_directions()["region_1_heavy"],
        0.8,
        certificate,
        tube,
    )
    policies = (
        "unrestricted_one_minute",
        "time_envelope",
        "dwell_certified",
        "held_1",
        "held_2",
    )
    profiles: dict[str, Any] = {}
    for name, specification in PROFILES.items():
        schedule = specification["schedule"]
        runs = {
            policy: pilot.simulate(
                policy,
                gains,
                closed_loop,
                certificate,
                tube,
                envelope_rate=runtime.SELECTED_ENVELOPE_RATE,
                initial_state=initial_state,
                proposal_period_minutes=DECISION_PERIOD_MINUTES,
                proposal_schedule=schedule,
                include_trajectory=True,
            )
            for policy in policies
        }
        envelope = runs["time_envelope"]
        dwell = runs["dwell_certified"]
        env_transport = envelope["transportation"]
        dwell_transport = dwell["transportation"]
        profiles[name] = {
            "label": specification["label"],
            "description": specification["description"],
            "priority_schedule": schedule,
            "desired_priority_changes": int(
                sum(left != right for left, right in zip(schedule, schedule[1:]))
            ),
            "runs": runs,
            "envelope_vs_dwell": {
                "priority_cost_improvement_percent": percent_improvement(
                    env_transport["time_varying_priority_tracking_cost"],
                    dwell_transport["time_varying_priority_tracking_cost"],
                ),
                "regional_imbalance_improvement_percent": percent_improvement(
                    env_transport["regional_imbalance_vehicle_hours"],
                    dwell_transport["regional_imbalance_vehicle_hours"],
                ),
                "network_vehicle_hours_difference_percent": percent_difference(
                    env_transport["vehicle_hours_in_network"],
                    dwell_transport["vehicle_hours_in_network"],
                ),
                "completed_trips_difference_percent": percent_difference(
                    env_transport["completed_trips_vehicles"],
                    dwell_transport["completed_trips_vehicles"],
                ),
                "priority_match_improvement_percentage_points": 100.0
                * (
                    env_transport["fraction_time_executed_mode_matches_priority"]
                    - dwell_transport["fraction_time_executed_mode_matches_priority"]
                ),
            },
        }

    output = {
        "scope": "persistent regional transportation-priority profiles",
        "fixed_elements": (
            "nominal cubic-MFD physics, installed controller pair, SOS tube, "
            "initial condition, and 15-minute decision clock"
        ),
        "interpretation": (
            "priority cost and regional imbalance measure service responsiveness; "
            "network vehicle-hours and completed trips separately measure aggregate "
            "traffic efficiency"
        ),
        "decision_period_minutes": DECISION_PERIOD_MINUTES,
        "initial_tube_occupancy": 0.8,
        "initial_error_norm_vehicles": float(
            pilot.ACCUMULATION_UNIT_VEHICLES
            * np.linalg.norm(initial_state - pilot.X_STAR)
        ),
        "profiles": profiles,
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
    )

    names = list(PROFILES)
    labels = [PROFILES[name]["label"] for name in names]
    priority = [
        profiles[name]["envelope_vs_dwell"]["priority_cost_improvement_percent"]
        for name in names
    ]
    imbalance = [
        profiles[name]["envelope_vs_dwell"]["regional_imbalance_improvement_percent"]
        for name in names
    ]
    network = [
        profiles[name]["envelope_vs_dwell"]["network_vehicle_hours_difference_percent"]
        for name in names
    ]
    match = [
        profiles[name]["envelope_vs_dwell"]["priority_match_improvement_percentage_points"]
        for name in names
    ]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 3.5))
    width = 0.36
    axes[0].bar(x - width / 2, priority, width, label="Priority cost", color="#35618f")
    axes[0].bar(x + width / 2, imbalance, width, label="Regional imbalance", color="#6f9e75")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Envelope improvement over dwell (%)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, match, color="#8a6d3b")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Priority-match improvement (percentage points)")
    axes[2].bar(x, network, color="#8f5a5a")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Network vehicle-hours difference (%)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)

    summary = {
        name: {
            "desired_priority_changes": profiles[name]["desired_priority_changes"],
            "envelope_changes": profiles[name]["runs"]["time_envelope"]["executed_changes"],
            "dwell_changes": profiles[name]["runs"]["dwell_certified"]["executed_changes"],
            **profiles[name]["envelope_vs_dwell"],
            "envelope_maximum_tube_occupancy": profiles[name]["runs"]["time_envelope"]["maximum_tube_occupancy"],
            "dwell_maximum_tube_occupancy": profiles[name]["runs"]["dwell_certified"]["maximum_tube_occupancy"],
        }
        for name in names
    }
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
