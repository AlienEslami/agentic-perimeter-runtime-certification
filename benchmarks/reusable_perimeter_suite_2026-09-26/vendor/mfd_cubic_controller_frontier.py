#!/usr/bin/env python3
"""Audit and plot the declared cubic-MFD controller-selection frontier."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_controller_frontier_results.json"
FIGURE_PDF = RESULTS_DIR / "mfd_cubic_controller_frontier.pdf"
FIGURE_PNG = RESULTS_DIR / "mfd_cubic_controller_frontier.png"


def main() -> None:
    record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    search = record["controller_search"]["operational_frontier"]
    frontier = search["frontier"]
    selected_scale = float(search["selected_scale"])
    eligible = [
        item for item in frontier if item["one_minute_monodromy_radius"] >= 1.005
    ]
    selected_by_rule = max(
        eligible, key=lambda item: item["analytic_common_radius_vehicles"]
    )
    if float(selected_by_rule["actuator_penalty_scale"]) != selected_scale:
        raise RuntimeError("archived controller no longer satisfies the declared selection rule")
    summary = {
        "declared_selection_rule": search["selection_rule"],
        "candidate_count": len(frontier),
        "one_minute_amplification_count": sum(
            item["one_minute_monodromy_radius"] > 1.0 for item in frontier
        ),
        "audit_threshold_count": len(eligible),
        "audit_threshold": 1.005,
        "selected_scale": selected_scale,
        "selected": selected_by_rule,
        "ranges": {
            "one_minute_monodromy_radius": [
                min(item["one_minute_monodromy_radius"] for item in frontier),
                max(item["one_minute_monodromy_radius"] for item in frontier),
            ],
            "analytic_common_radius_vehicles": [
                min(item["analytic_common_radius_vehicles"] for item in frontier),
                max(item["analytic_common_radius_vehicles"] for item in frontier),
            ],
            "dwell_minutes": [
                min(item["dwell_minutes"] for item in frontier),
                max(item["dwell_minutes"] for item in frontier),
            ],
        },
        "frontier": frontier,
        "interpretation": (
            "fast-switch amplification is present for eight of nine neighboring "
            "actuator-penalty scales; the selected controller is the largest analytic "
            "tube among the six candidates meeting the declared 0.5% audit threshold"
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    scales = np.asarray([item["actuator_penalty_scale"] for item in frontier])
    radii = np.asarray([item["analytic_common_radius_vehicles"] for item in frontier])
    rho = np.asarray([item["one_minute_monodromy_radius"] for item in frontier])
    dwell = np.asarray([item["dwell_minutes"] for item in frontier])
    selected = int(np.flatnonzero(np.isclose(scales, selected_scale))[0])
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))
    axes[0].plot(scales, rho, "o-", color="#8f4b45")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    axes[0].axhline(1.005, color="#555555", linestyle=":", linewidth=0.9)
    axes[0].set_ylabel("One-minute monodromy radius")
    axes[1].plot(scales, radii, "o-", color="#35618f")
    axes[1].set_ylabel("Analytic common radius (vehicles)")
    axes[2].plot(scales, dwell, "o-", color="#4a8d68")
    axes[2].set_ylabel("Certified dwell (minutes)")
    for axis, values in zip(axes, (rho, radii, dwell)):
        axis.scatter(
            [scales[selected]],
            [values[selected]],
            s=75,
            facecolor="#e2a63b",
            edgecolor="black",
            zorder=4,
            label="Selected",
        )
        axis.set_xlabel("Actuator-penalty scale")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({key: value for key, value in summary.items() if key != "frontier"}, indent=2))


if __name__ == "__main__":
    main()
