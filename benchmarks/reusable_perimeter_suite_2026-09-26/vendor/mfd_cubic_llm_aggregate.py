#!/usr/bin/env python3
"""Aggregate the checked three-model cubic-MFD live proposal experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "live_llm_results" / "cubic"
OUTPUT_JSON = INPUT_DIR / "mfd_cubic_live_v2_aggregate.json"
OUTPUT_CSV = INPUT_DIR / "mfd_cubic_live_v2_aggregate.csv"
OUTPUT_PDF = INPUT_DIR / "mfd_cubic_live_v2_aggregate.pdf"
OUTPUT_PNG = INPUT_DIR / "mfd_cubic_live_v2_aggregate.png"

FILES = (
    "mfd_cubic_live_v2_llama3_1_latest_summary.json",
    "mfd_cubic_live_v2_qwen3_8b_summary.json",
    "mfd_cubic_live_v2_mistral_7b_summary.json",
)


def main() -> None:
    records = []
    for name in FILES:
        path = INPUT_DIR / name
        if not path.exists():
            raise SystemExit(f"Missing live result: {path}")
        source = json.loads(path.read_text(encoding="utf-8"))
        envelope = source["results"]["time_envelope"]
        dwell = source["results"]["dwell_certified"]
        records.append(
            {
                "model": source["model"],
                "live_calls": source["number_of_live_calls"],
                "lower_cost_matches": source["lower_cost_matches"],
                "lower_cost_match_rate": source["lower_cost_match_rate"],
                "envelope_executed_changes": envelope["executed_changes"],
                "envelope_rejected_changes": envelope["rejected_changes"],
                "envelope_no_op_reselections": envelope["no_op_reselections"],
                "envelope_maximum_tube_occupancy": envelope["maximum_tube_occupancy"],
                "envelope_priority_tracking_cost": envelope["priority_tracking_cost"],
                "dwell_executed_changes": dwell["executed_changes"],
                "dwell_rejected_changes": dwell["rejected_changes"],
                "dwell_no_op_reselections": dwell["no_op_reselections"],
                "dwell_maximum_tube_occupancy": dwell["maximum_tube_occupancy"],
                "dwell_priority_tracking_cost": dwell["priority_tracking_cost"],
            }
        )
    output = {
        "prompt_version": "mfd-cubic-operational-cost-v2",
        "models": records,
        "total_live_calls": sum(item["live_calls"] for item in records),
        "all_certified_paths_inside_tube": all(
            item["envelope_maximum_tube_occupancy"] <= 1.0 + 1.0e-9
            and item["dwell_maximum_tube_occupancy"] <= 1.0 + 1.0e-9
            for item in records
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    labels = [item["model"].replace(":latest", "") for item in records]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
    axes[0].bar(labels, [100.0 * item["lower_cost_match_rate"] for item in records], color="#35618f")
    axes[0].set_ylabel("lower-cost selections (%)")
    axes[0].set_ylim(0.0, 105.0)
    width = 0.36
    positions = list(range(len(records)))
    axes[1].bar(
        [position - width / 2 for position in positions],
        [item["envelope_executed_changes"] for item in records],
        width=width,
        label="state envelope",
        color="#35618f",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        [item["dwell_executed_changes"] for item in records],
        width=width,
        label="fixed dwell",
        color="#4a8d68",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("executed changes")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].bar(
        [position - width / 2 for position in positions],
        [item["envelope_priority_tracking_cost"] for item in records],
        width=width,
        color="#35618f",
    )
    axes[2].bar(
        [position + width / 2 for position in positions],
        [item["dwell_priority_tracking_cost"] for item in records],
        width=width,
        color="#4a8d68",
    )
    axes[2].set_xticks(positions, labels)
    axes[2].set_ylabel("priority-tracking cost")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
