#!/usr/bin/env python3
"""Aggregate the development/held-out cubic-MFD LLM interface study."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "live_llm_results" / "cubic" / "prompt_improvement"
JSON_PATH = INPUT_ROOT / "mfd_cubic_prompt_improvement_aggregate.json"
CSV_PATH = INPUT_ROOT / "mfd_cubic_prompt_improvement_cells.csv"
PDF_PATH = INPUT_ROOT / "mfd_cubic_prompt_improvement_aggregate.pdf"
PNG_PATH = INPUT_ROOT / "mfd_cubic_prompt_improvement_aggregate.png"
MODELS = ("llama3.1:latest", "qwen3:8b", "mistral:7b")
BASELINE = "current_rule_first"


def read_records(phase: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((INPUT_ROOT / phase).glob("*/*_summary.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["source"] = str(path.relative_to(ROOT))
        records.append(record)
    return records


def cost_improvement(record: dict[str, Any]) -> float:
    envelope = record["results"]["time_envelope"]["priority_tracking_cost"]
    dwell = record["results"]["dwell_certified"]["priority_tracking_cost"]
    return 100.0 * (dwell - envelope) / dwell


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    calls = sum(record["number_of_live_calls"] for record in records)
    matches = sum(record["lower_cost_matches"] for record in records)
    improvements = [cost_improvement(record) for record in records]
    occupancies = [
        record["results"][policy]["maximum_tube_occupancy"]
        for record in records
        for policy in ("landing_only", "time_envelope", "dwell_certified")
    ]
    grounding = [
        record["maximum_grounding_error"]
        for record in records
        if record["maximum_grounding_error"] is not None
    ]
    return {
        "configuration_count": len(records),
        "total_live_calls": calls,
        "lower_cost_matches": matches,
        "lower_cost_match_rate": matches / calls,
        "configuration_match_rate_min": min(record["lower_cost_match_rate"] for record in records),
        "configuration_match_rate_max": max(record["lower_cost_match_rate"] for record in records),
        "envelope_cost_improvement_percent_mean": float(np.mean(improvements)),
        "envelope_cost_improvement_percent_min": min(improvements),
        "envelope_cost_improvement_percent_max": max(improvements),
        "envelope_executed_changes_min": min(record["results"]["time_envelope"]["executed_changes"] for record in records),
        "envelope_executed_changes_max": max(record["results"]["time_envelope"]["executed_changes"] for record in records),
        "dwell_executed_changes_min": min(record["results"]["dwell_certified"]["executed_changes"] for record in records),
        "dwell_executed_changes_max": max(record["results"]["dwell_certified"]["executed_changes"] for record in records),
        "maximum_certified_tube_occupancy": max(occupancies),
        "all_certified_paths_inside_tube": max(occupancies) <= 1.0 + 1.0e-10,
        "maximum_cost_copy_error": max(grounding, default=None),
        "mean_latency_seconds": float(np.mean([record["mean_latency_seconds"] for record in records])),
    }


def group_records(
    records: list[dict[str, Any]], key_name: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[key_name]].append(record)
    return grouped


def main() -> None:
    development = read_records("development")
    held_out = read_records("held_out")
    assert len(development) == 3 * 4 * 2
    assert len(held_out) == 3 * 2 * 5

    development_by_variant = group_records(development, "variant")
    development_cells: dict[str, Any] = {}
    selection_scores: dict[str, tuple[float, float, float]] = {}
    for variant, records in development_by_variant.items():
        by_model = group_records(records, "model")
        model_aggregates = {model: aggregate(by_model[model]) for model in MODELS}
        overall = aggregate(records)
        worst_model_rate = min(item["lower_cost_match_rate"] for item in model_aggregates.values())
        selection_scores[variant] = (
            worst_model_rate,
            overall["lower_cost_match_rate"],
            -overall["mean_latency_seconds"],
        )
        development_cells[variant] = {
            "overall": overall,
            "by_model": model_aggregates,
            "worst_model_match_rate": worst_model_rate,
        }

    selected = max(selection_scores, key=selection_scores.get)
    assert selected == "grounded_numeric"

    held_by_variant = group_records(held_out, "variant")
    assert set(held_by_variant) == {BASELINE, selected}
    held_cells: dict[str, Any] = {}
    for variant, records in held_by_variant.items():
        by_model = group_records(records, "model")
        held_cells[variant] = {
            "overall": aggregate(records),
            "by_model": {model: aggregate(by_model[model]) for model in MODELS},
        }

    baseline_rate = held_cells[BASELINE]["overall"]["lower_cost_match_rate"]
    selected_rate = held_cells[selected]["overall"]["lower_cost_match_rate"]
    payload = {
        "experiment": "cubic-MFD theorem-neutral LLM interface improvement",
        "fixed_components": [
            "literature-cubic MFD plant",
            "controller library",
            "15-minute rollout-cost definition",
            "priority schedules within each paired configuration",
            "landing, state-envelope, and dwell certificates",
            "nonlinear replay policies",
        ],
        "development": {
            "scenario_seeds": [20260817, 20260818],
            "selection_rule": "lexicographically maximize worst-model match rate, then overall match rate, then minimize mean latency",
            "variants": development_cells,
            "selected_variant": selected,
        },
        "held_out": {
            "scenario_seeds": [20260821, 20260822, 20260823, 20260824, 20260825],
            "variants": held_cells,
            "absolute_match_rate_gain": selected_rate - baseline_rate,
            "relative_error_reduction": 1.0
            - (1.0 - selected_rate) / (1.0 - baseline_rate),
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "phase",
                "variant",
                "model",
                "live_calls",
                "match_rate",
                "cost_improvement_mean_percent",
                "cost_improvement_min_percent",
                "cost_improvement_max_percent",
                "maximum_tube_occupancy",
            ),
        )
        writer.writeheader()
        for phase, cells in (("development", development_cells), ("held_out", held_cells)):
            for variant, cell in cells.items():
                for model in MODELS:
                    item = cell["by_model"][model]
                    writer.writerow(
                        {
                            "phase": phase,
                            "variant": variant,
                            "model": model,
                            "live_calls": item["total_live_calls"],
                            "match_rate": item["lower_cost_match_rate"],
                            "cost_improvement_mean_percent": item["envelope_cost_improvement_percent_mean"],
                            "cost_improvement_min_percent": item["envelope_cost_improvement_percent_min"],
                            "cost_improvement_max_percent": item["envelope_cost_improvement_percent_max"],
                            "maximum_tube_occupancy": item["maximum_certified_tube_occupancy"],
                        }
                    )

    labels = ("Llama 3.1 8B", "Qwen3 8B", "Mistral 7B")
    x = np.arange(len(MODELS))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    colors = ("#8c564b", "#2a9d8f")
    for offset, variant in enumerate((BASELINE, selected)):
        values = [100.0 * held_cells[variant]["by_model"][model]["lower_cost_match_rate"] for model in MODELS]
        bars = axes[0].bar(x + (offset - 0.5) * width, values, width, color=colors[offset], label=("Current" if offset == 0 else "Grounded"))
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("lower-cost selections (%)")
    axes[0].set_title("Held-out proposal accuracy")
    axes[0].legend(frameon=False, loc="lower center")
    axes[0].grid(axis="y", alpha=0.25)

    for offset, variant in enumerate((BASELINE, selected)):
        values = [held_cells[variant]["by_model"][model]["envelope_cost_improvement_percent_mean"] for model in MODELS]
        bars = axes[1].bar(x + (offset - 0.5) * width, values, width, color=colors[offset], label=("Current" if offset == 0 else "Grounded"))
        axes[1].bar_label(bars, fmt="%+.2f", fontsize=8, padding=2)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("envelope improvement over dwell (%)")
    axes[1].set_title("Held-out operational cost")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Theorem-neutral grounded LLM interface; five held-out schedules per model", fontsize=11)
    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(payload["held_out"], indent=2))


if __name__ == "__main__":
    main()
