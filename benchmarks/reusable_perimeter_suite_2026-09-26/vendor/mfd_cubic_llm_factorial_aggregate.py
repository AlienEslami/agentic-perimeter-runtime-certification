#!/usr/bin/env python3
"""Aggregate the 45-cell, 1,080-call cubic-MFD live factorial."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import live_llm_mfd_cubic_validation as live
from mfd_cubic_llm_factorial import PROMPT_TEMPLATES, SCENARIO_SEEDS


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "live_llm_results" / "cubic" / "factorial"
OUTPUT_JSON = INPUT_DIR / "mfd_cubic_factorial_aggregate.json"
OUTPUT_CSV = INPUT_DIR / "mfd_cubic_factorial_cells.csv"
OUTPUT_PDF = INPUT_DIR / "mfd_cubic_factorial_aggregate.pdf"
OUTPUT_PNG = INPUT_DIR / "mfd_cubic_factorial_aggregate.png"
MODELS = ("llama3.1:latest", "qwen3:8b", "mistral:7b")


def display_model(model: str) -> str:
    return {
        "llama3.1:latest": "Llama 3.1 8B",
        "qwen3:8b": "Qwen3 8B",
        "mistral:7b": "Mistral 7B",
    }[model]


def display_prompt(prompt: str) -> str:
    return {
        "rule_first": "Rule first",
        "costs_first": "Costs first",
        "compact": "Compact",
    }[prompt]


def main() -> None:
    configurations = []
    sources = []
    model_records = {}
    for model in MODELS:
        model_key = live.safe_model_name(model)
        path = INPUT_DIR / f"mfd_cubic_factorial_{model_key}_summary.json"
        if not path.exists():
            raise SystemExit(f"Missing factorial model summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model"] == model
        assert payload["prompt_templates"] == list(PROMPT_TEMPLATES)
        assert payload["scenario_seeds"] == list(SCENARIO_SEEDS)
        configurations.extend(payload["configurations"])
        sources.append(str(path.relative_to(ROOT)))
        model_records[model] = payload["installed_model_record"]

    cells = []
    for model in MODELS:
        for prompt in PROMPT_TEMPLATES:
            group = [
                item
                for item in configurations
                if item["model"] == model and item["prompt_template"] == prompt
            ]
            assert len(group) == len(SCENARIO_SEEDS)
            envelope_changes = [
                item["results"]["time_envelope"]["executed_changes"] for item in group
            ]
            dwell_changes = [
                item["results"]["dwell_certified"]["executed_changes"] for item in group
            ]
            improvements = []
            for item in group:
                envelope_cost = item["results"]["time_envelope"]["priority_tracking_cost"]
                dwell_cost = item["results"]["dwell_certified"]["priority_tracking_cost"]
                improvements.append(100.0 * (dwell_cost - envelope_cost) / dwell_cost)
            cells.append(
                {
                    "model": model,
                    "prompt_template": prompt,
                    "scenario_count": len(group),
                    "live_calls": sum(item["number_of_live_calls"] for item in group),
                    "lower_cost_match_rate_mean": float(
                        np.mean([item["lower_cost_match_rate"] for item in group])
                    ),
                    "lower_cost_match_rate_min": float(
                        min(item["lower_cost_match_rate"] for item in group)
                    ),
                    "lower_cost_match_rate_max": float(
                        max(item["lower_cost_match_rate"] for item in group)
                    ),
                    "envelope_executed_changes_mean": float(np.mean(envelope_changes)),
                    "envelope_executed_changes_min": int(min(envelope_changes)),
                    "envelope_executed_changes_max": int(max(envelope_changes)),
                    "dwell_executed_changes_mean": float(np.mean(dwell_changes)),
                    "dwell_executed_changes_min": int(min(dwell_changes)),
                    "dwell_executed_changes_max": int(max(dwell_changes)),
                    "envelope_cost_improvement_percent_mean": float(
                        np.mean(improvements)
                    ),
                    "envelope_cost_improvement_percent_min": float(min(improvements)),
                    "envelope_cost_improvement_percent_max": float(max(improvements)),
                    "maximum_certified_tube_occupancy": float(
                        max(
                            item["results"][policy]["maximum_tube_occupancy"]
                            for item in group
                            for policy in (
                                "landing_only",
                                "time_envelope",
                                "dwell_certified",
                            )
                        )
                    ),
                }
            )

    output = {
        "experiment": "cubic-MFD live prompt/scenario factorial",
        "sources": sources,
        "installed_model_records": model_records,
        "models": list(MODELS),
        "prompt_templates": list(PROMPT_TEMPLATES),
        "scenario_seeds": list(SCENARIO_SEEDS),
        "configuration_count": len(configurations),
        "total_live_calls": sum(item["number_of_live_calls"] for item in configurations),
        "cells": cells,
        "overall": {
            "mean_lower_cost_match_rate": float(
                np.mean([item["lower_cost_match_rate"] for item in configurations])
            ),
            "minimum_lower_cost_match_rate": float(
                min(item["lower_cost_match_rate"] for item in configurations)
            ),
            "maximum_lower_cost_match_rate": float(
                max(item["lower_cost_match_rate"] for item in configurations)
            ),
            "envelope_executed_changes_min": int(
                min(
                    item["results"]["time_envelope"]["executed_changes"]
                    for item in configurations
                )
            ),
            "envelope_executed_changes_max": int(
                max(
                    item["results"]["time_envelope"]["executed_changes"]
                    for item in configurations
                )
            ),
            "dwell_executed_changes_min": int(
                min(
                    item["results"]["dwell_certified"]["executed_changes"]
                    for item in configurations
                )
            ),
            "dwell_executed_changes_max": int(
                max(
                    item["results"]["dwell_certified"]["executed_changes"]
                    for item in configurations
                )
            ),
            "maximum_certified_tube_occupancy": float(
                max(item["maximum_certified_tube_occupancy"] for item in cells)
            ),
            "all_certified_paths_inside_tube": all(
                item["maximum_certified_tube_occupancy"] <= 1.0 + 1.0e-9
                for item in cells
            ),
        },
    }
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    shapes = (len(MODELS), len(PROMPT_TEMPLATES))
    match = np.full(shapes, np.nan)
    changes = np.full(shapes, np.nan)
    improvement = np.full(shapes, np.nan)
    for cell in cells:
        row = MODELS.index(cell["model"])
        column = PROMPT_TEMPLATES.index(cell["prompt_template"])
        match[row, column] = 100.0 * cell["lower_cost_match_rate_mean"]
        changes[row, column] = cell["envelope_executed_changes_mean"]
        improvement[row, column] = cell["envelope_cost_improvement_percent_mean"]

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)
    images = (
        axes[0].imshow(match, vmin=0.0, vmax=100.0, cmap="Blues"),
        axes[1].imshow(changes, vmin=0.0, vmax=24.0, cmap="Greens"),
        axes[2].imshow(
            improvement,
            vmin=min(-1.0, float(np.nanmin(improvement))),
            vmax=max(1.0, float(np.nanmax(improvement))),
            cmap="RdYlGn",
        ),
    )
    titles = (
        "Lower-cost selections (%)",
        "Envelope changes (mean)",
        "Envelope cost improvement (%)",
    )
    matrices = (match, changes, improvement)
    formats = ("{:.1f}", "{:.1f}", "{:+.1f}")
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=9)
        axis.set_xticks(
            range(len(PROMPT_TEMPLATES)),
            [display_prompt(item) for item in PROMPT_TEMPLATES],
            rotation=20,
            ha="right",
        )
        axis.set_yticks(range(len(MODELS)), [display_model(item) for item in MODELS])
        axis.tick_params(labelsize=8)
    for axis, matrix, formatter in zip(axes, matrices, formats):
        midpoint = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2.0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    formatter.format(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > midpoint else "black",
                )
    for axis, image in zip(axes, images):
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(output["overall"], indent=2))


if __name__ == "__main__":
    main()
