#!/usr/bin/env python3
"""Aggregate live MFD prompt/seed grids across installed model families."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PROMPT_ORDER = ["current", "costs_reversed", "neutral"]


def display_model(model: str) -> str:
    labels = {
        "llama3.1:latest": "Llama 3.1 8B",
        "qwen3:8b": "Qwen3 8B",
        "mistral:7b": "Mistral 7B",
    }
    return labels.get(model, model.replace(":latest", "").replace(":", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configurations: list[dict] = []
    sources = []
    for relative_path in args.summaries:
        path = relative_path if relative_path.is_absolute() else ROOT / relative_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        configurations.extend(payload["configurations"])
        sources.append(str(path.relative_to(ROOT)))

    models = list(dict.fromkeys(item["model"] for item in configurations))
    cells = []
    for model in models:
        for prompt in PROMPT_ORDER:
            group = [
                item for item in configurations
                if item["model"] == model and item["prompt_template"] == prompt
            ]
            if not group:
                continue
            switches = [item["unrestricted"]["executed_switches"] for item in group]
            exits = [item["unrestricted"]["first_tube_exit_time_h"] for item in group]
            finite_exits = [value for value in exits if value is not None]
            cells.append({
                "model": model,
                "prompt_template": prompt,
                "forecast_seeds": len(group),
                "live_calls": int(sum(item["calls"] for item in group)),
                "mean_lower_cost_match_rate": float(np.mean([
                    item["lower_cost_match_rate"] for item in group
                ])),
                "unrestricted_switch_min": int(min(switches)),
                "unrestricted_switch_max": int(max(switches)),
                "unrestricted_switch_mean": float(np.mean(switches)),
                "unrestricted_earliest_exit_h": (
                    float(min(finite_exits)) if finite_exits else None
                ),
                "unrestricted_exit_count": len(finite_exits),
                "unrestricted_maximum_occupancy": float(max(
                    item["unrestricted"]["maximum_tube_occupancy"] for item in group
                )),
                "certified_switch_values": sorted(set(
                    item["certified"]["executed_switches"] for item in group
                )),
                "certified_maximum_occupancy": float(max(
                    item["certified"]["maximum_tube_occupancy"] for item in group
                )),
            })

    all_finite_exits = [
        item["unrestricted"]["first_tube_exit_time_h"]
        for item in configurations
        if item["unrestricted"]["first_tube_exit_time_h"] is not None
    ]
    model_summaries = []
    for model in models:
        group = [item for item in configurations if item["model"] == model]
        model_exits = [
            item["unrestricted"]["first_tube_exit_time_h"]
            for item in group
            if item["unrestricted"]["first_tube_exit_time_h"] is not None
        ]
        model_summaries.append({
            "model": model,
            "configurations": len(group),
            "live_calls": int(sum(item["calls"] for item in group)),
            "lower_cost_match_rate_mean": float(np.mean([
                item["lower_cost_match_rate"] for item in group
            ])),
            "lower_cost_match_rate_min": float(min(
                item["lower_cost_match_rate"] for item in group
            )),
            "lower_cost_match_rate_max": float(max(
                item["lower_cost_match_rate"] for item in group
            )),
            "unrestricted_switch_min": int(min(
                item["unrestricted"]["executed_switches"] for item in group
            )),
            "unrestricted_switch_max": int(max(
                item["unrestricted"]["executed_switches"] for item in group
            )),
            "unrestricted_earliest_exit_h": (
                float(min(model_exits)) if model_exits else None
            ),
            "unrestricted_exit_count": len(model_exits),
            "unrestricted_maximum_occupancy": float(max(
                item["unrestricted"]["maximum_tube_occupancy"] for item in group
            )),
            "certified_switch_min": int(min(
                item["certified"]["executed_switches"] for item in group
            )),
            "certified_switch_max": int(max(
                item["certified"]["executed_switches"] for item in group
            )),
            "certified_maximum_occupancy": float(max(
                item["certified"]["maximum_tube_occupancy"] for item in group
            )),
        })
    aggregate = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": sources,
        "models": models,
        "prompt_templates": PROMPT_ORDER,
        "forecast_seeds_per_cell": 5,
        "configurations": len(configurations),
        "live_calls": int(sum(item["calls"] for item in configurations)),
        "model_summaries": model_summaries,
        "cells": cells,
        "overall": {
            "mean_lower_cost_match_rate": float(np.mean([
                item["lower_cost_match_rate"] for item in configurations
            ])),
            "unrestricted_switch_min": int(min(
                item["unrestricted"]["executed_switches"] for item in configurations
            )),
            "unrestricted_switch_max": int(max(
                item["unrestricted"]["executed_switches"] for item in configurations
            )),
            "unrestricted_earliest_exit_h": (
                float(min(all_finite_exits)) if all_finite_exits else None
            ),
            "certified_maximum_occupancy": float(max(
                item["certified"]["maximum_tube_occupancy"] for item in configurations
            )),
        },
    }
    json_path = RESULTS_DIR / "mfd_llm_model_prompt_grid.json"
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    csv_path = RESULTS_DIR / "mfd_llm_model_prompt_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    switch_matrix = np.full((len(models), len(PROMPT_ORDER)), np.nan)
    match_matrix = np.full_like(switch_matrix, np.nan)
    switch_labels = np.empty(switch_matrix.shape, dtype=object)
    for cell in cells:
        row = models.index(cell["model"])
        column = PROMPT_ORDER.index(cell["prompt_template"])
        switch_matrix[row, column] = cell["unrestricted_switch_mean"]
        match_matrix[row, column] = 100.0 * cell["mean_lower_cost_match_rate"]
        switch_labels[row, column] = (
            f"{cell['unrestricted_switch_min']}--{cell['unrestricted_switch_max']}"
        )

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)
    prompt_labels = ["Current", "Cost order\nreversed", "Neutral"]
    model_labels = [display_model(model) for model in models]
    images = [
        axes[0].imshow(switch_matrix, vmin=0, vmax=119, cmap="Blues"),
        axes[1].imshow(match_matrix, vmin=0, vmax=100, cmap="Greens"),
    ]
    axes[0].set_title("Unrestricted changes (seed range)")
    axes[1].set_title("Lower-cost selections (mean %)")
    for axis in axes:
        axis.set_xticks(range(len(PROMPT_ORDER)), prompt_labels)
        axis.set_yticks(range(len(models)), model_labels)
        axis.tick_params(labelsize=8)
    for row in range(len(models)):
        for column in range(len(PROMPT_ORDER)):
            if np.isfinite(switch_matrix[row, column]):
                axes[0].text(
                    column, row, switch_labels[row, column], ha="center", va="center",
                    fontsize=8, color="white" if switch_matrix[row, column] >= 55 else "black",
                )
                axes[1].text(
                    column, row, f"{match_matrix[row, column]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if match_matrix[row, column] >= 70 else "black",
                )
    for axis, image in zip(axes, images):
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(RESULTS_DIR / "mfd_llm_model_prompt_grid.pdf", bbox_inches="tight")
    figure.savefig(RESULTS_DIR / "mfd_llm_model_prompt_grid.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(aggregate["overall"], indent=2))


if __name__ == "__main__":
    main()
