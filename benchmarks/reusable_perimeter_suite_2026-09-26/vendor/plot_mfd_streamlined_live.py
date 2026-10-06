"""Readable two-panel live-agent summary from the preserved 240-call record.

Counts and denominators are read from each model/interface record. No live
requests are made and neither the source records nor their original figure
are overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "live_llm_results/cubic/sequential_semantic_agent_v3/mfd_sequential_semantic_agent_aggregate.json"
OUTPUT = ROOT / "reproducibility/results/mfd_streamlined_live"


def main():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    models = payload["models"]
    assert models == ["llama3.1:latest", "qwen3:8b", "mistral:7b"]
    labels = ["Llama 3.1\n8B", "Qwen3\n8B", "Mistral\n7B"]
    variants = (("stateful", "Stateful", "#0072B2", -.18),
                ("memoryless", "Memoryless", "#D55E00", .18))
    panels = (
        ("correct_context_requests", "context_request_count", 15,
         "(a) Context resolution", "Correct requests (%)"),
        ("true_oracle_proposals", "proposal_count", 40,
         "(b) Downstream choice", "Matching proposals (%)"),
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "axes.linewidth": .75,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.05))
    fig.subplots_adjust(left=.092, right=.986, bottom=.19, top=.73, wspace=.38)
    x = np.arange(len(models))
    counts = {}
    for ax, (numerator, denominator, expected, title, ylabel) in zip(axes, panels):
        ax.set_ylim(0, 115)
        ax.set_xlim(-.6, len(models) - .4)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=11, fontweight="bold")
        ax.grid(axis="y", color="#D5D9DD", linewidth=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#697078")
        ax.tick_params(width=.6, length=3, color="#697078")
        ax.tick_params(axis="x", length=0, pad=7)
        counts[numerator] = {}
        for variant, legend, color, offset in variants:
            summaries = [payload["runs"][model][variant]["summary"] for model in models]
            num = [row[numerator] for row in summaries]
            den = [row[denominator] for row in summaries]
            assert den == [expected] * len(models)
            values = 100 * np.asarray(num) / np.asarray(den)
            ax.bar(x + offset, values, width=.34, color=color, label=legend, zorder=3)
            for xpos, value, n, d in zip(x + offset, values, num, den):
                # Place each count outward from the pair's center so labels
                # never cover the neighboring bar or mask its actual height.
                text_x = xpos + .16 if variant == "stateful" else xpos - .16
                alignment = "right" if variant == "stateful" else "left"
                ax.text(text_x, value + 2.5, f"{n}/{d}", ha=alignment, va="bottom",
                        fontsize=12, color=color)
            counts[numerator][variant] = {
                model: {"count": n, "denominator": d, "percent": float(value)}
                for model, n, d, value in zip(models, num, den, values)
            }
    handles = [Patch(facecolor=color, label=legend) for _, legend, color, _ in variants]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=2, frameon=False, columnspacing=2.3, handlelength=1.6)
    fig.text(.54, .865, "Counts shown per model and interface across five traces",
             ha="center", va="center", fontsize=12, color="#545B63")
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240)
    plt.close(fig)

    total_calls = sum(payload["runs"][model][variant]["summary"]["call_count"]
                      for model in models for variant, _, _, _ in variants)
    assert total_calls == 240
    totals = {
        numerator: {variant: {
            "count": sum(row["count"] for row in counts[numerator][variant].values()),
            "denominator": sum(row["denominator"] for row in counts[numerator][variant].values()),
        } for variant, _, _, _ in variants} for numerator, *_ in panels
    }
    assert totals["correct_context_requests"]["stateful"] == {"count": 33, "denominator": 45}
    assert totals["correct_context_requests"]["memoryless"] == {"count": 1, "denominator": 45}
    assert totals["true_oracle_proposals"]["stateful"] == {"count": 115, "denominator": 120}
    assert totals["true_oracle_proposals"]["memoryless"] == {"count": 99, "denominator": 120}
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "figure_size_inches": [7.8, 4.05], "minimum_font_points": 12,
        "minimum_font_at_5_5_inches_points": 12 * 5.5 / 7.8,
        "source_live_calls": total_calls, "models": models, "counts": counts,
        "aggregate_checks": totals,
        "scope": "Context-request resolution and resulting true-objective controller choice are distinct stages; trace dependence is preserved and no uncertainty bars or model-ranking inference is added",
        "preservation": "All original raw calls, trajectories, event counts, and original four-panel figure remain unchanged",
    }
    OUTPUT.with_name(OUTPUT.name + "_figuredata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
