"""Streamlined manuscript figures from frozen incident and certified-MPC records.

The nominal and certified MPC curves are plotted once only after exact trajectory
and command equality checks. Trace integrals use the stored 10-second samples;
endpoint labels use the independently audited adaptive-replay table. No source
record is changed, no simulation is rerun, and no curve is fitted or rescaled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from plot_mfd_incident_transportation import (
    analytic_incumbent, endpoint_record, physical_curve, primary, shade_incident,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCES = {
    "library": RESULTS / "mfd_incident_operational_v2_certified_primary_results.json",
    "nominal_mpc": RESULTS / "mfd_incident_mpc_grid_results.json",
    "certified_mpc": RESULTS / "mfd_incident_mpc_certified_grid_results.json",
    "comparisons": RESULTS / "mfd_incident_results_comparison.json",
    "summary": RESULTS / "mfd_incident_mpc_certification_summary_results.json",
}
OUTPUT = RESULTS / "mfd_streamlined_mpc"
ARCHITECTURE = RESULTS / "mfd_streamlined_architecture"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_case(payload):
    return next(c for c in payload["cases"]
                if c["amplitude"] == .3 and c["duration_h"] == 1.)


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "axes.linewidth": .75,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 2., "savefig.facecolor": "white",
    })


def serializable_curve(curve):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in curve.items()}


def plot_traffic():
    loaded = {key: load(path) for key, path in SOURCES.items()}
    library = primary(loaded["library"])
    nominal = primary(loaded["nominal_mpc"])
    certified = primary(loaded["certified_mpc"])
    table = first_case(loaded["comparisons"])
    summary = loaded["summary"]
    summary_case = first_case(summary)

    assert summary["source_sha256"][SOURCES["certified_mpc"].name] == digest(SOURCES["certified_mpc"])
    assert summary["source_sha256"][SOURCES["comparisons"].name] == digest(SOURCES["comparisons"])
    assert certified["metrics"]["full_horizon_completed"]
    for field in ("time_h", "state", "command"):
        assert np.array_equal(np.asarray(nominal["trajectory"][field]),
                              np.asarray(certified["trajectory"][field])), field
    assert summary_case["nominal_mpc_tts_veh_h"] == summary_case["certified_mpc_tts_veh_h"]
    assert certified["metrics"]["successful_certificates"] == 24
    assert certified["metrics"]["failed_certificates"] == 0

    curves = {
        "incumbent": analytic_incumbent(np.linspace(0, 2, 721)),
        "library": physical_curve(library["times"], library["states"],
                                  str(SOURCES["library"].relative_to(HERE.parent))),
        "mpc": physical_curve(certified["trajectory"]["time_h"],
                              certified["trajectory"]["state"],
                              str(SOURCES["certified_mpc"].relative_to(HERE.parent))),
    }
    endpoints = {key: endpoint_record(curve) for key, curve in curves.items()}
    adaptive_tts = {
        "incumbent": endpoints["incumbent"]["total_time_spent_veh_h"],
        "library": table["library"]["tts_veh_h"],
        "mpc": table["mpc"]["tts_veh_h"],
    }
    quadrature_differences = {
        key: endpoints[key]["total_time_spent_veh_h"] - value
        for key, value in adaptive_tts.items()
    }
    assert max(abs(x) for x in quadrature_differences.values()) < .002
    assert abs(endpoints["incumbent"]["total_time_spent_veh_h"] - 14462.5) < 1e-8
    assert abs(endpoints["incumbent"]["maximum_queue_vehicles"] - 1850) < 1e-8
    assert table["mpc"]["tts_veh_h"] == certified["independent_replay"]["total_time_spent_veh_h"]
    reduction = 100 * (1 - adaptive_tts["mpc"] / adaptive_tts["library"])
    assert abs(reduction - summary_case["tts_reduction_percent_of_library"]) < 1e-10

    styles = {
        "incumbent": dict(color="#626972", linestyle=(0, (4, 2)),
                          label="Queue-blind incumbent (analytic)"),
        "library": dict(color="#0072B2", linestyle="-",
                        label="Certified feedback library"),
        "mpc": dict(color="#D55E00", linestyle="-",
                    label="Nominal MPC = certified MPC"),
    }
    fig = plt.figure(figsize=(7.8, 6.35))
    grid = fig.add_gridspec(2, 2, left=.12, right=.975, bottom=.085, top=.76,
                           hspace=.63, wspace=.45, height_ratios=[1, 1.05])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]
    panels = [
        ("region1_vehicles", "Region 1 accumulation (veh)", "(a) Regional service", (1400, 3400)),
        ("queue_vehicles", "Exterior queue (veh)", "(b) Boundary storage", (0, 2400)),
        ("cumulative_tts_veh_h", "Cumulative TTS (veh-h)", "(c) Queue-inclusive total time spent", (0, 16000)),
    ]
    for ax, (field, ylabel, title, limits) in zip(axes, panels):
        ax.set_xlim(0, 120)
        ax.set_ylim(*limits)
        shade_incident(ax)
        for key, curve in curves.items():
            ax.plot(60 * curve["time_h"], curve[field], **styles[key], zorder=3)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        ax.set_xticks([0, 30, 60, 90, 120])
        ax.grid(axis="y", color="#D5D9DD", linewidth=.55, alpha=.72, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#697078")
        ax.tick_params(width=.6, length=3, color="#697078")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_yticks([1500, 2000, 2500, 3000])
    axes[0].plot([60, 60], [1400, 2850], color="#333333", ls=(0, (2, 3)), lw=.8, zorder=2)
    axes[0].plot(60, 2850, marker="D", markersize=4, color="#333333", zorder=5)
    axes[0].annotate("2,850 veh\nat 60 min", xy=(60, 2850), xytext=(88, 3220),
                     ha="center", va="center", color="#333333", fontsize=12,
                     arrowprops=dict(arrowstyle="-", color="#555555", lw=.8))
    axes[1].set_yticks([0, 500, 1000, 1500, 2000])
    axes[1].axhline(2000, color="#333333", ls=(0, (3, 3)), lw=.85, zorder=2)
    axes[1].text(119, 2130, "Storage: 2,000 veh", ha="right", va="bottom", fontsize=12, color="#333333")
    axes[2].set_yticks([0, 5000, 10000, 15000])
    for key, value in adaptive_tts.items():
        if key == "mpc":
            axes[2].annotate(f"{value:,.2f}", xy=(120, endpoints[key]["total_time_spent_veh_h"]),
                             xytext=(118, 6500), color=styles[key]["color"], fontsize=12,
                             ha="right", va="top",
                             arrowprops=dict(arrowstyle="-", color=styles[key]["color"], lw=.8))
        else:
            axes[2].text(118, value + 290, f"{value:,.2f}", color=styles[key]["color"],
                         fontsize=12, ha="right", va="bottom")
    handles = [Line2D([0], [0], **style) for style in styles.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=1, frameon=False, handlelength=3, labelspacing=.42)
    fig.text(.54, .824, "Shading: 30% arrival surge, 15–75 min; five-minute ramps",
             ha="center", va="center", fontsize=12, color="#545B63")
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=240)
    plt.close(fig)

    provenance = {
        "scope": "Primary incident; one coincident nominal/certified MPC curve after exact equality checks",
        "amplitude": .3, "duration_h": 1., "figure_size_inches": [7.8, 6.35],
        "minimum_font_points": 12, "intended_manuscript_width_inches": 5.5,
        "minimum_font_at_manuscript_width_points": 12 * 5.5 / 7.8,
        "source_sha256": {str(path.relative_to(HERE.parent)): digest(path) for path in SOURCES.values()},
        "nominal_and_certified_mpc_time_state_command_exactly_equal": True,
        "primary_certificates_accepted": 24, "primary_certificates_rejected": 0,
        "curves": {key: serializable_curve(curve) for key, curve in curves.items()},
        "trace_endpoints": endpoints,
        "adaptive_tts_endpoint_labels_veh_h": adaptive_tts,
        "trace_minus_adaptive_tts_veh_h": quadrature_differences,
        "primary_tts_reduction_percent_of_library": reduction,
        "cumulative_curve_method": "Exact analytic incumbent; cumulative trapezoid of stored 10-second physical inventory for the library and MPC",
        "endpoint_label_method": "Independent adaptive-replay TTS from the existing audited comparison; no trace rescaling",
        "uncertainty_contract": "Initial full-state box +/-1 vehicle per coordinate; every measurable exterior arrival residual bounded by 60 vehicles/hour",
        "certificate_scope": "Finite prediction plan, physical constraints and regional deadline; TTS labels describe the matched nominal realizations",
        "timing_label_advice": "Total MPC-plus-certificate decision median 6.02 s/max 9.74 s; verification-only median 5.87 s; synchronous prototype and overlapping background validation",
    }
    OUTPUT.with_name(OUTPUT.name + "_figuredata.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"adaptive_tts": adaptive_tts, "trace_minus_adaptive": quadrature_differences,
            "reduction_percent_of_library": reduction}


def plot_architecture():
    """Vector-only overview; states/commands distinguish proposal and execution."""
    fig, ax = plt.subplots(figsize=(7.8, 4.25))
    ax.set_xlim(0, 7.8)
    ax.set_ylim(0, 4.25)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    blue = "#006B91"
    orange = "#B64C00"

    def box(x, y, w, h, title, body, color=blue, face="#F0F7FA"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.07",
                                   linewidth=1.2, edgecolor=color, facecolor=face))
        ax.text(x + w / 2, y + h - .20, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(x + w / 2, y + h / 2 - .15, body, ha="center", va="center",
                fontsize=12, linespacing=1.3, color="#25313B")

    def arrow(start, end, color="#52616C", style="-", connection=None):
        options = dict(arrowstyle="-|>", mutation_scale=12, lw=1.15,
                       color=color, linestyle=style, shrinkA=3, shrinkB=3)
        if connection is not None:
            options["connectionstyle"] = connection
        ax.add_patch(FancyArrowPatch(start, end, **options))

    box(.18, 2.60, 2.0, 1.17, "AI agent", "Memory + task\nState + feedback")
    box(2.88, 2.60, 2.0, 1.17, "Planning tool", "MPC or\nfeedback rollout")
    box(5.58, 2.60, 2.04, 1.17, "Plan certificate", "Inherited state set\nPhysical + service tests")
    box(5.58, .50, 2.04, 1.33, "Execution authority", "Pass: replace plan\nElse: retain suffix\nKeep original expiry", orange, "#FFF5ED")
    box(2.88, .50, 2.0, 1.33, "Traffic dynamics", "Nonlinear MFD\nDestination classes\nExterior queue")
    box(.18, .50, 2.0, 1.33, "Execution record", "Measured state\nAdmission result\nRemaining obligation")
    arrow((2.18, 3.16), (2.88, 3.16))
    ax.text(2.53, 3.47, "request", ha="center", fontsize=12, color="#52616C")
    arrow((4.88, 3.16), (5.58, 3.16))
    ax.text(5.23, 3.47, "plan", ha="center", fontsize=12, color="#52616C")
    arrow((6.60, 2.60), (6.60, 1.83), orange)
    ax.text(6.72, 2.21, "verdict", ha="left", va="center", fontsize=12, color=orange)
    arrow((5.58, 1.16), (4.88, 1.16), orange)
    ax.text(5.23, .77, "$u(t)$", ha="center", fontsize=12, color=orange)
    arrow((2.88, 1.16), (2.18, 1.16))
    ax.text(2.53, .68, "state", ha="center", fontsize=12, color="#52616C")
    arrow((1.18, 1.83), (1.18, 2.60))
    ax.text(1.30, 2.20, "update", ha="left", va="center", fontsize=12, color="#52616C")
    # Certifier receives the trusted physical information independently of the
    # proposal path; the record also returns the verdict to the agent.
    arrow((2.18, 1.65), (5.58, 2.68), style=(0, (3, 2)))
    ax.text(3.90, 2.23, "trusted state + outstanding task", ha="center", va="center",
            fontsize=12, color="#52616C", bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.plot([6.60, 6.60, 1.18], [.50, .27, .27], color=orange, lw=1.05)
    arrow((1.18, .27), (1.18, .50), orange)
    ax.text(3.90, .13, "execution log", ha="center", va="center", fontsize=12, color=orange,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    fig.savefig(ARCHITECTURE.with_suffix(".pdf"))
    fig.savefig(ARCHITECTURE.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main():
    setup_style()
    print(json.dumps(plot_traffic(), indent=2))
    plot_architecture()
    print("Created streamlined MPC and architecture PDF/PNG figures.")


if __name__ == "__main__":
    main()
