"""Readable tested certificate-frontier figure; never reinterpret partial bounds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
MAIN = RESULTS / "mfd_incident_uncertainty_frontier_results.json"
REFINE = RESULTS / "mfd_incident_uncertainty_frontier_refinement_results.json"
OUTPUT = RESULTS / "mfd_operating_frontiers"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(MAIN.read_text())
    refined = json.loads(REFINE.read_text())
    widths = [1., 5., 10., 25., 50., 100.]
    residuals = [60., 120., 300., 600., 1200.]
    lookup = {r["row_id"]: r for r in source["rows"]}
    state = [lookup[f"v{v:g}_r60_conservation_box"] for v in widths]
    residual = [lookup[f"v1_r{v:g}_conservation_box"] for v in residuals]
    ablation = [lookup[f"v10_r60_{metric}"] for metric in ("conservation_box", "class_box", "physical_box")]
    for row in state + residual:
        if row["accepted"]:
            assert row["certified_until_h"] == 1. and row["service_upper_vehicles"] <= 2850.
        else:
            assert row["service_upper_vehicles"] is None
            assert row["first_failure"]["kind"] == "continuous_physical_constraint_not_certified"
    figdata = dict(
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), MAIN, REFINE)},
        panels=dict(state_half_width_axis=state, arrival_residual_axis=residual),
        refinements=refined["refinements"], coordinate_ablation=ablation,
        ordinate="60 * certified_until_h: physical constraints verified only through this time",
        marker_meaning="Filled circle: entire one-hour plan admitted. Cross: certification stopped before one hour.",
        scope="First archived primary MPC plan; fixed forecast, fixed commands and 2850-vehicle deadline. Tested sufficient contracts, not necessary thresholds.")
    OUTPUT.with_name(OUTPUT.name + "_figuredata.json").write_text(json.dumps(figdata, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 12, "axes.titlesize": 13,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    teal, orange, purple, gray = "#087F8C", "#C45118", "#8057A6", "#8994A2"
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.0), sharey=True)
    fig.subplots_adjust(left=.082, right=.98, bottom=.285, top=.785, wspace=.18)
    for ax, rows, values, title, xlabel in zip(
        axes, (state, residual), (widths, residuals),
        ("(a) Initial-state estimation error", "(b) Arrival forecast residual"),
        ("Regional/class half-width (vehicles)", "Arrival residual half-width (vehicles/hour)")):
        durations = [60. * r["certified_until_h"] for r in rows]
        ax.plot(values, durations, color=gray, linewidth=1.1, zorder=1)
        for x, duration, row in zip(values, durations, rows):
            ax.scatter(x, duration, marker="o" if row["accepted"] else "x",
                       s=60 if row["accepted"] else 75,
                       linewidths=1.8, color=teal if row["accepted"] else orange, zorder=3)
        ax.axhline(60., color=teal, linewidth=.8, linestyle=(0, (4, 4)), alpha=.5)
        ax.set_xscale("log")
        ax.set_xticks(values)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.set_ylim(0., 66.)
        ax.set_yticks([0, 15, 30, 45, 60])
        ax.grid(axis="y", color="#DEE3EA", linewidth=.7)
        ax.set_title(title, loc="left", pad=13, fontweight="bold")
        ax.set_xlabel(xlabel, labelpad=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(gray)
        ax.spines["left"].set_color(gray)
    axes[0].set_ylabel("Verified physical-constraint duration (min)", labelpad=10)
    axes[0].text(.02, .80, "Arrival residual: 60 veh/h\nQueue error capped at 10 veh", transform=axes[0].transAxes,
                 fontsize=10.5, va="top", color="#374151", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
    axes[1].text(.02, .80, "State/queue error:\n1 vehicle half-width", transform=axes[1].transAxes,
                 fontsize=10.5, va="top", color="#374151", bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))

    for ax, width, residual_bound, baseline, ylim, yticks in (
        (axes[0], 25., 60., state[3], (32., 35.), [32., 33., 34., 35.]),
        (axes[1], 1., 300., residual[2], (42.25, 43.05), [42.25, 42.5, 42.75, 43.])):
        finer = sorted([r for r in refined["refinements"] if r["state_half_width_vehicles"] == width
                        and r["residual_vehicles_per_hour"] == residual_bound],
                       key=lambda r: r["maximum_step_seconds"], reverse=True)
        steps = [15.] + [r["maximum_step_seconds"] for r in finer]
        times = [60. * baseline["certified_until_h"]] + [60. * r["certified_until_h"] for r in finer]
        small = ax.inset_axes([.07, .11, .43, .27])
        small.plot(range(3), times, color=purple, marker="D", markersize=3.5, linewidth=1.1)
        small.set_xticks(range(3), [f"{s:g}" for s in steps], fontsize=9)
        small.set_ylim(ylim)
        small.set_yticks(yticks)
        small.tick_params(axis="y", labelsize=8.5)
        small.set_xlabel("Verification step (s)", fontsize=9, labelpad=2)
        small.set_title(f"Refinement at {width:g} veh / {residual_bound:g} veh/h", fontsize=9.5, pad=4)
        small.grid(axis="y", color="#E8E4EE", linewidth=.6)
        for spine in ("top", "right"):
            small.spines[spine].set_visible(False)
        small.spines["bottom"].set_color(gray)
        small.spines["left"].set_color(gray)

    fig.suptitle("One-hour certification range of the fixed MPC plan", y=.975, fontsize=18, fontweight="bold")
    fig.text(.5, .915, "Conservation-coordinate certificates with a 2,850-vehicle regional deadline", ha="center", fontsize=12, color="#374151")
    fig.legend(handles=[
        Line2D([], [], marker="o", color=teal, linewidth=0, markersize=7, label="One-hour plan admitted"),
        Line2D([], [], marker="x", color=orange, linewidth=0, markersize=7, markeredgewidth=1.8, label="Stopped at a physical lower-support check"),
        Line2D([], [], marker="D", color=purple, linewidth=1.1, markersize=4, label="Step refinement (insets)")],
        loc="upper center", bbox_to_anchor=(.52, .89), ncol=3, frameon=False, fontsize=10.5,
        columnspacing=1.8, handlelength=1.4)

    fig.text(.085, .183, "Coordinate comparison at 10 vehicles and 60 veh/h", fontsize=11.5, fontweight="bold", color="#374151", va="top")
    fig.text(.085, .143,
             "One-hour regional bound: conservation 1,844.58 veh; class-only 2,387.53 veh.\n"
             "Raw physical coordinates stop before the deadline; no one-hour bound is reported.",
             fontsize=10.7, color="#374151", linespacing=1.4, va="top")
    fig.text(.085, .057,
             "Markers are tested contracts. Partial durations apply only up to the displayed time;\n"
             "failed certificates and selected numerical adverse replays are distinct evidence.",
             fontsize=10.2, color="#596579", linespacing=1.35, va="top")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.14,
                metadata={"Title": "Fixed MPC plan: tested uncertainty certification range", "Author": ""})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=.14)
    plt.close(fig)
    print(OUTPUT.with_suffix(".pdf"))
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
