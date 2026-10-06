#!/usr/bin/env python3
"""Conceptual and certificate-geometry figures for the MFD manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility" / "results"


def box(axis, xy, width, height, text, color, linewidth=1.2, fontsize=8):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor="#26333d", linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text,
              ha="center", va="center", fontsize=fontsize)


def arrow(axis, start, end, color="#26333d", style="-|>", linewidth=1.2):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                  mutation_scale=11, linewidth=linewidth,
                                  color=color))


def authority_figure():
    figure, axis = plt.subplots(figsize=(7.4, 3.6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 5.5)
    axis.axis("off")
    box(axis, (0.25, 2.0), 1.75, 1.0,
        "AI agent\ninformation state\n+ proposal policy", "#d9e8f5", fontsize=7)
    box(axis, (3.0, 1.75), 2.1, 1.5, "Runtime supervisor\nlanding + constraints\nprofile + binding", "#f8e2ae", 1.8)
    box(axis, (6.25, 2.0), 1.45, 1.0, "Executed\ncontroller", "#d9efdc")
    box(axis, (8.35, 2.0), 1.35, 1.0, "Two-region\nMFD plant", "#d9efdc")
    box(axis, (3.0, 0.15), 2.25, 0.85, "Authorized certificate library\nflow + transition records", "#eadff2", fontsize=7)
    box(axis, (0.25, 0.15), 1.75, 0.85, "Trusted construction\nverification + approval", "#eadff2", fontsize=7)
    arrow(axis, (2.0, 2.5), (3.0, 2.5))
    arrow(axis, (5.1, 2.5), (6.25, 2.5), linewidth=1.8)
    arrow(axis, (7.7, 2.5), (8.35, 2.5))
    arrow(axis, (9.0, 2.0), (9.0, 1.45), style="-|>")
    arrow(axis, (8.85, 1.45), (4.05, 1.45), style="-|>")
    axis.text(6.45, 1.58, "measured MFD state", ha="center", fontsize=7,
              color="#3d7b54")
    arrow(axis, (4.05, 1.0), (4.05, 1.75))
    arrow(axis, (2.0, 0.58), (3.0, 0.58))
    arrow(axis, (4.05, 3.25), (4.05, 3.85), style="-")
    arrow(axis, (4.05, 3.85), (1.10, 3.85), style="-")
    arrow(axis, (1.10, 3.85), (1.10, 3.0))
    axis.text(2.58, 3.98, "execution + audit feedback", ha="center",
              fontsize=7, color="#315f8a")
    axis.plot([2.55, 2.55], [1.15, 4.65], color="#a23b3b", linewidth=3.0)
    axis.text(2.45, 4.78, "certified action interface", ha="center",
              va="bottom", color="#8c2f2f", fontsize=8, fontweight="bold")
    axis.text(1.1, 4.30, "agent planning + proposals", ha="center", fontsize=8, color="#315f8a")
    axis.text(6.4, 4.30, "state-dependent certified execution", ha="center", fontsize=8, color="#3d7b54")
    figure.tight_layout()
    figure.savefig(RESULTS / "mfd_authority_architecture.pdf", bbox_inches="tight")
    figure.savefig(RESULTS / "mfd_authority_architecture.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def radius_figure():
    certified = 3.9556337543179643
    constraint = 47.78124390568455
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].barh(["nonlinear-validity", "constraint-admissible"],
                 [certified, constraint], color=["#35618f", "#b7c7d8"])
    axes[0].set_xlabel("verified error-space ball radius (vehicles)")
    axes[0].grid(axis="x", alpha=0.25)
    for index, value in enumerate((certified, constraint)):
        axes[0].text(value + 0.8, index, f"{value:.2f}", va="center", fontsize=8)

    scale = 1.0 / constraint
    outer = Circle((0, 0), 1.0, facecolor="#dfe8f1", edgecolor="#61778b", linewidth=1.2)
    inner = Circle((0, 0), certified * scale, facecolor="#35618f", edgecolor="#244868", linewidth=1.0)
    axes[1].add_patch(outer)
    axes[1].add_patch(inner)
    axes[1].set_xlim(-1.1, 1.1)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    axes[1].annotate(
        "nonlinear-validity ball (3.96 vehicles)",
        xy=(certified * scale * 0.7, certified * scale * 0.7),
        xytext=(0.22, -0.28), fontsize=7, color="#244868",
        arrowprops={"arrowstyle": "->", "color": "#244868", "linewidth": 0.8},
    )
    axes[1].text(0, 0.72, "constraint-admissible\nball", color="#394b5a",
                 ha="center", va="center", fontsize=8)
    axes[1].set_title("same declared six-dimensional error coordinates", fontsize=9)
    figure.tight_layout()
    figure.savefig(RESULTS / "mfd_certificate_radius_geometry.pdf", bbox_inches="tight")
    figure.savefig(RESULTS / "mfd_certificate_radius_geometry.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    authority_figure()
    radius_figure()


if __name__ == "__main__":
    main()
