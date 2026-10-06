#!/usr/bin/env python3
"""Congestion-indexed audit of the installed cubic-MFD controller library.

The two selected feedback gains, proposal clock, Lyapunov construction rule,
state composition, command equilibrium, and cubic production law are frozen.
Only the symmetric regional equilibrium n* is varied.  At every operating
point the primitive demand vector is reconstructed to make that point an exact
equilibrium, the MFD Jacobians are recomputed, and the installed-gain closed
loops are re-certified when both remain Hurwitz.

This is deliberately an installed-library sensitivity audit rather than a
controller redesign: variation in the resulting sufficient dwell therefore
comes from the congestion-dependent MFD linearization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import mfd_cubic_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS / "mfd_cubic_pilot_results.json"
OUTPUT_JSON = RESULTS / "mfd_cubic_congestion_sweep_results.json"
FIGURE_PDF = RESULTS / "mfd_cubic_congestion_sweep.pdf"
FIGURE_PNG = RESULTS / "mfd_cubic_congestion_sweep.png"

RATIO_GRID = np.linspace(0.55, 0.98, 18)


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


def critical_accumulation() -> float:
    roots = np.sort(
        np.roots([3.0 * pilot.MFD_A, 2.0 * pilot.MFD_B, pilot.MFD_C])
    )
    physical = roots[np.isreal(roots)].real
    candidates = physical[(physical > 0.0) & (physical < pilot.JAM_ACCUMULATION)]
    if candidates.size == 0:
        raise RuntimeError("cubic MFD has no physical critical accumulation")
    return float(np.min(candidates))


def equilibrium_state(n_star: float) -> np.ndarray:
    # Freeze the class shares from the installed 3000-vehicle equilibrium.
    scale = n_star / float(pilot.X_STAR[0])
    return pilot.X_STAR * scale


def equilibrium_demand(state_star: np.ndarray) -> np.ndarray:
    n1, n2, n11, n22, n10, n20 = state_star
    u12, u21, u02, u20 = pilot.U_STAR
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    q11, q12, q10 = [pilot.class_flow(value, n1) for value in (n11, n12, n10)]
    q22, q21, q20 = [pilot.class_flow(value, n2) for value in (n22, n21, n20)]
    d10 = q10 * u12
    d20 = q20 * u20 - q10 * u12
    d11 = q11 - q21 * u21
    d22 = q22 - q12 * u12 - pilot.Q02 * u02
    d1 = q11 + (q12 + q10) * u12 - q21 * u21
    d2 = (
        q22 - (q12 + q10) * u12 + q21 * u21
        - (pilot.Q01 + pilot.Q02) * u02 + q20 * u20
    )
    demand = np.asarray([d11, d1 - d11 - d10, d10, d22, d2 - d22 - d20, d20])
    return demand


def rhs(
    state: np.ndarray,
    command: np.ndarray,
    demand: np.ndarray,
) -> np.ndarray:
    return pilot.mfd_rhs(state, command, demand=demand)


def record_at_ratio(
    ratio: float,
    critical: float,
    gains: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    n_star = ratio * critical
    state_star = equilibrium_state(n_star)
    demand = equilibrium_demand(state_star)
    residual = rhs(state_star, pilot.U_STAR, demand)
    a_matrix = pilot.complex_step_jacobian(
        lambda state: rhs(state, pilot.U_STAR, demand), state_star
    )
    b_matrix = pilot.complex_step_jacobian(
        lambda command: rhs(state_star, command, demand), pilot.U_STAR
    )
    closed_loop = tuple(a_matrix + b_matrix @ gain for gain in gains)
    abscissae = [pilot.spectral_abscissa(matrix) for matrix in closed_loop]
    stable = max(abscissae) < -1.0e-9
    base = {
        "congestion_ratio_nstar_over_ncr": ratio,
        "equilibrium_accumulation_units": n_star,
        "equilibrium_accumulation_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES * n_star,
        "critical_margin_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES * (critical - n_star),
        "mfd_slope_units_per_hour": (
            3.0 * pilot.MFD_A * n_star**2
            + 2.0 * pilot.MFD_B * n_star
            + pilot.MFD_C
        ),
        "minimum_primitive_demand_units_per_hour": float(np.min(demand)),
        "equilibrium_residual_inf": float(np.max(np.abs(residual))),
        "closed_loop_spectral_abscissae_per_hour": abscissae,
        "both_modes_hurwitz": stable,
    }
    if not stable:
        return base
    certificate = pilot.independent_certificate(closed_loop, pilot.PROPOSAL_INTERVAL)
    one_minute_radius = pilot.spectral_radius(
        expm(closed_loop[1] * pilot.PROPOSAL_INTERVAL)
        @ expm(closed_loop[0] * pilot.PROPOSAL_INTERVAL)
    )
    return {
        **base,
        "certificate_rate_per_hour": certificate["rate"],
        "transition_factor": certificate["transition_factor"],
        "sufficient_dwell_hours": certificate["dwell_time"],
        "proposal_grid_dwell_minutes": 60.0 * certificate["realized_dwell"],
        "one_minute_monodromy_radius": one_minute_radius,
    }


def main() -> None:
    installed = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    gains = tuple(
        np.asarray(value, dtype=float)
        for value in installed["controller_search"]["selected_gains"]
    )
    critical = critical_accumulation()
    installed_ratio = float(pilot.X_STAR[0] / critical)
    ratios = np.unique(np.append(RATIO_GRID, installed_ratio))
    records = [record_at_ratio(float(ratio), critical, gains) for ratio in ratios]
    installed_record = min(
        records,
        key=lambda item: abs(item["congestion_ratio_nstar_over_ncr"] - installed_ratio),
    )
    archived = installed["certificate"]
    if abs(installed_record["certificate_rate_per_hour"] - archived["rate"]) > 2.0e-10:
        raise RuntimeError("installed-point flow-rate reconstruction changed")
    if abs(installed_record["transition_factor"] - archived["transition_factor"]) > 2.0e-10:
        raise RuntimeError("installed-point transition-factor reconstruction changed")
    if max(item["equilibrium_residual_inf"] for item in records) > 2.0e-12:
        raise RuntimeError("congestion sweep contains a non-equilibrium operating point")

    feasible = [item for item in records if item["both_modes_hurwitz"]]
    output = {
        "scope": (
            "installed-gain congestion audit: gains, class shares, command equilibrium, "
            "proposal clock, cubic MFD, and Lyapunov synthesis rule are fixed"
        ),
        "critical_accumulation_units": critical,
        "critical_accumulation_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES * critical,
        "installed_ratio": installed_ratio,
        "installed_record": installed_record,
        "all_points_hurwitz": len(feasible) == len(records),
        "dwell_range_minutes_over_hurwitz_points": [
            min(item["proposal_grid_dwell_minutes"] for item in feasible),
            max(item["proposal_grid_dwell_minutes"] for item in feasible),
        ],
        "records": records,
    }
    OUTPUT_JSON.write_text(json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8")

    x = np.asarray([item["congestion_ratio_nstar_over_ncr"] for item in feasible])
    dwell = np.asarray([item["proposal_grid_dwell_minutes"] for item in feasible])
    rate = np.asarray([item["certificate_rate_per_hour"] for item in feasible])
    nu = np.asarray([item["transition_factor"] for item in feasible])
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.25))
    axes[0].plot(x, dwell, "o-", color="#35618f", linewidth=1.4, markersize=3.8)
    axes[0].axvline(installed_ratio, color="#8f4b45", linestyle="--", linewidth=1.0)
    axes[0].scatter(
        [installed_ratio], [installed_record["proposal_grid_dwell_minutes"]],
        color="#e2a63b", edgecolor="black", zorder=4, label="installed point",
    )
    axes[0].set_xlabel(r"congestion index $n^\star/n^{\rm cr}$")
    axes[0].set_ylabel("sufficient proposal-grid dwell (min)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(x, rate, "o-", color="#4a8d68", label=r"flow rate $\lambda$")
    secondary = axes[1].twinx()
    secondary.plot(x, nu, "s-", color="#8f4b45", label=r"transition factor $\bar\nu$")
    axes[1].axvline(installed_ratio, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel(r"congestion index $n^\star/n^{\rm cr}$")
    axes[1].set_ylabel(r"$\lambda$ (h$^{-1}$)", color="#4a8d68")
    secondary.set_ylabel(r"$\bar\nu$", color="#8f4b45")
    axes[1].tick_params(axis="y", colors="#4a8d68")
    secondary.tick_params(axis="y", colors="#8f4b45")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(json_ready({key: value for key, value in output.items() if key != "records"}), indent=2))


if __name__ == "__main__":
    main()
