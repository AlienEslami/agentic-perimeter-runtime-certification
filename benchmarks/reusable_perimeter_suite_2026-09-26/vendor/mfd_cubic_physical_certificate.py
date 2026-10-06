#!/usr/bin/env python3
"""Translate the cubic-MFD Lyapunov tubes into physical traffic guarantees.

For an installed ellipsoid ``e.T @ P @ e <= level``, the support function in
any linear direction ``c`` is exact:

    max |c.T @ e| = sqrt(level * c.T @ inv(P) @ c).

This script applies that identity to regional accumulation, imbalance,
network-total accumulation, perimeter commands, and the declared physical
constraints.  It then bounds the cubic regional production over the resulting
certified accumulation intervals.  The output is intended to support an
MFD-specific proposition in the Part B manuscript; it does not alter the
controller, certificate, or simulations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import mfd_cubic_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
PILOT_JSON = RESULTS_DIR / "mfd_cubic_pilot_results.json"
SOS_JSON = RESULTS_DIR / "mfd_cubic_sos_pilot_results.json"
OUTPUT_JSON = RESULTS_DIR / "mfd_cubic_physical_certificate_results.json"


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


def support_radius(inverse_metric: np.ndarray, level: float, row: np.ndarray) -> float:
    """Exact support radius of an origin-centered ellipsoid in ``row``."""
    value = float(np.asarray(row) @ inverse_metric @ np.asarray(row))
    return math.sqrt(max(0.0, level * value))


def interval_extrema(function, lower: float, upper: float, stationary: list[float]) -> tuple[float, float]:
    points = [lower, upper]
    points.extend(root for root in stationary if lower <= root <= upper)
    values = [float(function(point)) for point in points]
    return min(values), max(values)


def critical_accumulation() -> float:
    roots = np.roots([3.0 * pilot.MFD_A, 2.0 * pilot.MFD_B, pilot.MFD_C])
    candidates = [
        float(np.real(root))
        for root in roots
        if abs(float(np.imag(root))) <= 1.0e-10
        and 0.0 < float(np.real(root)) < pilot.JAM_ACCUMULATION
        and 6.0 * pilot.MFD_A * float(np.real(root)) + 2.0 * pilot.MFD_B < 0.0
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one production maximum, obtained {candidates}")
    return candidates[0]


def tube_physical_record(
    mode: int,
    level_name: str,
    level: float,
    metric: np.ndarray,
    gain: np.ndarray,
    critical: float,
) -> dict[str, Any]:
    inverse = np.linalg.inv(metric)
    unit = pilot.ACCUMULATION_UNIT_VEHICLES

    coordinate_rows = {
        name: np.eye(pilot.DIMENSION)[index]
        for index, name in enumerate(pilot.STATE_ORDER)
    }
    coordinate_bounds_units = {
        name: support_radius(inverse, level, row)
        for name, row in coordinate_rows.items()
    }
    coordinate_bounds_vehicles = {
        name: unit * value for name, value in coordinate_bounds_units.items()
    }

    imbalance_row = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    total_row = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    imbalance_bound_units = support_radius(inverse, level, imbalance_row)
    total_bound_units = support_radius(inverse, level, total_row)

    n_intervals = []
    derivative_roots = [
        float(np.real(root))
        for root in np.roots([3.0 * pilot.MFD_A, 2.0 * pilot.MFD_B, pilot.MFD_C])
        if abs(float(np.imag(root))) <= 1.0e-10
    ]
    production_ranges = []
    for index, name in enumerate(("n1", "n2")):
        radius = coordinate_bounds_units[name]
        lower = max(0.0, float(pilot.X_STAR[index] - radius))
        upper = min(pilot.JAM_ACCUMULATION, float(pilot.X_STAR[index] + radius))
        n_intervals.append([lower, upper])
        production_ranges.append(
            interval_extrema(pilot.mfd_outflow, lower, upper, derivative_roots)
        )

    production_min = sum(bounds[0] for bounds in production_ranges)
    production_max = sum(bounds[1] for bounds in production_ranges)
    production_target = 2.0 * float(pilot.mfd_outflow(pilot.X_STAR[0]))
    production_conversion = unit / 3600.0

    command_support = [support_radius(inverse, level, row) for row in gain]
    command_ranges = [
        [float(pilot.U_STAR[index] - radius), float(pilot.U_STAR[index] + radius)]
        for index, radius in enumerate(command_support)
    ]
    direction_rows = {
        "u21_minus_u12": gain[1] - gain[0],
        "u20_minus_u02": gain[3] - gain[2],
    }
    directional_support = {
        name: support_radius(inverse, level, row)
        for name, row in direction_rows.items()
    }

    state_names, state_rows, state_margins = pilot.state_constraint_rows()
    state_constraint_margins = {
        name: unit * (margin - support_radius(inverse, level, row))
        for name, row, margin in zip(state_names, state_rows, state_margins)
    }
    input_box_margins = {
        pilot.INPUT_ORDER[index]: 0.5 - radius
        for index, radius in enumerate(command_support)
    }
    input_directional_margins = {
        name: pilot.DIRECTIONAL_LIMIT - radius
        for name, radius in directional_support.items()
    }

    upper_n = max(interval_[1] for interval_ in n_intervals)
    return {
        "mode": mode + 1,
        "level_name": level_name,
        "level": level,
        "coordinate_absolute_bounds_vehicles": coordinate_bounds_vehicles,
        "regional_imbalance_absolute_bound_vehicles": unit * imbalance_bound_units,
        "network_total_deviation_absolute_bound_vehicles": unit * total_bound_units,
        "absolute_network_vehicle_hour_deviation_bound_per_hour": unit * total_bound_units,
        "regional_accumulation_intervals_vehicles": [
            [unit * lower, unit * upper] for lower, upper in n_intervals
        ],
        "minimum_margin_to_critical_accumulation_vehicles": unit * (critical - upper_n),
        "minimum_margin_to_jam_accumulation_vehicles": unit * (pilot.JAM_ACCUMULATION - upper_n),
        "tube_remains_on_productive_branch": bool(upper_n <= critical + 1.0e-12),
        "network_production_target_vehicles_per_second": production_conversion * production_target,
        "network_production_interval_vehicles_per_second": [
            production_conversion * production_min,
            production_conversion * production_max,
        ],
        "maximum_network_production_shortfall_vehicles_per_second": production_conversion
        * max(0.0, production_target - production_min),
        "guaranteed_minimum_network_production_fraction_of_target": production_min
        / production_target,
        "perimeter_command_intervals": {
            name: command_ranges[index]
            for index, name in enumerate(pilot.INPUT_ORDER)
        },
        "perimeter_direction_absolute_bounds": directional_support,
        "minimum_state_constraint_margin_vehicles": min(state_constraint_margins.values()),
        "state_constraint_margins_vehicles": state_constraint_margins,
        "minimum_input_box_margin": min(input_box_margins.values()),
        "input_box_margins": input_box_margins,
        "minimum_input_directional_margin": min(input_directional_margins.values()),
        "input_directional_margins": input_directional_margins,
    }


def main() -> None:
    if not PILOT_JSON.exists() or not SOS_JSON.exists():
        raise SystemExit("Run the cubic model and SOS pilot scripts first")

    pilot_record = json.loads(PILOT_JSON.read_text(encoding="utf-8"))
    sos_record = json.loads(SOS_JSON.read_text(encoding="utf-8"))
    gains = [
        np.asarray(value, dtype=float)
        for value in pilot_record["controller_search"]["selected_gains"]
    ]
    metrics = [
        np.asarray(value, dtype=float)
        for value in pilot_record["certificate"]["matrices"]
    ]
    common_level = float(sos_record["common_level"])
    individual_levels = [
        float(item["level"]) for item in sos_record["sos_certificates"]
    ]
    critical = critical_accumulation()

    records = []
    for mode, (metric, gain, individual) in enumerate(
        zip(metrics, gains, individual_levels)
    ):
        records.append(
            tube_physical_record(
                mode, "installed_common", common_level, metric, gain, critical
            )
        )
        if individual > common_level + 1.0e-12:
            records.append(
                tube_physical_record(
                    mode, "modewise_maximum", individual, metric, gain, critical
                )
            )

    common_records = [
        record for record in records if record["level_name"] == "installed_common"
    ]
    common_guarantee = {
        "regional_imbalance_absolute_bound_vehicles": max(
            record["regional_imbalance_absolute_bound_vehicles"]
            for record in common_records
        ),
        "network_total_deviation_absolute_bound_vehicles": max(
            record["network_total_deviation_absolute_bound_vehicles"]
            for record in common_records
        ),
        "minimum_margin_to_critical_accumulation_vehicles": min(
            record["minimum_margin_to_critical_accumulation_vehicles"]
            for record in common_records
        ),
        "minimum_margin_to_jam_accumulation_vehicles": min(
            record["minimum_margin_to_jam_accumulation_vehicles"]
            for record in common_records
        ),
        "guaranteed_minimum_network_production_fraction_of_target": min(
            record["guaranteed_minimum_network_production_fraction_of_target"]
            for record in common_records
        ),
        "maximum_network_production_shortfall_vehicles_per_second": max(
            record["maximum_network_production_shortfall_vehicles_per_second"]
            for record in common_records
        ),
        "minimum_state_constraint_margin_vehicles": min(
            record["minimum_state_constraint_margin_vehicles"]
            for record in common_records
        ),
        "minimum_input_box_margin": min(
            record["minimum_input_box_margin"] for record in common_records
        ),
        "minimum_input_directional_margin": min(
            record["minimum_input_directional_margin"] for record in common_records
        ),
    }

    output = {
        "scope": "physical interpretation of the installed literature-cubic MFD Lyapunov tubes",
        "identity": "max_abs_cTe = sqrt(level * c.T * inv(P) * c)",
        "units": {
            "state": "100 vehicles",
            "time": "hours",
            "production_output": "vehicles per second",
            "perimeter_command": "normalized dimensionless command",
        },
        "critical_accumulation_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES * critical,
        "jam_accumulation_vehicles": pilot.ACCUMULATION_UNIT_VEHICLES
        * pilot.JAM_ACCUMULATION,
        "target_accumulation_vehicles_per_region": pilot.ACCUMULATION_UNIT_VEHICLES
        * pilot.X_STAR[0],
        "installed_common_level": common_level,
        "installed_common_library_guarantee": common_guarantee,
        "records": records,
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_ready(output), indent=2))


if __name__ == "__main__":
    main()
