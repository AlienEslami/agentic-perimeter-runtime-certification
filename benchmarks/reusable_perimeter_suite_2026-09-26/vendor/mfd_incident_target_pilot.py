#!/usr/bin/env python3
"""Fixed-demand operating-target pilot for the existing six-state cubic MFD.

This is a numerical feasibility/stability audit, not an incident-capacity model,
queue model, SOS proof, or authorization to switch between target-centered tubes.
The original benchmark files and physical demand are never changed. In the main
library BOTH external gates are also fixed at their original feedforward values,
so changed targets cannot obtain an apparent benefit by rejecting more demand.

Run: python reproducibility/mfd_incident_target_pilot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, solve_continuous_are, solve_continuous_lyapunov
from scipy.optimize import brentq, minimize

import mfd_cubic_pilot as p


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "mfd_incident_target_pilot_results.json"
BASELINE_RECORD = p.RESULTS_DIR / "mfd_cubic_pilot_results.json"
BASELINE_SOS = p.RESULTS_DIR / "mfd_cubic_sos_pilot_results.json"
EXTERNAL_COMMAND = p.U_STAR[2:].copy()
FIXED_DEMAND = p.DEMAND.copy()
TARGETS = (
    ("nominal", 30.0, 30.0),
    ("region_1_reserve_200", 28.0, 31.0),
    ("region_1_reserve_400", 26.0, 32.0),
    ("region_1_reserve_600", 24.0, 33.0),
    ("region_1_only_reserve_500", 25.0, 30.0),
    ("region_2_reserve_400", 32.0, 26.0),
)


def speed(total: float) -> float:
    return float(p.MFD_A * total**2 + p.MFD_B * total + p.MFD_C)


def equilibrium_for_totals(n1: float, n2: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve all class states and transfer gates for fixed demand/external gates.

    At equilibrium q12*u12=d12, q10*u12=d10,
    q21*u21=d21+Q01*u02, q20*u20=d10+d20. Substitution into
    the n11/n22 balances gives the formulas below. No demand is backsolved.
    """
    d11, d12, d10, d22, d21, d20 = FIXED_DEMAND
    u02, u20 = EXTERNAL_COMMAND
    v1, v2 = speed(n1), speed(n2)
    if min(v1, v2, n1, n2) <= 0.0:
        raise ValueError("Positive regional accumulation and speed required")
    denominator_1 = p.mfd_outflow(n1) - d11 - d21 - p.Q01 * u02
    denominator_2 = (
        p.mfd_outflow(n2) - d22 - d12 - p.Q02 * u02 - (d10 + d20) / u20
    )
    if min(denominator_1, denominator_2) <= 0.0:
        raise ValueError("No positive transfer command at the requested totals")
    u12 = (d12 + d10) / denominator_1
    u21 = (d21 + p.Q01 * u02) / denominator_2
    state = np.array([
        n1, n2,
        (d11 + d21 + p.Q01 * u02) / v1,
        (d22 + d12 + p.Q02 * u02) / v2,
        d10 / (v1 * u12),
        (d10 + d20) / (v2 * u20),
    ])
    return state, np.array([u12, u21, u02, u20])


def physical_margins(state: np.ndarray, command: np.ndarray) -> dict[str, float]:
    n1, n2, n11, n22, n10, n20 = state
    margins = {f"{name}_lower": float(value) for name, value in zip(p.STATE_ORDER, state)}
    margins.update({"n1_upper": p.JAM_ACCUMULATION - n1,
                    "n2_upper": p.JAM_ACCUMULATION - n2,
                    "n12_lower": n1 - n11 - n10,
                    "n21_lower": n2 - n22 - n20})
    for name, value in zip(p.INPUT_ORDER, command):
        margins[f"{name}_lower"] = float(value)
        margins[f"{name}_upper"] = float(1.0 - value)
    margins["u21_minus_u12_directional"] = float(p.DIRECTIONAL_LIMIT - abs(command[1] - command[0]))
    margins["u20_minus_u02_directional"] = float(p.DIRECTIONAL_LIMIT - abs(command[3] - command[2]))
    return margins


def target_constraint_level(metric: np.ndarray, gain: np.ndarray,
                            state: np.ndarray, command: np.ndarray) -> dict[str, Any]:
    """Exact ellipsoid support test, with target-specific asymmetric margins."""
    inverse = np.linalg.inv(metric)
    directions = [np.eye(6)[index] for index in range(6)]
    names = [f"{name}_lower" for name in p.STATE_ORDER]
    offsets = list(state)
    directions += [-np.eye(6)[0], -np.eye(6)[1],
                   np.array([1, 0, -1, 0, -1, 0]), np.array([0, 1, 0, -1, 0, -1])]
    names += ["n1_upper", "n2_upper", "n12_lower", "n21_lower"]
    offsets += [p.JAM_ACCUMULATION - state[0], p.JAM_ACCUMULATION - state[1],
                state[0] - state[2] - state[4], state[1] - state[3] - state[5]]
    for index, row in enumerate(gain):
        directions += [row, -row]
        names += [f"{p.INPUT_ORDER[index]}_lower", f"{p.INPUT_ORDER[index]}_upper"]
        offsets += [command[index], 1.0 - command[index]]
    for left, right in ((1, 0), (3, 2)):
        delta = command[left] - command[right]
        row = gain[left] - gain[right]
        directions += [row, -row]
        names += [f"direction_{left}_{right}_lower", f"direction_{left}_{right}_upper"]
        offsets += [p.DIRECTIONAL_LIMIT + delta, p.DIRECTIONAL_LIMIT - delta]
    candidates = [(float(offset**2 / (row @ inverse @ row)), name)
                  for row, offset, name in zip(directions, offsets, names)
                  if row @ inverse @ row > 1.0e-24]
    assert min(offsets) > 0.0
    level, binding = min(candidates)
    return {"level": level, "binding_constraint": binding}


def closed_loop_polynomials(state: np.ndarray, command: np.ndarray,
                            gain: np.ndarray) -> list[p.Polynomial]:
    """Target-centered polynomial construction without changing pilot globals."""
    add, sub, mul, scale, const = p.poly_add, p.poly_sub, p.poly_mul, p.poly_scale, p.constant
    n1, n2, n11, n22, n10, n20 = [p.affine(state[i], np.eye(6)[i]) for i in range(6)]
    n12, n21 = sub(sub(n1, n11), n10), sub(sub(n2, n22), n20)
    u12, u21, u02, u20 = [p.affine(command[i], gain[i]) for i in range(4)]
    q11, q12, q10 = [p.class_flow_polynomial(value, n1) for value in (n11, n12, n10)]
    q22, q21, q20 = [p.class_flow_polynomial(value, n2) for value in (n22, n21, n20)]
    d11, d12, d10, d22, d21, d20 = FIXED_DEMAND
    dn11 = add(sub(mul(q21, u21), q11), const(d11))
    dn22 = add(add(sub(mul(q12, u12), q22), scale(u02, p.Q02)), const(d22))
    dn10 = sub(const(d10), mul(q10, u12))
    dn20 = add(sub(mul(q10, u12), mul(q20, u20)), const(d20))
    dn1 = add(sub(sub(mul(q21, u21), q11), mul(add(q12, q10), u12)), const(d11 + d12 + d10))
    dn2 = add(add(sub(sub(sub(mul(add(q12, q10), u12), q22), mul(q21, u21)),
                        mul(q20, u20)), scale(u02, p.Q01 + p.Q02)), const(d22 + d21 + d20))
    return [dn1, dn2, dn11, dn22, dn10, dn20]


def local_polynomial_audit(state: np.ndarray, command: np.ndarray,
                          gain: np.ndarray, matrix: np.ndarray) -> dict[str, Any]:
    """Conservative coefficient-norm bound, evaluated in floating point.

    With Acl.T P + P Acl + rate P=-I and remainder coefficient sums C_d,
    dot(V)+rate V <= -||e||^2 + 2 sqrt(lambda_max(P))*sum(C_d ||e||^(d+1)).
    Choose the radius so the last term is <=0.5||e||^2. This is not an
    interval-arithmetic/formal certificate and is not a switched-target proof.
    """
    abscissa = p.spectral_abscissa(matrix)
    rate = min(2.0, -0.7 * abscissa)
    metric = solve_continuous_lyapunov((matrix + rate * np.eye(6) / 2).T, -np.eye(6))
    values, directions = np.linalg.eigh(metric)
    assert min(values) > 0.0
    sqrt_metric = directions @ np.diag(np.sqrt(values)) @ directions.T
    vectors: dict[tuple[int, ...], np.ndarray] = {}
    recovered_linear = np.zeros((6, 6))
    constant_residual = np.zeros(6)
    for output, polynomial in enumerate(closed_loop_polynomials(state, command, gain)):
        constant_residual[output] = polynomial.get(p.ZERO_EXPONENT, 0.0)
        for index in range(6):
            exponent = tuple(int(i == index) for i in range(6))
            recovered_linear[output, index] = polynomial.get(exponent, 0.0)
        for exponent, coefficient in polynomial.items():
            if sum(exponent) >= 2:
                vectors.setdefault(exponent, np.zeros(6))[output] = coefficient
    constants = {degree: float(sum(np.linalg.norm(sqrt_metric @ vector)
                                 for exponent, vector in vectors.items() if sum(exponent) == degree))
                 for degree in (2, 3, 4)}
    radius = brentq(lambda r: 2 * math.sqrt(max(values)) * sum(constants[d] * r**(d - 1)
                                                                    for d in (2, 3, 4)) - 0.5,
                    0.0, 100.0, xtol=1.0e-14)
    constraints = target_constraint_level(metric, gain, state, command)
    level = min(float(min(values) * radius**2), constraints["level"])
    lyapunov_residual = matrix.T @ metric + metric @ matrix + rate * metric + np.eye(6)
    assert np.max(abs(constant_residual)) < 1.0e-9
    assert np.max(abs(recovered_linear - matrix)) < 1.0e-8
    assert np.max(abs(lyapunov_residual)) < 1.0e-8
    return {"status": "floating_point_analytic_bound_not_formal_certificate",
            "rate_per_hour": rate, "metric": metric, "level": level,
            "guaranteed_euclidean_ball_radius_vehicles": 100 * math.sqrt(level / max(values)),
            "maximum_ellipsoid_euclidean_radius_vehicles": 100 * math.sqrt(level / min(values)),
            "nonlinear_coefficient_norm_sums": constants,
            "constraint_only_level": constraints,
            "constant_residual_inf": float(max(abs(constant_residual))),
            "linear_reconstruction_residual_inf": float(np.max(abs(recovered_linear - matrix))),
            "lyapunov_equation_residual_inf": float(np.max(abs(lyapunov_residual)))}


def step_response_audit(state: np.ndarray, command: np.ndarray, gain: np.ndarray) -> dict[str, Any]:
    """Diagnostic unguarded target step, NOT a certified target transition."""
    solution = solve_ivp(lambda _t, x: p.mfd_rhs(x, command + gain @ (x - state), FIXED_DEMAND),
                         (0.0, 3.0), p.X_STAR, rtol=1.0e-9, atol=1.0e-11,
                         max_step=1.0 / 600.0, t_eval=np.linspace(0.0, 3.0, 1801))
    commands = np.array([command + gain @ (x - state) for x in solution.y.T])
    margins = [physical_margins(x, u) for x, u in zip(solution.y.T, commands)]
    minimum_margin = min(min(item.values()) for item in margins)
    return {"status": "sampled_nonlinear_diagnostic_only", "solver_success": bool(solution.success),
            "horizon_hours": float(solution.t[-1]),
            "initial_raw_command": commands[0],
            "minimum_sampled_physical_margin": minimum_margin,
            "sampled_constraints_satisfied": bool(minimum_margin >= -1.0e-9),
            "final_target_error_vehicles": float(100 * np.linalg.norm(solution.y[:, -1] - state)),
            "maximum_target_error_vehicles": float(100 * max(np.linalg.norm(solution.y.T - state, axis=1)))}


def feasible_region() -> dict[str, Any]:
    critical = float(min(np.roots([3 * p.MFD_A, 2 * p.MFD_B, p.MFD_C])))
    d11, d12, d10, d22, d21, d20 = FIXED_DEMAND
    u02, u20 = EXTERNAL_COMMAND
    thresholds = [d11 + d21 + p.Q01 * u02 + d12 + d10,
                  d22 + d12 + p.Q02 * u02 + (d10 + d20) / u20 + d21 + p.Q01 * u02]
    bounds = [[float(brentq(lambda n: p.mfd_outflow(n) - threshold, 0.0, critical)),
               float(brentq(lambda n: p.mfd_outflow(n) - threshold, critical, 99.0))]
              for threshold in thresholds]
    _, minimizing_commands = equilibrium_for_totals(critical, critical)
    worst_direction_difference = max(1.0 - minimizing_commands[0], 1.0 - minimizing_commands[1])
    checked = 0
    maximum_residual = 0.0
    minimum_margin = math.inf
    for n1 in np.linspace(*bounds[0], 41):
        for n2 in np.linspace(*bounds[1], 41):
            state, command = equilibrium_for_totals(n1, n2)
            maximum_residual = max(maximum_residual, float(np.max(abs(p.mfd_rhs(state, command, FIXED_DEMAND)))))
            minimum_margin = min(minimum_margin, min(physical_margins(state, command).values()))
            checked += 1
    assert maximum_residual < 1.0e-9 and minimum_margin >= -1.0e-8
    assert worst_direction_difference < p.DIRECTIONAL_LIMIT
    return {"scope": "fixed primitive demand, fixed external commands, original unmodified MFD",
            "derivation": "G(n1)>=threshold_1 and G(n2)>=threshold_2; u12/u21 then in (0,1]. Minimum transfer commands occur at the production maximum. Their worst pairwise difference is below the directional limit; all class flows are positive.",
            "critical_accumulation_units": critical, "production_thresholds_units_per_hour": thresholds,
            "full_physical_accumulation_intervals_units": bounds,
            "productive_branch_intervals_units": [[bound[0], critical] for bound in bounds],
            "minimum_transfer_commands": minimizing_commands[:2],
            "worst_possible_transfer_command_difference": float(worst_direction_difference),
            "numerical_grid_points_checked": checked,
            "maximum_grid_equilibrium_residual_inf": maximum_residual,
            "minimum_grid_physical_margin": minimum_margin,
            "boundary_warning": "u12/u21 equals 1 on lower/upper boundaries; these are feasible equilibria with no upper-gate feedback reserve, not an installed robust library."}


def projection_audit(region: dict[str, Any]) -> dict[str, Any]:
    """Constrained optimization projects an infeasible requested target."""
    requested = np.array([15.0, 34.0])
    critical = region["critical_accumulation_units"]
    bounds = [(interval[0], critical) for interval in region["full_physical_accumulation_intervals_units"]]
    result = minimize(lambda totals: float(np.sum((totals - requested)**2)), [20.0, 32.0],
                      method="SLSQP", bounds=bounds,
                      constraints=[{"type": "ineq", "fun": lambda totals: 0.95 - equilibrium_for_totals(*totals)[1][:2]}],
                      options={"ftol": 1.0e-12, "maxiter": 200})
    state, command = equilibrium_for_totals(*result.x)
    assert result.success and max(command[:2]) <= 0.95000001
    return {"requested_target_units": requested, "required_upper_gate_margin": 0.05,
            "productive_branch_required": True, "optimizer": "SLSQP",
            "success": bool(result.success), "message": str(result.message),
            "projected_target_units": result.x, "state": state, "command": command,
            "equilibrium_residual_inf": float(np.max(abs(p.mfd_rhs(state, command, FIXED_DEMAND)))),
            "interpretation": "Numerical projection, not a proof of global optimality."}


def main() -> None:
    baseline = json.loads(BASELINE_RECORD.read_text(encoding="utf-8"))
    sos = json.loads(BASELINE_SOS.read_text(encoding="utf-8"))
    selection = baseline["controller_search"]["operational_frontier"]["selected"]
    r_scales = [selection["mode_1_total_R_scale"], selection["mode_2_total_R_scale"]]
    original_metrics = [np.array(value) for value in baseline["certificate"]["matrices"]]
    library = []
    for name, n1, n2 in TARGETS:
        state, command = equilibrium_for_totals(n1, n2)
        residual = float(np.max(abs(p.mfd_rhs(state, command, FIXED_DEMAND))))
        margins = physical_margins(state, command)
        assert residual < 1.0e-10 and min(margins.values()) > 0.0
        a = p.complex_step_jacobian(lambda x: p.mfd_rhs(x, command, FIXED_DEMAND), state)
        b = p.complex_step_jacobian(lambda u: p.mfd_rhs(state, u, FIXED_DEMAND), command)
        finite_a = p.centered_jacobian(lambda x: p.mfd_rhs(x, command, FIXED_DEMAND), state)
        finite_b = p.centered_jacobian(lambda u: p.mfd_rhs(state, u, FIXED_DEMAND), command)
        controllers = []
        for mode in range(2):
            r = np.diag(p.R_PRIORITY[mode] * r_scales[mode])
            riccati = solve_continuous_are(a, b, np.diag(p.Q_PRIORITY[mode]), r)
            gain = -np.linalg.solve(r, b.T @ riccati)
            matrix = a + b @ gain
            assert p.spectral_abscissa(matrix) < -1.0e-6
            controllers.append({"mode": mode + 1, "gain": gain,
                                "closed_loop_spectral_abscissa_per_hour": p.spectral_abscissa(matrix),
                                "local_bound": local_polynomial_audit(state, command, gain, matrix),
                                "direct_step_from_original_equilibrium": step_response_audit(state, command, gain)})
        shift = state - p.X_STAR
        library.append({"name": name, "state": state, "command": command,
                        "region_1_reserve_relative_to_nominal_vehicles": 100 * (30.0 - n1),
                        "region_2_reserve_relative_to_nominal_vehicles": 100 * (30.0 - n2),
                        "primitive_demand_units_per_hour": FIXED_DEMAND,
                        "external_admitted_rates_units_per_hour": np.array([p.Q01, p.Q02]) * command[2],
                        "equilibrium_residual_inf": residual, "physical_margins": margins,
                        "jacobian_cross_check_A_inf": float(np.max(abs(a - finite_a))),
                        "jacobian_cross_check_B_inf": float(np.max(abs(b - finite_b))),
                        "original_SOS_tube_center_occupancies": [float(shift @ metric @ shift / sos["common_level"])
                                                                  for metric in original_metrics],
                        "equilibrium_transportation_rates": p.transportation_rates(state, command),
                        "controllers": controllers})
    region = feasible_region()
    assert np.array_equal(p.DEMAND, FIXED_DEMAND)
    assert np.array_equal(p.X_STAR, np.array([30., 30., 15., 15., 5., 5.]))
    result = {"status": "fixed_demand_numerical_target_library_pilot",
              "units": {"accumulation": "100 vehicles", "time": "hour", "command": "dimensionless"},
              "fixed_model": {"state_order": p.STATE_ORDER, "input_order": p.INPUT_ORDER,
                              "primitive_demand_units_per_hour": FIXED_DEMAND,
                              "Q01": p.Q01, "Q02": p.Q02, "external_command": EXTERNAL_COMMAND,
                              "MFD_coefficients": [p.MFD_A, p.MFD_B, p.MFD_C]},
              "feasible_region": region, "constrained_target_projection": projection_audit(region),
              "library": library,
              "limitations": [
                  "Target reserves are lower nominal accumulations under the ORIGINAL MFD, not verified equilibria under incident-reduced capacity.",
                  "Primitive demand and BOTH external feedforward gates are identical at every equilibrium. No demand is recomputed per target.",
                  "LQR gains permit transient external-gate adjustments; direct-step diagnostics therefore do not imply identical transient admission or queue costs.",
                  "Endpoint local bounds use floating-point polynomial/linear-algebra calculations, not validated intervals or new SOS solves.",
                  "Distinct-target switching changes the equilibrium center; the original zero-reset/common-equilibrium dwell theorem cannot be reused unchanged.",
                  "Sampled nonlinear simulations do not certify all trajectories or intersample safety. A failed direct step is retained as a diagnostic, not hidden.",
                  "There are no external queues or rejected-demand accounting in this six-state model. Do not infer welfare gains from reduced in-network vehicle-hours alone."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(p.json_ready(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(p.json_ready({"output": str(OUTPUT), "productive_branch": region["productive_branch_intervals_units"],
          "library": [{"name": item["name"], "totals": item["state"][:2], "command": item["command"],
                       "abscissae": [mode["closed_loop_spectral_abscissa_per_hour"] for mode in item["controllers"]],
                       "step_safe_sampled": [mode["direct_step_from_original_equilibrium"]["sampled_constraints_satisfied"]
                                             for mode in item["controllers"]],
                       "old_tube_occupancies": item["original_SOS_tube_center_occupancies"]}
                      for item in library]}), indent=2))


if __name__ == "__main__":
    main()
