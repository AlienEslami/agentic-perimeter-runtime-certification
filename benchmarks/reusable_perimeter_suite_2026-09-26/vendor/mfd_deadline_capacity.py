#!/usr/bin/env python3
"""Traffic-derived necessary deadline bound on the union of installed tubes.

For y_s = s*(n1-n2), s in {+1,-1}, this script bounds -dot(y_s)
over both common-level ellipsoids, every paired-box actuator command, and
every measurable rank-one demand redistribution |delta(t)| <= amplitude.
The maximization is favorable to service, so the resulting time lower bound
applies even to an optimistic actuator controller, not just the gain library.
It is a necessary condition, not a reachability or robust-feasibility claim.

The state domain is explicitly the UNION OF COMMON-LEVEL TUBES. The bound
does not constrain controllers whose physical trajectory leaves that union.
All certificate calculations use outward intervals. Random/LP tests are
independent diagnostics, not the justification of the analytic bounds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mpmath import iv, mp
from scipy.optimize import linprog

import mfd_cubic_pilot as plant
import mfd_cubic_nonstationary_study as study
import mfd_cubic_runtime_pilot as runtime
from mfd_deadline_certificates import I, imat, isum, symmetric_eigen_upper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_deadline_capacity_results.json"


def up(value) -> float:
    """Safely convert an mpmath interval's upper endpoint to binary64."""
    return float(np.nextafter(float(value.b), np.inf))


def down(value) -> float:
    return float(np.nextafter(float(value.a), -np.inf))


def inverse_of_interval_matrix(matrix: I) -> tuple[I, dict]:
    """Enclose A^-1 for every A in matrix using a Neumann residual bound."""
    midpoint = matrix.mid()
    approximate = np.linalg.inv(midpoint)
    residual = I(np.eye(6)) - imat(matrix, approximate)
    residual_norm = float(np.max(isum(I(residual.mag()[:, j]) for j in range(6)).hi))
    approximate_norm = float(np.max(isum(I(abs(approximate[:, j])) for j in range(6)).hi))
    if not 0 <= residual_norm < 1:
        raise ArithmeticError("The interval inverse residual did not certify invertibility")
    error = up(iv.mpf(approximate_norm) * iv.mpf(residual_norm) / (1-iv.mpf(residual_norm)))
    enclosed = I(np.nextafter(approximate-error, -np.inf), np.nextafter(approximate+error, np.inf))
    return enclosed, {"residual_infinity_norm_upper": residual_norm, "inverse_entry_error_upper": error}


def linear_range(coefficients: np.ndarray, inverse: I, level: float) -> I:
    """Exact ellipsoid support formula with interval-enclosed inverse data."""
    a = I(np.asarray(coefficients, dtype=float))
    variance = imat(a.lo.reshape(1, 6), imat(inverse, a))
    if float(variance.hi[0]) < 0:
        raise ArithmeticError("Negative support variance")
    radius = up(iv.sqrt(iv.mpf(level)*iv.mpf(max(0., float(variance.hi[0])))))
    mean = imat(a.lo.reshape(1, 6), I(plant.X_STAR))[0]
    return mean + I(-radius, radius)


def polynomial_speed(total: I) -> I:
    """Bound p(n)=G(n)/n using its verified decrease on the given interval."""
    slope = I(2.) * I(plant.MFD_A) * total + I(plant.MFD_B)
    if float(slope.hi) >= 0:
        raise ArithmeticError("p(n) is not certified decreasing on this accumulation interval")

    def value(n: float) -> I:
        return I(plant.MFD_A)*I(n)*I(n) + I(plant.MFD_B)*I(n) + I(plant.MFD_C)

    bound = I(value(float(total.hi)).lo, value(float(total.lo)).hi)
    if float(bound.lo) <= 0:
        raise ArithmeticError("The class-flow factor must be positive")
    return bound


def phi(a: float, b: float, difference_limit: float) -> float:
    """Diagnostic scalar form of max(av-bw) over the paired unit box.

    For fixed v, w=max(0,v-difference_limit) is optimal. The resulting
    piecewise-linear objective increases up to v=difference_limit; its
    remaining slope is a-b. The two candidate endpoints give this formula.
    """
    return difference_limit*a + (1-difference_limit)*max(a-b, 0.)


def phi_upper(a: I, b: I, difference_limit: float) -> I:
    """Phi is increasing in a and decreasing in b; use favorable endpoints."""
    if min(float(a.lo), float(b.lo)) < 0 or not 0 <= difference_limit <= 1:
        raise ValueError("Paired-gate support requires a,b >= 0 and 0 <= limit <= 1")
    ah, bl, gap = I(a.hi), I(b.lo), I(difference_limit)
    difference = ah-bl
    positive = I(max(0., float(difference.lo)), max(0., float(difference.hi)))
    return gap*ah + (I(1.)-gap)*positive


def interval_record(value: I) -> list[float]:
    return [float(value.lo), float(value.hi)]


def tube_capacity(matrix: np.ndarray, level: float, amplitude: float, index: int) -> dict:
    symmetric = (I(matrix) + I(matrix.T))*I(.5)
    minimum_eigenvalue = -symmetric_eigen_upper(-symmetric)
    if minimum_eigenvalue <= 0:
        raise ArithmeticError("The symmetric part of P is not certified positive definite")
    inverse, inverse_check = inverse_of_interval_matrix(symmetric)
    directions = {
        "n1": [1, 0, 0, 0, 0, 0], "n2": [0, 1, 0, 0, 0, 0],
        "n11": [0, 0, 1, 0, 0, 0], "n22": [0, 0, 0, 1, 0, 0],
        "n20": [0, 0, 0, 0, 0, 1],
        "n12_plus_n10": [1, 0, -1, 0, 0, 0],
        "n21": [0, 1, 0, -1, 0, -1],
    }
    ranges = {key: linear_range(np.asarray(a), inverse, level) for key, a in directions.items()}
    if any(float(value.lo) < 0 for value in ranges.values()):
        raise ArithmeticError("The stated positive-class domain is not certified")
    speeds = {"p1": polynomial_speed(ranges["n1"]), "p2": polynomial_speed(ranges["n2"])}
    flows = {
        "q11": ranges["n11"]*speeds["p1"],
        "q22": ranges["n22"]*speeds["p2"],
        "q20": ranges["n20"]*speeds["p2"],
        "q12_plus_q10": ranges["n12_plus_n10"]*speeds["p1"],
        "q21": ranges["n21"]*speeds["p2"],
    }
    d1 = isum(I(z) for z in plant.DEMAND[:3])
    d2 = isum(I(z) for z in plant.DEMAND[3:])
    favorable_demand = I(amplitude)*(d1+d2)
    nominal_offset = flows["q11"]-flows["q22"]-d1+d2
    qsum = I(plant.Q01)+I(plant.Q02)
    r1 = I(2.)*flows["q12_plus_q10"]
    r2 = I(2.)*flows["q21"]
    limit = plant.DIRECTIONAL_LIMIT
    forward = nominal_offset+favorable_demand+phi_upper(r1, r2, limit)+phi_upper(qsum, flows["q20"], limit)
    reverse = -nominal_offset+favorable_demand+phi_upper(r2, r1, limit)+phi_upper(flows["q20"], qsum, limit)
    return {
        "tube_mode": index+1,
        "common_level": level,
        "symmetric_P_minimum_eigenvalue_lower": minimum_eigenvalue,
        "inverse_verification": inverse_check,
        "linear_ranges_normalized": {key: interval_record(value) for key, value in ranges.items()},
        "MFD_class_speed_ranges_per_hour": {key: interval_record(value) for key, value in speeds.items()},
        "class_flow_ranges_normalized_per_hour": {key: interval_record(value) for key, value in flows.items()},
        "maximum_region1_heavy_draining_rate_upper_normalized_per_hour": float(forward.hi),
        "maximum_region2_heavy_draining_rate_upper_normalized_per_hour": float(reverse.hi),
    }


def independent_checks(matrices: list[np.ndarray], level: float, bounds: list[dict], amplitude: float) -> dict:
    rng = np.random.default_rng(20260910)
    lp_error = 0.
    for _ in range(1000):
        a, b = rng.uniform(0, 1000, 2)
        limit = float(rng.uniform())
        lp = linprog([-a, b], A_ub=[[1, -1], [-1, 1]], b_ub=[limit, limit], bounds=[(0, 1), (0, 1)], method="highs")
        if not lp.success:
            raise AssertionError(lp.message)
        lp_error = max(lp_error, abs(float(-lp.fun)-phi(a, b, limit)))
    if lp_error > 2e-10:
        raise AssertionError(f"Paired-box identity diagnostic failed: {lp_error}")

    derivative_error, maximum_sample_fraction = 0., 0.
    for k in range(1000):
        mode = k % len(matrices)
        vector = rng.normal(size=6)
        vector *= float(rng.uniform(0., .99))*np.sqrt(level/(vector@matrices[mode]@vector))
        x = plant.X_STAR+vector
        u12, u02 = rng.uniform(0, 1, 2)
        limit = plant.DIRECTIONAL_LIMIT
        u21 = rng.uniform(max(0., u12-limit), min(1., u12+limit))
        u20 = rng.uniform(max(0., u02-limit), min(1., u02+limit))
        u = np.array([u12, u21, u02, u20])
        delta = float(rng.uniform(-amplitude, amplitude))
        demand = plant.DEMAND*(1+delta*np.array([1, 1, 1, -1, -1, -1]))
        dx = plant.mfd_rhs(x, u, demand)
        q11 = plant.class_flow(x[2], x[0])
        q22 = plant.class_flow(x[3], x[1])
        q20 = plant.class_flow(x[5], x[1])
        transfer1 = plant.class_flow(x[0]-x[2], x[0])
        transfer2 = plant.class_flow(x[1]-x[3]-x[5], x[1])
        explicit = 2*transfer1*u12-2*transfer2*u21+(plant.Q01+plant.Q02)*u02-q20*u20+q11-q22-sum(demand[:3])+sum(demand[3:])
        for orientation in (+1, -1):
            observed = -orientation*(dx[0]-dx[1])
            derivative_error = max(derivative_error, abs(observed-orientation*explicit))
            key = f"maximum_region{1 if orientation==1 else 2}_heavy_draining_rate_upper_normalized_per_hour"
            upper = bounds[mode][key]
            if observed > upper+1e-10:
                raise AssertionError("Sample exceeded the analytic rate bound")
            maximum_sample_fraction = max(maximum_sample_fraction, observed/upper)
    if derivative_error > 2e-10:
        raise AssertionError(f"Conservation derivative diagnostic failed: {derivative_error}")

    # Independent high-precision reference calculations test interval inversion
    # and linear supports; the Neumann/support arguments provide the proof.
    high_precision_checks = 0
    with mp.workdps(80):
        for matrix in matrices:
            symmetric = (I(matrix)+I(matrix.T))*I(.5)
            inverse, _ = inverse_of_interval_matrix(symmetric)
            exact_input = mp.matrix(matrix.tolist())
            reference_inverse = ((exact_input+exact_input.T)/2)**-1
            for i in range(6):
                for j in range(6):
                    high_precision_checks += 1
                    if not mp.mpf(float(inverse.lo[i, j])) <= reference_inverse[i, j] <= mp.mpf(float(inverse.hi[i, j])):
                        raise AssertionError("High-precision inverse escaped interval")
            for _ in range(50):
                a = rng.integers(-3, 4, 6).astype(float)
                enclosed = linear_range(a, inverse, level)
                row = mp.matrix([a.tolist()])
                variance = (row*reference_inverse*row.T)[0]
                radius = mp.sqrt(mp.mpf(level)*variance)
                mean = (row*mp.matrix(plant.X_STAR.tolist()))[0]
                high_precision_checks += 2
                if not (mp.mpf(float(enclosed.lo)) <= mean-radius and mean+radius <= mp.mpf(float(enclosed.hi))):
                    raise AssertionError("High-precision support escaped interval")
    return {
        "paired_gate_identity_LP_cases": 1000,
        "paired_gate_identity_maximum_absolute_error": lp_error,
        "conservation_test_states": 1000,
        "oriented_conservation_test_cases": 2000,
        "conservation_maximum_absolute_error_normalized_per_hour": derivative_error,
        "maximum_sample_rate_to_certified_upper_ratio": maximum_sample_fraction,
        "high_precision_reference_decimal_digits": 80,
        "high_precision_inverse_and_support_checks": high_precision_checks,
        "all_checks_passed": True,
    }


def run() -> dict:
    _, certificate, tube = study.load_installed()
    amplitude, level = .003, float(tube["common_level"])
    bounds = [tube_capacity(matrix, level, amplitude, index) for index, matrix in enumerate(certificate["matrices"])]
    upper1 = max(item["maximum_region1_heavy_draining_rate_upper_normalized_per_hour"] for item in bounds)
    upper2 = max(item["maximum_region2_heavy_draining_rate_upper_normalized_per_hour"] for item in bounds)
    direction = np.array([-1., 1., -.5, .5, -.25, .25])
    initial = runtime.state_on_tube(direction, .8, certificate, tube, mode=0)
    # This matches the exact binary target installed in run_study().
    initial_display = abs(100*(initial[0]-initial[1]))
    target = initial_display*.1
    oriented_initial = imat(np.array([[-1., 1., 0, 0, 0, 0]]), I(initial))[0]
    deficit = iv.mpf(float(oriented_initial.lo))-iv.mpf(target)/100
    if upper2 <= 0:
        raise ArithmeticError("The computed bound does not permit positive imbalance draining")
    lower_hours = down(deficit/iv.mpf(upper2))
    lower_minutes = down(deficit/iv.mpf(upper2)*60)
    return {
        "scope": "Necessary optimistic actuator-capacity deadline bound for continuous trajectories contained in the union of both common-level certified tubes",
        "not_a_feasibility_certificate": "A deadline above this lower bound need not be achievable, either robustly or by the installed gain library. Leaving the union of tubes is outside the bound's scope.",
        "demand_family": "Every measurable delta(t) in [-0.003,0.003], with regional demand multipliers 1+delta and 1-delta",
        "actuator_domain": "0<=u12,u21,u02,u20<=1; |u21-u12|<=0.75; |u20-u02|<=0.75",
        "physical_resets": "Physical accumulations remain continuous at changes of the control configuration",
        "state_domain": "Union, not intersection, of (x-X_STAR)^T P_i (x-X_STAR)<=common_level",
        "orientation": "+1: decrease n1-n2; -1: decrease n2-n1",
        "derivation": {
            "paired_support": "Phi(a,b)=delta_gate*a+(1-delta_gate)*max(a-b,0)",
            "region1_maximum": "q11-q22-d1_0+d2_0+amplitude*(d1_0+d2_0)+Phi(2*(q12+q10),2*q21)+Phi(Qsum,q20)",
            "region2_maximum": "q22-q11+d1_0-d2_0+amplitude*(d1_0+d2_0)+Phi(2*q21,2*(q12+q10))+Phi(q20,Qsum)",
            "rate_to_time": "T >= (initial_oriented_imbalance-target)/sup_domain(maximum_oriented_draining_rate)",
            "bounding_method": "Exact linear ellipsoid support with verified interval inverse of Sym(P), verified decreasing positive MFD class factor p(n), positive class products, favorable gate and demand endpoints",
        },
        "tube_bounds": bounds,
        "union_rate_upper_normalized_per_hour": {"region1_heavy": upper1, "region2_heavy": upper2},
        "union_rate_upper_vehicles_per_hour": {"region1_heavy": up(iv.mpf(upper1)*100), "region2_heavy": up(iv.mpf(upper2)*100)},
        "primary_region2_heavy_case": {
            "initial_mode": 1, "initial_occupancy": .8,
            "initial_state": initial.tolist(),
            "initial_imbalance_vehicles": float(initial_display),
            "installed_target_vehicles": float(target),
            "relief_fraction": .9,
            "necessary_deadline_lower_hours": lower_hours,
            "necessary_deadline_lower_minutes": lower_minutes,
        },
        "independent_diagnostics": independent_checks(certificate["matrices"], level, bounds, amplitude),
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"primary": result["primary_region2_heavy_case"], "rate_upper_vehicles_per_hour": result["union_rate_upper_vehicles_per_hour"], "checks": result["independent_diagnostics"]}, indent=2))
