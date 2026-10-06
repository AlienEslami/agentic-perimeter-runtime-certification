#!/usr/bin/env python3
"""Discovery-only deadline-service study on the unchanged installed MFD library.

Every binary mode schedule on a 15-minute grid through two hours is evaluated.
The operator service targets (one-half, one-quarter, one-tenth of the initial
absolute regional imbalance) and scenario grid are fixed in this script, not
selected from the results.  Batch RK4 trajectories are independently checked
against DOP853 for every reported envelope-separation witness.  Sampled margin
checks here are diagnostics: this script does not establish robust reachability
or continuous-time certification of the future schedule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

import mfd_cubic_pilot as pilot
import mfd_cubic_nonstationary_study as study
import mfd_cubic_runtime_pilot as runtime


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_deadline_exploration_results.json"
STEP_H = 0.25
DEPTH = 8
SUBSTEPS = 120
FRACTIONS = (0.5, 0.25, 0.1)


def batch_rhs(states: np.ndarray, modes: np.ndarray, gains: np.ndarray, delta: float) -> np.ndarray:
    """Vectorization of pilot.mfd_rhs; independently audited below."""
    errors = states - pilot.X_STAR
    commands = pilot.U_STAR + np.einsum("nij,nj->ni", gains[modes], errors)
    n1, n2, n11, n22, n10, n20 = states.T
    u12, u21, u02, u20 = commands.T
    n12 = n1 - n11 - n10
    n21 = n2 - n22 - n20
    c1 = pilot.MFD_A * n1**2 + pilot.MFD_B * n1 + pilot.MFD_C
    c2 = pilot.MFD_A * n2**2 + pilot.MFD_B * n2 + pilot.MFD_C
    q11, q12, q10 = n11 * c1, n12 * c1, n10 * c1
    q22, q21, q20 = n22 * c2, n21 * c2, n20 * c2
    d11, d12, d10, d22, d21, d20 = pilot.DEMAND + delta * study.shift_direction()
    return np.column_stack((
        -q11 - (q12 + q10) * u12 + q21 * u21 + d11 + d12 + d10,
        -q22 + (q12 + q10) * u12 - q21 * u21 + (pilot.Q01 + pilot.Q02) * u02
        - q20 * u20 + d22 + d21 + d20,
        -q11 + q21 * u21 + d11,
        -q22 + q12 * u12 + pilot.Q02 * u02 + d22,
        -q10 * u12 + d10,
        -q20 * u20 + q10 * u12 + d20,
    ))


def diagnostics(states: np.ndarray, modes: np.ndarray, gains: np.ndarray, matrices: np.ndarray, ell: float):
    errors = states - pilot.X_STAR
    commands = pilot.U_STAR + np.einsum("nij,nj->ni", gains[modes], errors)
    n1, n2, n11, n22, n10, n20 = states.T
    input_margin = np.min(np.column_stack((commands, 1.0 - commands,
        0.75 - np.abs(commands[:, 1] - commands[:, 0]),
        0.75 - np.abs(commands[:, 3] - commands[:, 2]))), axis=1)
    state_margin = 100.0 * np.min(np.column_stack((n1, n2, 100.0-n1, 100.0-n2,
        n11, n22, n10, n20, n1-n11-n10, n2-n22-n20)), axis=1)
    occupancy = np.einsum("ni,nij,nj->n", errors, matrices[modes], errors) / ell
    return input_margin, state_margin, occupancy


def delta_value(pattern: str, amplitude: float, depth: int) -> float:
    if pattern == "positive_constant":
        return amplitude
    if pattern == "negative_constant":
        return -amplitude
    if pattern == "alternating_30_min":
        return amplitude if (depth // 2) % 2 == 0 else -(2.0 / 3.0) * amplitude
    raise ValueError(pattern)


def decode(code: int, depth: int) -> list[int]:
    return [(code >> shift) & 1 for shift in range(depth-1, -1, -1)]


def explore_case(name: str, x0: np.ndarray, initial_mode: int, amplitude: float, pattern: str,
                 gains: np.ndarray, certificate: dict[str, Any], tube: dict[str, Any]) -> dict[str, Any]:
    matrices = np.asarray(certificate["matrices"])
    ell = float(tube["common_level"])
    robust = study.robust_record(amplitude, 15.0, certificate, tube)
    floor = robust["common_ultimate_level"]
    rate = robust["practical_flow_rate_per_hour"]
    initial_value = float((x0-pilot.X_STAR) @ matrices[initial_mode] @ (x0-pilot.X_STAR))
    initial_imbalance = 100.0 * abs(float(x0[0]-x0[1]))
    if initial_imbalance < 1.0e-8:
        raise ValueError("Regional-service target requires a nonzero initial imbalance")
    states = x0[None, :]
    modes = np.array([initial_mode])
    physical_safe = np.array([True])
    envelope_safe = np.array([True])
    minimum_input = np.array([np.inf])
    minimum_state = np.array([np.inf])
    maximum_occupancy = np.array([0.0])
    minimum_imbalance = np.array([initial_imbalance])
    rows = []
    dt = STEP_H / SUBSTEPS
    for depth in range(DEPTH):
        prior_modes = np.repeat(modes, 2)
        states = np.repeat(states, 2, axis=0)
        modes = np.tile(np.array([0, 1]), len(states)//2)
        physical_safe = np.repeat(physical_safe, 2)
        envelope_safe = np.repeat(envelope_safe, 2)
        minimum_input = np.repeat(minimum_input, 2)
        minimum_state = np.repeat(minimum_state, 2)
        maximum_occupancy = np.repeat(maximum_occupancy, 2)
        minimum_imbalance = np.repeat(minimum_imbalance, 2)
        threshold = floor + np.exp(-rate * STEP_H * depth) * (initial_value-floor)
        _im, _sm, occ = diagnostics(states, modes, gains, matrices, ell)
        envelope_safe &= (modes == prior_modes) | (ell * occ <= threshold + 1.0e-10)
        delta = delta_value(pattern, amplitude, depth)
        for step in range(SUBSTEPS+1):
            im, sm, occ = diagnostics(states, modes, gains, matrices, ell)
            physical_safe &= (im >= -1.0e-10) & (sm >= -1.0e-8) & (occ <= 1.0+1.0e-9)
            minimum_input = np.minimum(minimum_input, im)
            minimum_state = np.minimum(minimum_state, sm)
            maximum_occupancy = np.maximum(maximum_occupancy, occ)
            minimum_imbalance = np.minimum(minimum_imbalance, 100.0*np.abs(states[:, 0]-states[:, 1]))
            if step == SUBSTEPS:
                break
            k1 = batch_rhs(states, modes, gains, delta)
            k2 = batch_rhs(states + 0.5*dt*k1, modes, gains, delta)
            k3 = batch_rhs(states + 0.5*dt*k2, modes, gains, delta)
            k4 = batch_rhs(states + dt*k3, modes, gains, delta)
            states += (dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
        imbalance = 100.0*np.abs(states[:, 0]-states[:, 1])
        env = envelope_safe & physical_safe
        def best(mask: np.ndarray) -> dict[str, Any] | None:
            choices = np.where(mask)[0]
            if not choices.size:
                return None
            index = int(choices[np.argmin(imbalance[choices])])
            return {"schedule_zero_based": decode(index, depth+1),
                "terminal_imbalance_vehicles": float(imbalance[index]),
                "minimum_predeadline_imbalance_vehicles": float(minimum_imbalance[index]),
                "terminal_state": states[index].tolist(),
                "maximum_sampled_tube_occupancy": float(maximum_occupancy[index]),
                "minimum_sampled_input_margin": float(minimum_input[index]),
                "minimum_sampled_state_margin_vehicles": float(minimum_state[index])}
        held = {}
        for held_mode, index in ((0, 0), (1, 2**(depth+1)-1)):
            held[str(held_mode)] = {"terminal_imbalance_vehicles": float(imbalance[index]),
                "sampled_tube_admissible": bool(physical_safe[index]),
                "strict_envelope_admissible": bool(env[index]),
                "minimum_predeadline_imbalance_vehicles": float(minimum_imbalance[index])}
        rows.append({"deadline_minutes": 15*(depth+1), "enumerated_prefixes": len(states),
            "sampled_tube_admissible_prefixes": int(np.sum(physical_safe)),
            "strict_envelope_admissible_prefixes": int(np.sum(env)),
            "best_tube_admissible": best(physical_safe), "best_strict_envelope": best(env),
            "held_modes": held,
            "targets": [{"fraction_of_initial": frac, "target_imbalance_vehicles": frac*initial_imbalance,
                "tube_admissible_terminal_successes": int(np.sum(physical_safe & (imbalance <= frac*initial_imbalance))),
                "strict_envelope_terminal_successes": int(np.sum(env & (imbalance <= frac*initial_imbalance))),
                "tube_admissible_reach_by_deadline_successes": int(np.sum(physical_safe & (minimum_imbalance <= frac*initial_imbalance))),
                "strict_envelope_reach_by_deadline_successes": int(np.sum(env & (minimum_imbalance <= frac*initial_imbalance)))} for frac in FRACTIONS]})
    return {"name": name, "initial_mode_zero_based": initial_mode, "initial_state": x0.tolist(),
        "initial_imbalance_vehicles": initial_imbalance, "amplitude": amplitude, "pattern": pattern,
        "initial_tube_occupancy": initial_value/ell,
        "common_robust_floor_tube_occupancy": floor/ell, "deadlines": rows}


def validate_schedule(case: dict[str, Any], schedule: list[int], gains: np.ndarray,
                      certificate: dict[str, Any], tube: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(case["initial_state"])
    matrices = np.asarray(certificate["matrices"])
    sampled_states = [state.copy()]
    sampled_times = [0.0]
    minimum_input, minimum_state, maximum_occ = np.inf, np.inf, 0.0
    for depth, mode in enumerate(schedule):
        delta = delta_value(case["pattern"], case["amplitude"], depth)
        demand = pilot.DEMAND + delta*study.shift_direction()
        solution = solve_ivp(lambda _t, x: pilot.mfd_rhs(x, pilot.U_STAR+gains[mode]@(x-pilot.X_STAR), demand),
            (depth*STEP_H, (depth+1)*STEP_H), state, method="DOP853",
            rtol=2.0e-12, atol=2.0e-13, max_step=STEP_H/200,
            t_eval=np.linspace(depth*STEP_H, (depth+1)*STEP_H, 201))
        if not solution.success:
            raise RuntimeError(solution.message)
        ss = solution.y.T
        im, sm, occ = diagnostics(ss, np.full(len(ss), mode), gains, matrices, tube["common_level"])
        minimum_input = min(minimum_input, float(np.min(im)))
        minimum_state = min(minimum_state, float(np.min(sm)))
        maximum_occ = max(maximum_occ, float(np.max(occ)))
        sampled_states.extend(ss[1:].tolist())
        sampled_times.extend(solution.t[1:].tolist())
        state = ss[-1]
    imbalance = 100.0*np.abs(np.asarray(sampled_states)[:, 0]-np.asarray(sampled_states)[:, 1])
    return {"terminal_state": state.tolist(), "terminal_imbalance_vehicles": float(imbalance[-1]),
        "minimum_predeadline_imbalance_vehicles": float(np.min(imbalance)),
        "maximum_sampled_tube_occupancy": maximum_occ, "minimum_sampled_input_margin": minimum_input,
        "minimum_sampled_state_margin_vehicles": minimum_state}


def main() -> None:
    raw_gains, certificate, tube = study.load_installed()
    gains = np.asarray(raw_gains)
    test_rng = np.random.default_rng(1917)
    test_states = pilot.X_STAR + test_rng.normal(0, .03, (30, 6))
    test_modes = test_rng.integers(0, 2, 30)
    batched = batch_rhs(test_states, test_modes, gains, .003)
    direct = np.array([pilot.mfd_rhs(x, pilot.U_STAR+gains[m]@(x-pilot.X_STAR),
        pilot.DEMAND+.003*study.shift_direction()) for x, m in zip(test_states, test_modes)])
    rhs_residual = float(np.max(np.abs(batched-direct)))
    if rhs_residual > 2.0e-12:
        raise AssertionError(rhs_residual)
    cases = []
    for mode in (0, 1):
        metric = certificate["matrices"][mode]
        region_vector = np.array([1., -1., 0., 0., 0., 0.])
        support_direction = np.linalg.solve(metric, region_vector)
        directions = {"structured_region_1_heavy": runtime.structured_directions()["region_1_heavy"],
            "structured_region_2_heavy": runtime.structured_directions()["region_2_heavy"],
            "positive_imbalance_support": support_direction,
            "negative_imbalance_support": -support_direction}
        for name, direction in directions.items():
            x0 = runtime.state_on_tube(direction.copy(), .8, certificate, tube, mode)
            for amplitude in (0.0, .003, .006):
                patterns = ("positive_constant",) if amplitude == 0 else (
                    "positive_constant", "negative_constant", "alternating_30_min")
                for pattern in patterns:
                    cases.append(explore_case(name, x0, mode, amplitude, pattern, gains, certificate, tube))
                    print(f"completed {len(cases)}: {name}, mode={mode}, amplitude={amplitude}, {pattern}", flush=True)
    witnesses = []
    for case_index, case in enumerate(cases):
        for row in case["deadlines"]:
            for target in row["targets"]:
                if target["tube_admissible_terminal_successes"] and not target["strict_envelope_terminal_successes"]:
                    best = row["best_tube_admissible"]
                    reference = row["best_strict_envelope"]
                    witnesses.append({"case_index": case_index, "name": case["name"],
                        "initial_mode_zero_based": case["initial_mode_zero_based"],
                        "amplitude": case["amplitude"], "pattern": case["pattern"],
                        "deadline_minutes": row["deadline_minutes"], "target": target,
                        "best_tube_admissible": best, "best_strict_envelope": reference,
                        "reach_by_deadline_also_separated": target["strict_envelope_reach_by_deadline_successes"] == 0})
    hold_separations = []
    for case_index, case in enumerate(cases):
        for row in case["deadlines"]:
            held = row["held_modes"][str(case["initial_mode_zero_based"]) ]
            for target in row["targets"]:
                if (target["tube_admissible_terminal_successes"]
                    and held["sampled_tube_admissible"]
                    and held["terminal_imbalance_vehicles"] > target["target_imbalance_vehicles"]):
                    hold_separations.append({"case_index": case_index, "name": case["name"],
                        "initial_mode_zero_based": case["initial_mode_zero_based"],
                        "amplitude": case["amplitude"], "pattern": case["pattern"],
                        "deadline_minutes": row["deadline_minutes"], "target": target,
                        "initial_mode_hold": held,
                        "best_tube_admissible": row["best_tube_admissible"],
                        "best_strict_envelope": row["best_strict_envelope"],
                        "interpretation": "task-aware choice improves this fixed-deadline service target; strict envelope also admits the successful choice"})
    for witness in witnesses + hold_separations:
        case = cases[witness["case_index"]]
        validated = validate_schedule(case, witness["best_tube_admissible"]["schedule_zero_based"], gains, certificate, tube)
        rk4_state = np.asarray(witness["best_tube_admissible"]["terminal_state"])
        discrepancy = 100.0*float(np.max(np.abs(rk4_state-np.asarray(validated["terminal_state"]))))
        if discrepancy > .0001:
            raise AssertionError(f"Batch/scalar integrator disagreement: {discrepancy} vehicles")
        witness["independent_DOP853_validation"] = validated
        witness["maximum_terminal_state_difference_vehicles"] = discrepancy
        if "initial_mode_hold" in witness:
            held_schedule = [case["initial_mode_zero_based"]] * len(witness["best_tube_admissible"]["schedule_zero_based"])
            witness["held_mode_independent_DOP853_validation"] = validate_schedule(case, held_schedule, gains, certificate, tube)
    result = {"scope": "discovery-only trajectory comparison, not a formal robust future-plan certificate",
        "unchanged_library": "mfd_cubic_pilot_results.json and mfd_cubic_sos_pilot_results.json",
        "protocol": {"proposal_period_minutes": 15, "maximum_deadline_minutes": 120,
            "initial_occupancy": .8, "target_fractions": FRACTIONS,
            "disturbance_amplitudes": [0., .003, .006],
            "physical_service_variable": "absolute regional imbalance 100*abs(n1-n2), vehicles",
            "selection": "fixed targets and full scenario grid; all cases retained",
            "RK4_substeps_per_decision": SUBSTEPS,
            "safety_checks": "sampled state/input margins and tube occupancy; no claim of continuous-time proof",
            "baseline": "best of every strict robust-envelope-admissible schedule, not just a myopic proposer"},
        "vectorized_rhs_maximum_residual": rhs_residual, "case_count": len(cases),
        "witness_count_terminal_separation": len(witnesses),
        "witness_count_reach_by_deadline_separation": sum(w["reach_by_deadline_also_separated"] for w in witnesses),
        "witnesses": witnesses, "hold_service_separations": hold_separations, "cases": cases}
    OUTPUT.write_text(json.dumps(study.json_ready(result), indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "witnesses": len(witnesses),
        "hold_service_separations": len(hold_separations),
        "reach_by_deadline_separation": result["witness_count_reach_by_deadline_separation"],
        "rhs_residual": rhs_residual}, indent=2))


if __name__ == "__main__":
    main()
