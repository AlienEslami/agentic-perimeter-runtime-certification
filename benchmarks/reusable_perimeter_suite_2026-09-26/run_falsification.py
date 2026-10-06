"""Adversarial uncertainty search for unchanged, uncertified MPC deadlines.

Run as a separate diagnostic; never modify original plans or results.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from common import HERE, START, DURATIONS, batch_field, fixture, mpc, plant, save

OUTPUT = HERE / "results/falsification_2026-09-27"
HORIZON = 2 / 3
CORNER_SIGNS = np.asarray(list(itertools.product((-1., 1.), repeat=7)))


def source_hashes():
    paths = [HERE / "FALSIFICATION_PROTOCOL.md", HERE / "run_falsification.py",
             HERE / "common.py", HERE / "anchor.json",
             HERE / "vendor/mfd_incident_model.py", HERE / "vendor/mfd_incident_mpc.py"]
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def batch_replay(initial_parameters, profiles, commands, contract, step_seconds=10):
    """Normalized uncertainty parameters and 40 one-minute residual levels."""
    initial_parameters = np.asarray(initial_parameters, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    assert initial_parameters.shape[1] == 7 and profiles.shape == (len(initial_parameters), 40)
    assert np.max(np.abs(initial_parameters)) <= 1 and np.max(np.abs(profiles)) <= 1
    x = np.asarray(fixture()["center"]) + contract["width_vehicles"] / 100 * initial_parameters
    assert np.min(x @ mpc.STATE_A.T + mpc.STATE_B) >= 0
    minimum = np.min(x @ mpc.STATE_A.T + mpc.STATE_B, axis=1)
    peak_queue = x[:, 6].copy()
    h = step_seconds / 3600
    minute_steps = round(60 / step_seconds)
    assert abs(minute_steps * step_seconds - 60) < 1e-12
    steps = 40 * minute_steps
    for k in range(steps):
        t = float(START) + k * h
        u = commands[0 if k < 5 * minute_steps else 1]
        forcing = contract["residual_veh_h"] / 100 * profiles[:, k // minute_steps]

        def rhs(y, clock):
            offer = plant.arrival(clock, amplitude=contract["amplitude"], duration=contract["duration_h"]) + forcing
            return batch_field(y, u, offer)[0]

        f1 = rhs(x, t); x2 = x + h * f1 / 2
        f2 = rhs(x2, t + h / 2); x3 = x + h * f2 / 2
        f3 = rhs(x3, t + h / 2); x4 = x + h * f3
        f4 = rhs(x4, t + h)
        new = x + h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
        for y in (x2, x3, x4, new):
            minimum = np.minimum(minimum, np.min(y @ mpc.STATE_A.T + mpc.STATE_B, axis=1))
            peak_queue = np.maximum(peak_queue, y[:, 6])
        x = new
    return dict(deadline_n1_vehicles=100 * x[:, 0], minimum_margin_vehicles=100 * minimum,
                peak_queue_vehicles=100 * peak_queue, final_states=x)


class SensitivityPrediction:
    """Exact derivative of the fixed-step, branchwise RK4 terminal map."""
    def __init__(self, commands, contract, segments):
        self.commands, self.contract, self.segments = commands, contract, segments
        self.key = None; self.calls = 0

    def evaluate(self, vector):
        if self.key is not None and np.array_equal(vector, self.key):
            return self.record
        self.key = np.array(vector).copy(); self.calls += 1
        c = self.contract; n = 7 + self.segments
        x = np.asarray(fixture()["center"]) + c["width_vehicles"] / 100 * vector[:7]
        sensitivity = np.zeros((7, n)); sensitivity[:, :7] = np.eye(7) * c["width_vehicles"] / 100
        h = 1 / 360
        steps_per_segment = 240 // self.segments
        assert steps_per_segment * self.segments == 240
        for k in range(240):
            t = float(START) + k * h
            u = self.commands[0 if k < 30 else 1]
            j = k // steps_per_segment
            forcing = c["residual_veh_h"] / 100 * vector[7 + j]

            def rhs(y, sy, clock):
                offer = plant.arrival(clock, amplitude=c["amplitude"], duration=c["duration_h"]) + forcing
                f, fx, _ = mpc.field_jacobians(y, u, offer)
                ds = fx @ sy
                ds[6, 7 + j] += c["residual_veh_h"] / 100
                return f, ds

            f1, s1 = rhs(x, sensitivity, t)
            f2, s2 = rhs(x + h*f1/2, sensitivity + h*s1/2, t + h/2)
            f3, s3 = rhs(x + h*f2/2, sensitivity + h*s2/2, t + h/2)
            f4, s4 = rhs(x + h*f3, sensitivity + h*s3, t + h)
            x += h/6 * (f1 + 2*f2 + 2*f3 + f4)
            sensitivity += h/6 * (s1 + 2*s2 + 2*s3 + s4)
        self.record = (-100 * float(x[0]), -100 * sensitivity[0].copy())
        return self.record


def independent_replay(initial_parameters, profile, commands, contract):
    """Scalar original plant, adaptive integration, exact discontinuity splits."""
    initial = np.asarray(fixture()["center"]) + contract["width_vehicles"] / 100 * initial_parameters
    y = np.r_[initial, np.zeros(5)]
    minimum = float(np.min(mpc.STATE_A @ initial + mpc.STATE_B))
    peak = initial[6]; evaluations = 0; accepted_steps = 0
    cuts = float(START) + np.arange(41) / 60
    row = np.array([1., contract["weight"], 0, 0, 0, 0, 1.])
    for j, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
        u = commands[0 if j < 5 else 1]
        forcing = contract["residual_veh_h"] / 100 * profile[j]

        def rhs(t, yy):
            offer = plant.arrival(t, amplitude=contract["amplitude"], duration=contract["duration_h"]) + forcing
            f = plant.field(yy[:7], u, offer)
            return np.r_[f, 100 * sum(yy[[0, 1, 6]]), 100 * yy[6],
                         100 * plant.departure_rate(yy[:7], u), 100 * (row @ yy[:7]),
                         100 * (sum(plant.INTERNAL_DEMAND) + offer)]

        sol = solve_ivp(rhs, (left, right), y, method="DOP853", rtol=1e-11,
                        atol=1e-12, max_step=2 / 3600, dense_output=True)
        assert sol.success, sol.message
        samples = sol.sol(np.linspace(left, right, 61))[:7].T
        minimum = min(minimum, float(np.min(samples @ mpc.STATE_A.T + mpc.STATE_B)))
        peak = max(peak, float(np.max(samples[:, 6])))
        y = sol.y[:, -1]; evaluations += sol.nfev; accepted_steps += len(sol.t) - 1
    balance = 100 * (sum(y[[0, 1, 6]]) - sum(initial[[0, 1, 6]])) - y[11] + y[9]
    return dict(deadline_n1_vehicles=float(100*y[0]), minimum_margin_vehicles=100*minimum,
                peak_queue_vehicles=100*peak, final_state=y[:7].tolist(),
                tts_veh_h=float(y[7]), queue_veh_h=float(y[8]), completed_trips=float(y[9]),
                priority_cost=float(y[10]), conservation_residual_vehicles=float(balance),
                evaluations=evaluations, accepted_steps=accepted_steps)


def records():
    result = []
    for path in sorted((HERE / "results/full_scenario").glob("D*.json")):
        record = json.loads(path.read_text())
        policy = record["policies"]["full_scenario_mpc"]
        if not policy["certificate"]["accepted"]:
            assert policy["certificate"]["verified_continuous_physical"]
            assert not policy["certificate"]["verified_scheduled_service"]
            result.append((path, record))
    assert len(result) == 36
    return result


def audit():
    _, record = next((p, r) for p, r in records() if r["contract"]["id"] == "D027")
    c = record["contract"]; commands = np.asarray(record["policies"]["full_scenario_mpc"]["commands"])
    rng = np.random.default_rng(20260927)
    x = np.asarray(fixture()["center"]) + rng.uniform(-.01, .01, (10, 7))
    offers = rng.uniform(75, 81, 10)
    batch = batch_field(x, commands[1], offers)[0]
    scalar = np.array([plant.field(y, commands[1], a) for y, a in zip(x, offers)])
    field_error = float(np.max(np.abs(batch - scalar)))
    assert field_error < 1e-11
    gradient_errors = {}
    for segments in (8, 40):
        predictor = SensitivityPrediction(commands, c, segments)
        v = rng.uniform(-.7, .7, 7 + segments)
        _, gradient = predictor.evaluate(v); gradient = gradient.copy()
        eps = 2e-5; numerical = []
        for k in range(len(v)):
            direction = np.eye(len(v))[k] * eps
            numerical.append((predictor.evaluate(v + direction)[0] - predictor.evaluate(v - direction)[0]) / (2 * eps))
        error = float(np.max(np.abs(gradient - numerical)))
        assert error < 2e-5, error
        gradient_errors[str(segments)] = error
    result = dict(field_maximum_error=field_error, gradient_maximum_absolute_errors=gradient_errors,
                  source_hashes=source_hashes())
    save(OUTPUT / "implementation_audit.json", result)
    return result


def run_case(path):
    begin = time.perf_counter(); record = json.loads(path.read_text())
    c = record["contract"]; policy = record["policies"]["full_scenario_mpc"]
    commands = np.asarray(policy["commands"]); cid = c["id"]
    target = OUTPUT / "cases" / f"{cid}.json"
    hashes = source_hashes()
    if target.exists():
        cached = json.loads(target.read_text())
        assert cached["source_hashes"] == hashes
        return cid, "cached"
    seed = 2026092700 + int(cid[1:]); rng = np.random.default_rng(seed)
    high = np.ones((128, 40))
    baseline = batch_replay(CORNER_SIGNS, high, commands, c)
    old = {row["corner"]: row["deadline_n1_vehicles"] for row in policy["realizations"] if row["profile"] == "constant_high"}
    archived = np.array([old["".join("+" if s > 0 else "-" for s in corner)] for corner in CORNER_SIGNS])
    replay_error = float(np.max(np.abs(archived - baseline["deadline_n1_vehicles"])))
    assert replay_error < 1e-7, (cid, replay_error)
    ranking = np.argsort(baseline["deadline_n1_vehicles"])[::-1]
    block_profiles = np.asarray(list(itertools.product((-1., 1.), repeat=8)))
    structured_initial = np.repeat(CORNER_SIGNS[ranking[:8]], 256, axis=0)
    structured_profile = np.tile(np.repeat(block_profiles, 5, axis=1), (8, 1))
    families = [
        ("constant_high_corners", CORNER_SIGNS, high, baseline),
        ("exhaustive_5min_top8_corners", structured_initial, structured_profile, None),
        ("random_interior_1min_levels", rng.uniform(-1, 1, (2048, 7)), rng.uniform(-1, 1, (2048, 40)), None),
        ("random_corners_1min_bangbang", CORNER_SIGNS[rng.integers(0, 128, 2048)], rng.choice((-1., 1.), (2048, 40)), None)]
    summaries = []; starts = []; best = None
    for name, initials, profiles, result in families:
        if result is None:
            result = batch_replay(initials, profiles, commands, c)
        j = int(np.argmax(result["deadline_n1_vehicles"]))
        value = float(result["deadline_n1_vehicles"][j])
        candidate = (value, initials[j].copy(), profiles[j].copy(), name)
        starts.append(candidate)
        if best is None or value > best[0]:
            best = candidate
        summaries.append(dict(family=name, trajectories=len(initials),
                              maximum_deadline_n1_vehicles=value,
                              sampled_deadline_violations=int(np.sum(result["deadline_n1_vehicles"] > c["deadline_vehicles"] + 1e-6)),
                              sampled_physical_violations=int(np.sum(result["minimum_margin_vehicles"] < -1e-5)),
                              minimum_physical_margin_vehicles=float(np.min(result["minimum_margin_vehicles"]))))
    attempts = []
    for segments in (8, 40):
        predictor = SensitivityPrediction(commands, c, segments)
        for _, initial, profile, name in starts:
            compressed = profile.reshape(segments, 40 // segments).mean(axis=1)
            guess = np.r_[initial, compressed]
            solution = minimize(predictor.evaluate, guess, jac=True, method="L-BFGS-B",
                                bounds=[(-1., 1.)] * len(guess),
                                options=dict(maxiter=150, ftol=1e-14, gtol=1e-9, maxls=30))
            solved_initial = np.clip(solution.x[:7], -1, 1)
            solved_profile = np.repeat(np.clip(solution.x[7:], -1, 1), 40 // segments)
            check = batch_replay(solved_initial[None, :], solved_profile[None, :], commands, c)
            value = float(check["deadline_n1_vehicles"][0])
            agreement = abs(value + float(solution.fun))
            assert agreement < 1e-7, agreement
            attempts.append(dict(segments=segments, start_family=name, success=bool(solution.success),
                                 message=str(solution.message), iterations=int(solution.nit),
                                 evaluations=int(solution.nfev), maximum_deadline_n1_vehicles=value,
                                 initial_parameters=solved_initial.tolist(), residual_parameters=solved_profile.tolist(),
                                 forward_batch_agreement_vehicles=agreement))
            if value > best[0]:
                best = (value, solved_initial, solved_profile, f"local_{segments}_{name}")
    value, initial, profile, family = best
    adaptive = independent_replay(initial, profile, commands, c)
    refinements = {}
    for step in (10, 2, 1):
        fine = batch_replay(initial[None, :], profile[None, :], commands, c, step)
        refinements[str(step)] = float(fine["deadline_n1_vehicles"][0])
    disagreement = max(abs(a - adaptive["deadline_n1_vehicles"]) for a in refinements.values())
    tolerance = max(1e-5, 10 * disagreement)
    excess = adaptive["deadline_n1_vehicles"] - c["deadline_vehicles"]
    status = "confirmed_deadline_violation" if excess > tolerance else ("borderline" if excess > -tolerance else "no_violation_found")
    output = dict(contract=c, source_hashes=hashes, original_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  original_physical_certificate_passed=True,
                  original_service_certificate=policy["certificate"]["service_check_results"],
                  original_commands=commands.tolist(), random_seed=seed,
                  archived_constant_high_maximum_error_vehicles=replay_error,
                  families=summaries, local_searches=attempts, status=status,
                  strongest_candidate=dict(source=family, initial_parameters=initial.tolist(),
                      initial_state=(np.asarray(fixture()["center"]) + c["width_vehicles"] / 100 * initial).tolist(),
                      residual_parameters=profile.tolist(), residual_veh_h=(c["residual_veh_h"] * profile).tolist()),
                  independent_replay=adaptive, rk4_terminal_values_vehicles=refinements,
                  maximum_integration_disagreement_vehicles=disagreement,
                  classification_tolerance_vehicles=tolerance,
                  deadline_margin_vehicles=-excess, wall_seconds=time.perf_counter()-begin)
    save(target, output)
    return cid, dict(status=status, margin_vehicles=round(-excess, 6), seconds=round(output["wall_seconds"], 1))


def summarize_results():
    results = [json.loads(p.read_text()) for p in sorted((OUTPUT / "cases").glob("D*.json"))]
    assert len(results) == 36
    result = dict(cases=36, confirmed_deadline_violations=sum(r["status"] == "confirmed_deadline_violation" for r in results),
                  borderline_cases=sum(r["status"] == "borderline" for r in results),
                  no_violation_found=sum(r["status"] == "no_violation_found" for r in results),
                  fixed_family_trajectories=sum(s["trajectories"] for r in results for s in r["families"]),
                  local_searches=sum(len(r["local_searches"]) for r in results),
                  local_objective_evaluations=sum(s["evaluations"] for r in results for s in r["local_searches"]),
                  local_solver_non_successes=sum(not s["success"] for r in results for s in r["local_searches"]),
                  fixed_family_physical_violations=sum(s["sampled_physical_violations"] for r in results for s in r["families"]),
                  minimum_deadline_margin_vehicles=min(r["deadline_margin_vehicles"] for r in results),
                  maximum_deadline_margin_vehicles=max(r["deadline_margin_vehicles"] for r in results),
                  maximum_integration_disagreement_vehicles=max(r["maximum_integration_disagreement_vehicles"] for r in results),
                  maximum_conservation_error_vehicles=max(abs(r["independent_replay"]["conservation_residual_vehicles"]) for r in results),
                  cases_detail=[dict(id=r["contract"]["id"], width_vehicles=r["contract"]["width_vehicles"],
                                     residual_veh_h=r["contract"]["residual_veh_h"], deadline_vehicles=r["contract"]["deadline_vehicles"],
                                     status=r["status"], deadline_margin_vehicles=r["deadline_margin_vehicles"],
                                     source=r["strongest_candidate"]["source"]) for r in results])
    save(OUTPUT / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(), indent=2)); return
    if args.summary:
        print(json.dumps(summarize_results(), indent=2)); return
    assert (OUTPUT / "implementation_audit.json").exists(), "Run --audit first"
    audit_record = json.loads((OUTPUT / "implementation_audit.json").read_text())
    assert audit_record["source_hashes"] == source_hashes()
    paths = [p for p, r in records() if not args.only or r["contract"]["id"] in args.only.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, path): path.stem for path in paths}
        for future in as_completed(futures):
            print(future.result(), flush=True)
    if not args.only:
        summary = summarize_results()
        print(json.dumps({k: v for k, v in summary.items() if k != "cases_detail"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
