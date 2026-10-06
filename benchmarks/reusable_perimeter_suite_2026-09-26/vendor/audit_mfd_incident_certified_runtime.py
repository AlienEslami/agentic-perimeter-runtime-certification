"""Independent certificate-contract, retained-suffix, and falsification audits.

Test-only monkeypatches reject replacement verification; they never grant
acceptance. Production runtime/verifier files are not modified. Numerical
adverse replays are kept separate from outward proof-contract checks.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

import mfd_incident_model as model
import mfd_incident_library_study as runtime
import mfd_incident_verifier as verifier
from audit_mfd_incident_model import reference_field
from audit_mfd_incident_library_study import pulse
from audit_mfd_incident_operational_study import constraints, replay as numerical_replay


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PROTOCOL = RESULTS / "mfd_incident_commissioned_library_v2.json"
GRID = RESULTS / "mfd_incident_operational_v2_certified_grid_results.json"
RETAINED = RESULTS / "mfd_incident_retained_suffix_audit_results.json"
ADVERSE = RESULTS / "mfd_incident_adverse_fixed_plan_audit_results.json"
GRID_AUDIT = RESULTS / "mfd_incident_certified_grid_audit_results.json"
NOMINAL_GRID = RESULTS / "mfd_incident_operational_v2_nominal_grid_results.json"
AUDITED_AFFINE_CONTRACTS = set()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def library_from_records(records):
    return [model.Mode(r["name"], *[np.array(r[key]) for key in ("center", "feedforward", "gain", "A", "P")])
            for r in records]


def mode_fingerprint(record):
    return hashlib.sha256(b"".join(np.asarray(record[key], dtype=float).tobytes()
                                   for key in ("center", "feedforward", "gain", "A", "P"))).hexdigest()


def accepted_attempt(event):
    matches = [item for item in event.get("verification_attempts", []) if item.get("passed")]
    assert len(matches) == 1
    return matches[0]


def exact_affine_contract(record, controller):
    """Check stored interval rows against independently built exact fractions."""
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if key in AUDITED_AFFINE_CONTRACTS:
        return
    rational = lambda value: Fraction.from_float(float(value))
    gain = [[rational(v) for v in row] for row in controller["gain"]]
    center = list(map(rational, controller["center"]))
    offset = [rational(controller["feedforward"][i])-sum(gain[i][j]*center[j] for j in range(7)) for i in range(4)]
    rows, bounds = [], []
    for j in range(7):
        rows.append([Fraction(-int(j == k)) for k in range(7)])
        bounds.append(Fraction(0))
    for left, a, b in ((0, 2, 4), (1, 3, 5)):
        row = [Fraction(0)]*7
        row[left], row[a], row[b] = Fraction(-1), Fraction(1), Fraction(1)
        rows.append(row)
        bounds.append(Fraction(0))
    for j, limit in ((0, 100), (1, 100), (6, 20)):
        rows.append([Fraction(int(j == k)) for k in range(7)])
        bounds.append(Fraction(limit))
    for j in range(4):
        rows.extend([gain[j], [-v for v in gain[j]]])
        bounds.extend([1-offset[j], offset[j]])
    for a, b in ((1, 0), (3, 2)):
        row = [gain[a][j]-gain[b][j] for j in range(7)]
        rows.extend([row, [-v for v in row]])
        bounds.extend([Fraction(3, 4)-offset[a]+offset[b], Fraction(3, 4)+offset[a]-offset[b]])
    assert len(record["rows_lower"]) == len(rows) == 24
    for i, row in enumerate(rows):
        for j, coefficient in enumerate(row):
            assert rational(record["rows_lower"][i][j]) <= coefficient <= rational(record["rows_upper"][i][j])
        assert rational(record["bounds_lower"][i]) <= bounds[i] <= rational(record["bounds_upper"][i])
    AUDITED_AFFINE_CONTRACTS.add(key)


def audit_certificate(cert, run, pair, origin_time):
    assert cert["accepted"] and cert["verified"] and cert["verified_continuous_physical"]
    assert cert["verified_scheduled_service"] and cert["verified_terminal_service"]
    descriptor = cert["proof_descriptor"]
    proof_id = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()
    assert proof_id == cert["proof_id"]
    assert Fraction(descriptor["initial_time"]) == origin_time
    durations = list(map(Fraction, descriptor["durations"]))
    assert all(d > 0 for d in durations)
    end = origin_time+sum(durations)
    assert end == min(Fraction(2), origin_time+1)
    first_block = min(Fraction(1, 12), end-origin_time)
    assert durations == [value for value in (first_block, end-origin_time-first_block) if value > 0]
    assert float(end) == cert["certified_until_h"]
    assert float(end) == cert["enclosed_until_h"] and not cert["failures"]
    fingerprints = descriptor["mode_fingerprints"]
    assert len(fingerprints) == len(durations)
    assert fingerprints == [mode_fingerprint(run["controller_library"][p]) for p in pair[:len(durations)]]
    contract = descriptor["model_contract"]
    assert contract["MFD_coefficients"] == [.0014877, -.29815, 15.0912]
    assert contract["internal_demand"] == run["model_parameters"]["internal_demand"]
    assert contract["admission_capacity"] == 120 and contract["release_rate"] == 60
    assert contract["base_arrival"] == 60 and contract["queue_limit"] == 20
    assert contract["regional_jam_limits"] == [100., 100.] and contract["command_box"] == [0., 1.]
    assert contract["directional_limit"] == .75 and contract["admission_split"] == .5
    assert contract["equations"] == "seven_state_class_conservation_with_min_admission_v1"
    assert contract["state_order"] == ["n1", "n2", "n11", "n22", "n10", "n20", "exterior_queue"]
    assert Fraction(descriptor["pulse"][0]) == Fraction.from_float(run["amplitude"])
    assert Fraction(descriptor["pulse"][1]) == Fraction.from_float(run["duration_h"])
    assert list(map(Fraction, descriptor["pulse"][2:])) == [Fraction(1, 4), Fraction.from_float(1/12)]
    assert descriptor["residual_bound"] == .6
    assert descriptor["metric_strategy"] == "conservation_box"
    assert descriptor["terminal_regional_upper"] is None and descriptor["terminal_queue_upper"] is None
    assert len(descriptor["affine_constraint_contracts"]) == len(durations)
    for position, affine in enumerate(descriptor["affine_constraint_contracts"]):
        assert affine["mode_fingerprint"] == fingerprints[position]
        exact_affine_contract(affine, run["controller_library"][pair[position]])
    assert descriptor["initial_state"] == run["states"][round(float(origin_time)*360)]
    widths = cert["initial_box_units"]
    assert widths == [.01]*7
    initial_L = np.array(descriptor["initial_L"])
    assert initial_L.shape == (7, 7)
    for row, bound in zip(initial_L, descriptor["initial_radius"]):
        exact_support = sum(abs(Fraction.from_float(float(a)))*Fraction.from_float(float(w)) for a, w in zip(row, widths))
        assert Fraction.from_float(float(bound)) >= exact_support
    assert all(min(r) >= 0 for r in [list(cert["continuous_constraint_margins_lower"].values())])
    requested = descriptor["service_checks"]
    if origin_time < 1:
        assert requested == [dict(exact_time="1", coordinate=0, upper=28.5)]
        assert len(cert["service_check_results"]) == 1 and cert["service_check_results"][0]["passed"]
        assert cert["service_check_results"][0]["upper"] <= 28.5
    else:
        assert requested == []
    expected_cuts = [Fraction(i, 12) for i in range(25) if origin_time < Fraction(i, 12) <= end]
    assert list(map(Fraction, descriptor["timecuts"])) == expected_cuts+([Fraction(1)] if requested else [])
    checkpoints = {Fraction(c["exact_time_fraction"]): c for c in cert["reachable_checkpoints"]}
    assert origin_time in checkpoints and end in checkpoints
    required = [Fraction(i, 12) for i in range(25) if origin_time <= Fraction(i, 12) <= end]
    assert all(t in checkpoints for t in required)
    for t, shape in checkpoints.items():
        assert shape["proof_id"] == proof_id
        assert origin_time <= t <= end
        assert shape["shape_kind"] == "componentwise_box"
        assert np.array(shape["radius"]).shape == (7,) and min(shape["radius"]) >= 0
        assert shape["metric_L"] == descriptor["initial_L"]
        assert shape["time_h"] == float(t)
        assert shape["mode_fingerprint"] in fingerprints
    return dict(proof_id=proof_id, exact_expiry=str(end), checked_service=bool(requested),
                checkpoint_count=len(checkpoints), exact_affine_contract_rows_checked=24*len(durations),
                initial_information_box_exact_support_checked=True,
                minimum_continuous_margin=min(cert["continuous_constraint_margins_lower"].values()))


def audit_execution_contract(run):
    states = np.array(run["states"])
    events = run["events"]
    records, retained = [], []
    accepted_by_origin = {}
    for index, event in enumerate(events):
        if event.get("executed_mode") is None:
            assert event["reason"] == "no_verified_initial_plan_or_suffix"
            assert run["stopped_infeasible"] and not run["full_horizon_completed"]
            continue
        now = Fraction(event["exact_time_h"])
        next_time = Fraction(event["exact_next_decision_h"])
        assert now == Fraction(index, 12) and next_time == now+Fraction(1, 12)
        assert event["library_sha256"] == run["library_sha256"]
        if event["reason"] == "verified_replacement":
            attempt = accepted_attempt(event)
            cert = attempt["certificate"]
            assert event["executed_mode"] == attempt["pair"][0]
            record = audit_certificate(cert, run, attempt["pair"], now)
            accepted_by_origin[index] = (cert, attempt["pair"], now)
            assert event["retained_plan_origin_event"] == index
            records.append(dict(event=index, **record))
        else:
            assert event["reason"] == "retained_previous_plan"
            origin = event["retained_plan_origin_event"]
            assert origin in accepted_by_origin
            cert, pair, origin_time = accepted_by_origin[origin]
            descriptor = cert["proof_descriptor"]
            durations = list(map(Fraction, descriptor["durations"]))
            cursor, segment = origin_time, None
            for i, length in enumerate(durations):
                if cursor <= now < cursor+length and next_time <= cursor+length:
                    segment = i
                    break
                cursor += length
            assert segment is not None
            assert event["executed_mode"] == pair[segment]
            assert Fraction(event["retained_plan_exact_expiry_h"]) == origin_time+sum(durations)
            assert Fraction(event["retained_plan_exact_expiry_h"]) >= next_time
            checkpoints = {Fraction(c["exact_time_fraction"]): c for c in cert["reachable_checkpoints"]}
            checks = []
            for at in (now, next_time):
                shape = checkpoints[at]
                point = states[round(float(at)*360)]
                point_check = verifier.contains_information_box(shape, point, 0.)
                measurement_check = verifier.contains_information_box(shape, point, .01)
                assert point_check["passed"]
                checks.append(dict(exact_time=str(at), proof_id=shape["proof_id"],
                                   point_membership=point_check, fresh_information_box_membership=measurement_check))
            retained.append(dict(event=index, original_event=origin, original_proof_id=cert["proof_id"],
                                 authorized_segment=segment, executed_mode=event["executed_mode"], checks=checks,
                                 justification="Restriction of original accepted plan under unchanged execution and uncertainty contract."))
    return dict(accepted_plan_contracts=records, retained_execution_checks=retained,
                accepted_count=len(records), retained_count=len(retained),
                point_membership_checks=sum(len(r["checks"]) for r in retained),
                fresh_information_box_passes=sum(c["fresh_information_box_membership"]["passed"]
                                                 for r in retained for c in r["checks"]),
                note="Stored numerical points are tested outward; fixed-parent flow restriction supplies the actual retained-suffix proof.")


def forced_retention():
    protocol = load_protocol()
    library = library_from_records(protocol["controller_library"])
    original = verifier.PlanVerifier.verify_plan
    tests = []
    for scenario in ("replacements_unavailable_until_deadline", "replacements_unavailable_through_expiry"):
        calls = dict(forced_rejections=0, authentic_calls=0)
        def controlled(self, initial_state, modes, durations_h=None, **kwargs):
            now = Fraction(kwargs.get("initial_time_h", 0))
            deny = now > 0 and (now < 1 or scenario.endswith("through_expiry"))
            if deny:
                calls["forced_rejections"] += 1
                return dict(accepted=False, verified=False, audit_forced_replacement_rejection=True,
                            reason="Independent test denied replacement authority; it did not certify or modify a command.")
            calls["authentic_calls"] += 1
            return original(self, initial_state, modes, durations_h, **kwargs)
        with patch.object(verifier.PlanVerifier, "verify_plan", controlled):
            run = runtime.simulate(.3, 1., certified=True, library=library)
        contract = audit_execution_contract(run)
        assert contract["retained_count"] == 11
        assert run["service_pass"]
        if scenario.endswith("through_expiry"):
            assert run["stopped_infeasible"] and not run["full_horizon_completed"]
            assert abs(run["times"][-1]-1.) < 1e-12 and len(run["commands"]) == 360
            assert run["events"][-1]["executed_mode"] is None
        else:
            assert run["full_horizon_completed"] and run["all_decisions_certified"]
            assert abs(run["times"][-1]-2.) < 1e-12
        tests.append(dict(scenario=scenario, authority_calls=calls, contract_audit=contract, run=run))
        print(json.dumps(dict(scenario=scenario, retained=contract["retained_count"],
                              complete=run["full_horizon_completed"], service_pass=run["service_pass"], **calls)), flush=True)
    payload = dict(status="pass", test_scope="Forced rejection only; no forced acceptance or production code edits",
                   scenarios=tests, numerical_membership_is_not_universal_disturbance_proof=True)
    save(RETAINED, payload)
    return payload


def fixed_plan_falsification(retained_payload=None):
    payload = retained_payload or json.loads(RETAINED.read_text(encoding="utf-8"))
    run = payload["scenarios"][0]["run"]
    attempt = accepted_attempt(run["events"][0])
    cert, pair = attempt["certificate"], attempt["pair"]
    library = library_from_records(run["controller_library"])
    descriptor = cert["proof_descriptor"]
    initial = np.array(descriptor["initial_state"])
    widths = np.array(cert["initial_box_units"])
    assert np.all(widths == .01)
    shapes = {Fraction(s["exact_time_fraction"]): s for s in cert["reachable_checkpoints"]}
    durations = list(map(Fraction, descriptor["durations"]))
    segment_ends = list(itertools.accumulate(durations))
    corner_indices = [0, 127, 1, 2, 4, 8, 16, 32, 64, 126, 125, 123, 119, 111, 95, 63]
    cases = []
    for corner, pattern in itertools.product(corner_indices, ("positive", "negative", "alternating")):
        signs = np.array([1. if corner & (1 << j) else -1. for j in range(7)])
        # A nearest float of center +/- width can lie outside the exact binary
        # interval. Round toward the interior so every tested point is allowed.
        initial_point = np.empty(7)
        for j, sign in enumerate(signs):
            endpoint = Fraction.from_float(float(initial[j]))+int(sign)*Fraction.from_float(float(widths[j]))
            near = float(endpoint)
            if sign > 0 and Fraction.from_float(near) > endpoint:
                near = np.nextafter(near, -np.inf)
            if sign < 0 and Fraction.from_float(near) < endpoint:
                near = np.nextafter(near, np.inf)
            initial_point[j] = near
            assert abs(Fraction.from_float(near)-Fraction.from_float(float(initial[j]))) <= Fraction.from_float(float(widths[j]))
        current = np.r_[initial_point, 0., 0., 0.]
        assert verifier.contains_information_box(shapes[Fraction(0)], initial_point, 0.)["passed"]
        checkpoint_checks, min_state, min_input = [], np.inf, np.inf
        maximum_queue = 0.
        cuts = sorted(set([Fraction(0), Fraction(1), *segment_ends, *shapes,
                           *[Fraction(j, 12) for j in range(13)],
                           *map(Fraction, cert["arrival"]["exact_knot_fractions"])]))
        cuts = [t for t in cuts if 0 <= t <= 1]
        for left, right in zip(cuts[:-1], cuts[1:]):
            middle = (left+right)/2
            segment = next(i for i, end in enumerate(segment_ends) if middle < end)
            mode = library[pair[segment]]
            residual = .6 if pattern == "positive" else -.6 if pattern == "negative" else (.6 if int(middle*12) % 2 == 0 else -.6)
            def rhs(t, y):
                derivative, exits = reference_field(y[:7], mode.command(y[:7]), pulse(t, .3, 1.)+residual)
                return np.r_[derivative, 100*sum(y[[0, 1, 6]]), 100*y[6], 100*exits]
            solution = solve_ivp(rhs, (float(left), float(right)), current, method="DOP853", rtol=3e-11,
                                 atol=2e-12, dense_output=True, max_step=1/360)
            assert solution.success
            sample_times = np.linspace(float(left), float(right), max(2, int(np.ceil(float(right-left)*1800))+1))
            for point in solution.sol(sample_times)[:7].T:
                a, b = constraints(point, mode.command(point))
                min_state, min_input = min(min_state, float(min(a))), min(min_input, float(min(b)))
                maximum_queue = max(maximum_queue, float(point[6]))
            current = solution.y[:, -1]
            if right in shapes:
                check = verifier.contains_information_box(shapes[right], current[:7], 0.)
                assert check["passed"]
                checkpoint_checks.append(dict(exact_time=str(right), passed=True))
        assert min_state >= 0 and min_input >= 0 and current[0] <= 28.5
        cases.append(dict(corner_index=corner, residual_pattern=pattern, residual_bound=.6,
                          initial_state=initial_point.tolist(), final_state=current[:7].tolist(),
                          minimum_dense_sampled_state_margin=min_state, minimum_dense_sampled_input_margin=min_input,
                          maximum_dense_sampled_queue_vehicles=100*maximum_queue, deadline_n1_vehicles=100*current[0],
                          checkpoint_checks=checkpoint_checks, total_time_spent_veh_h=float(current[7]),
                          queue_time_veh_h=float(current[8]), departures_vehicles=float(current[9])))
    result = dict(status="no_falsification_found", proof_id=cert["proof_id"],
                  scope="48 numerical fixed-plan trajectories; not a proof over all corners or measurable residuals",
                  corner_convention="nearest inward-rounded binary points at 16 declared box corners; exact Fraction containment checked",
                  tested_corner_count=len(corner_indices), tested_residual_patterns=3, cases=cases,
                  minimum_dense_sampled_input_margin=min(c["minimum_dense_sampled_input_margin"] for c in cases),
                  maximum_deadline_n1_vehicles=max(c["deadline_n1_vehicles"] for c in cases),
                  all_sampled_points_inside_retained_checkpoints=True)
    save(ADVERSE, result)
    print(json.dumps(dict(adverse_cases=len(cases), status=result["status"],
                          minimum_input_margin=result["minimum_dense_sampled_input_margin"])), flush=True)
    return result


def grid_audit():
    payload = json.loads(GRID.read_text(encoding="utf-8"))
    protocol = load_protocol()
    protocol_hash = hashlib.sha256((json.dumps(protocol, sort_keys=True, indent=2)+"\n").encode()).hexdigest()
    assert payload["commissioned_protocol_sha256"] == protocol_hash
    nominal = json.loads(NOMINAL_GRID.read_text(encoding="utf-8"))
    nominal_by_case = {(r["amplitude"], r["duration_h"]): r for r in nominal["runs"]}
    cases = []
    for run in payload["runs"]:
        assert run["certified"] and run["all_decisions_certified"] and run["full_horizon_completed"]
        assert len(run["events"]) == 24 and len(run["commands"]) == 720
        contracts = audit_execution_contract(run)
        numerical = numerical_replay(run, protocol)
        nominal_run = nominal_by_case[(run["amplitude"], run["duration_h"])]
        equal_keys = ("states", "commands", "total_time_spent_veh_h", "queue_time_veh_h", "departures")
        assert all(run[key] == nominal_run[key] for key in equal_keys)
        cases.append(dict(amplitude=run["amplitude"], duration_h=run["duration_h"], contracts=contracts,
                          numerical_replay=numerical, identical_to_nominal_trajectory_and_cost=True,
                          rejected_certificates=run["rejected_certificates"], retained_fallbacks=run["fallbacks"]))
    expected = set(map(tuple, protocol["development_cases"]+protocol["holdout_cases"]))
    available = {(run["amplitude"], run["duration_h"]) for run in payload["runs"]}
    assert len(available) == len(cases)
    assert available <= expected
    complete = available == expected
    result = dict(status="pass" if complete else "partial_grid_audited", expected_cases=len(expected),
                  available_cases=len(cases), complete_grid=complete, source=str(GRID),
                  source_sha256=hashlib.sha256(GRID.read_bytes()).hexdigest(), cases=cases,
                  commissioned_protocol_sha256=protocol_hash,
                  nominal_source_sha256=hashlib.sha256(NOMINAL_GRID.read_bytes()).hexdigest(),
                  summary=dict(accepted_plan_count=sum(c["contracts"]["accepted_count"] for c in cases),
                               retained_execution_count=sum(c["contracts"]["retained_count"] for c in cases),
                               rejected_replacement_count=sum(c["rejected_certificates"] for c in cases),
                               minimum_continuous_certificate_margin=min(r["minimum_continuous_margin"] for c in cases
                                                                         for r in c["contracts"]["accepted_plan_contracts"]),
                               all_nominal_trajectories_and_costs_identical=True),
                  numerical_checks_separate_from_stored_outward_certificates=True)
    save(GRID_AUDIT, result)
    print(json.dumps(dict(grid_status=result["status"], available_cases=len(cases))), flush=True)
    return result


def reaudit_retained_records():
    payload = json.loads(RETAINED.read_text(encoding="utf-8"))
    for scenario in payload["scenarios"]:
        scenario["contract_audit"] = audit_execution_contract(scenario["run"])
    payload["exact_affine_contract_and_initial_support_rechecked"] = True
    save(RETAINED, payload)
    print("Saved retained traces pass the expanded exact-contract audit.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("retained", "retained-records", "adverse", "grid", "all"), default="all")
    args = parser.parse_args()
    if args.mode == "retained-records":
        reaudit_retained_records()
        return
    retained = forced_retention() if args.mode in ("retained", "all") else None
    if args.mode in ("adverse", "all"):
        fixed_plan_falsification(retained)
    if args.mode in ("grid", "all"):
        if GRID.exists():
            grid_audit()
        else:
            print("Certified grid not yet available; retained/fixed-plan audits remain complete.", flush=True)


if __name__ == "__main__":
    main()
