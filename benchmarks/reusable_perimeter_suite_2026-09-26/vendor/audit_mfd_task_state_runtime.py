"""Independent task-state execution/contract audit and high-accuracy replay.

No live calls, numerical optimization, or reachability kernels are executed.
Semantic raw-message scoring belongs to the companion agent audit. This audit
reconstructs execution authority and independently integrates actual commands.
Unprotected pressure episodes may overflow storage: that is reported, not hidden
by a safety assertion. Development episodes have no held-out performance score.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from audit_mfd_structure_online_runtime import (
    check_certificate, descriptor_hash, exact_gate_margins, frac,
    independent_arrival, independent_class_field, independent_physical_margin,
    physical_box_minimum,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
INPUT = RESULTS / "task_state_runtime"
OUTPUT = RESULTS / "mfd_task_state_runtime_independent_audit_results.json"
PERIOD = Fraction(1, 12)
END = Fraction(1)
BASE = [30., 30., 15., 15., 5., 5., 2.]
ROWS = dict(network=[1., 1., 0., 0., 0., 0., 1.],
            region_1=[4., 1., 0., 0., 0., 0., 1.],
            region_2=[1., 4., 0., 0., 0., 0., 1.])
PRESSURE = [.8, .5, 0., .75]
TOLERANCES = dict(dop853_rtol=2e-12, dop853_atol=2e-13, maximum_step_h=1/1200,
    epoch_state_difference_vehicles=.05, integral_difference_vehicle_hours=.02,
    task_integral_difference_vehicle_hours=.05, departure_difference_vehicles=.1,
    mass_balance_vehicles=1e-6, refinement_state_difference_vehicles=1e-5,
    refinement_integral_difference=1e-6,
    diagnostic_guard_record="Departure guard0.1veh covers observed development trapezoid error0.0524veh. State guard0.05veh follows first heldout RK4-versus-DOP853 error0.01061veh; independent refinement and original1veh observation containment are checked separately. No experimental input, command or certificate threshold changed.",
    meaning="Numerical implementation consistency tolerances, never certificate or safety allowances")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle_request(epoch):
    """Independent transcription of the authorized held-out task schedule."""
    objective, remaining = "network", 0
    if epoch in (1, 2):
        objective, remaining = "region_1", 30-5*epoch
    elif epoch in (3, 4):
        objective, remaining = "region_2", 15 if epoch == 3 else 20
    elif epoch == 5:
        objective, remaining = "region_1", 5
    elif epoch == 7:
        objective, remaining = "region_2", 20
    return dict(objective=objective, remaining_minutes=remaining, duration_minutes=remaining,
                cost_row=ROWS[objective])


def verify_sources(mapping, checked):
    for name, expected in mapping.items():
        path = ROOT / name
        if (name, expected) not in checked:
            assert path.is_file(), ("missing bound source", name)
            actual = sha(path)
            assert actual == expected, ("bound source bytes differ", name, expected, actual)
            checked[(name, expected)] = dict(path=name, sha256=expected, current_bytes_verified=True)


def audit_compilation(event):
    """Independently reconstruct dyadic rounding using exact Fraction arithmetic."""
    raw = event.get("raw_candidate_commands")
    record = event.get("command_compilation")
    if raw is None:
        assert record is None
        return None
    if record is None:
        assert event.get("candidate_commands") is None and event.get("command_compilation_failure") is not None
        return dict(passed=True, compiler_refused=True)
    scale = 1 << 40
    integers = []
    for values in raw:
        row = [round(min(Fraction(1), max(Fraction(0), frac(v)))*scale) for v in values]
        for a, b in ((1, 0), (3, 2)):
            if row[a]-row[b] > 3*scale//4:
                excess = row[a]-row[b]-3*scale//4
                row[a] -= (excess+1)//2; row[b] += excess//2
            elif row[b]-row[a] > 3*scale//4:
                excess = row[b]-row[a]-3*scale//4
                row[b] -= (excess+1)//2; row[a] += excess//2
        integers.append(row)
    converted = [[float(Fraction(value, scale)) for value in row] for row in integers]
    assert record["rule"] == "preverification_dyadic_gate_compilation_v1"
    assert record["denominator"] == scale and record["max_correction_allowed"] == 1e-7
    assert record["raw_commands"] == raw and record["compiled_commands"] == converted
    errors = [abs(frac(a)-frac(b)) for old, new in zip(raw, converted) for a, b in zip(old, new)]
    maximum = max(errors)
    assert maximum <= frac(1e-7)
    assert float(maximum) == record["max_absolute_correction"]
    for row, recorded_margins in zip(converted, record["exact_gate_margins"]):
        u = list(map(frac, row))
        margins = [m for value in u for m in (value, 1-value)]
        margins += [m for a, b in ((1, 0), (3, 2)) for m in (Fraction(3, 4)-u[a]+u[b], Fraction(3, 4)+u[a]-u[b])]
        assert min(margins) >= 0 and list(map(str, margins)) == recorded_margins
    assert event["candidate_commands"] == converted[:len(event["candidate_exact_durations_h"]) ]
    return dict(passed=True, compiler_refused=False, denominator=scale,
        exact_maximum_correction=str(maximum), maximum_correction=float(maximum),
        compiled_gate_constraints_exact=True, certified_command_matches_compilation=True)


def compiler_unit_checks():
    """Deterministic tests of the frozen pure compiler; no planner/live calls."""
    module = importlib.import_module("mfd_task_state_runtime_v2")
    accepted = []
    targets = [
        ("positive_exact_direction_boundary", [[0., .75, 0., .75]]),
        ("negative_exact_direction_boundary", [[.75, 0., .75, 0.]]),
        ("upper_box_and_negative_direction_boundary", [[1., .25, 1., .25]]),
        ("one_ulp_positive_direction_violation", [[0., float(np.nextafter(.75, np.inf)), .5, .5]]),
        ("one_ulp_upper_box_violation", [[float(np.nextafter(1., np.inf)), .25, .5, .5]]),
        ("one_ulp_lower_box_violation", [[float(np.nextafter(0., -np.inf)), .5, .5, .5]]),
        ("five_e_minus_12_direction_violation", [[0., .75+5e-12, .5, .5]]),
        ("negative_five_e_minus_12_direction_violation", [[.75+5e-12, 0., .5, .5]]),
        ("mixed_tiny_gate_and_direction_violations", [[.250000000001, 1.000000000004, .9999999999998, .249999999990]]),
    ]
    rng = np.random.default_rng(20370912)
    for index in range(64):
        while True:
            raw = rng.uniform(0., 1., (2, 4))
            if all(min(exact_gate_margins(row)) >= 0 for row in raw):
                break
        targets.append((f"random_feasible_{index:02}", raw.tolist()))
    for name, raw in targets:
        array = np.asarray(raw); before = array.copy()
        converted, record = module.compile_commands(array)
        assert np.array_equal(array, before), "Compiler mutated the raw command array"
        event = dict(raw_candidate_commands=raw, command_compilation=record,
            candidate_commands=converted.tolist(), candidate_exact_durations_h=["1"]*len(converted))
        check = audit_compilation(event)
        second, again = module.compile_commands(converted)
        assert np.array_equal(second, converted) and again["max_absolute_correction"] == 0.
        accepted.append(dict(name=name, passed=True, rows=len(converted),
            maximum_correction=check["maximum_correction"], exact_gate_constraints=True,
            independently_reconstructed=True, idempotent=True, raw_input_unmodified=True))
    refusals = []
    invalid = [
        ("malformed_vector", [.5, .5, .5, .5]),
        ("malformed_three_columns", [[.5, .5, .5]]),
        ("empty_candidate", []),
        ("empty_two_dimensional_candidate", np.empty((0, 4))),
        ("nonfinite_nan", [[.5, .5, float("nan"), .5]]),
        ("nonfinite_positive_infinity", [[.5, .5, float("inf"), .5]]),
        ("nonfinite_negative_infinity", [[.5, .5, float("-inf"), .5]]),
        ("box_correction_exceeds_cap", [[1.+2e-7, .5, .5, .5]]),
        ("negative_box_correction_exceeds_cap", [[-2e-7, .5, .5, .5]]),
        ("direction_correction_exceeds_cap", [[0., .750001, .5, .5]]),
    ]
    for name, value in invalid:
        try:
            module.compile_commands(value)
        except ValueError as exc:
            refusals.append(dict(name=name, passed=True, refused=True, reason=str(exc)))
        else:
            raise AssertionError(("Invalid compiler input admitted", name))
    return dict(passed=True, source_sha256=sha(ROOT/"reproducibility/mfd_task_state_runtime_v2.py"),
        random_seed=20370912, exact_boundaries_and_repairs=accepted,
        rejected_inputs=refusals, accepted_test_count=len(accepted), refused_test_count=len(refusals),
        physical_state_or_certificate_tolerance_modified=False)


def replay(schedule, model, *, include_task_score, refine=False):
    """Independent continuous replay without observation-center resets.

    DOP853 integrates class balances and all costs jointly. Arrival corners split
    intervals explicitly; the held-out evaluation row is fixed within each
    five-minute epoch, regardless of the model's requested objective. Queue peaks
    use dense-output endpoint/mesh and scalar maximization diagnostics.
    """
    y = np.r_[BASE, np.zeros(7)]
    checkpoints, mesh_time, mesh_state = [], [0.], [BASE]
    minimum = np.inf; queue_peak = 200.; queue_peak_time = 0.; overflow = None
    clock = Fraction(0)
    for event in schedule:
        left, right = Fraction(event["left"]), Fraction(event["right"])
        epoch = event["epoch"]
        assert left == clock and right == left+PERIOD
        command = np.asarray(event["command"])
        cost_row = np.asarray(oracle_request(epoch)["cost_row"])
        # Use the same exact binary pulse-ramp coefficient specified by the
        # commissioned model; these are corners, not approximate quadrature cuts.
        corners = [Fraction(1, 4), Fraction(1, 4)+frac(1/12)]
        cuts = sorted({left, right, *(t for t in corners if left < t < right)})
        for a, b in zip(cuts[:-1], cuts[1:]):
            def rhs(t, state):
                offered = independent_arrival(t, model)
                field, departures = independent_class_field(state[:7], command, offered, model)
                return np.r_[field, 100.*sum(state[[0, 1, 6]]), 100.*state[6], 100.*departures,
                    100.*(sum(model["internal_demand"])+offered), 100.*state[0], 100.*state[1], 100.*(cost_row@state[:7])]
            sol = solve_ivp(rhs, (float(a), float(b)), y, method="DOP853",
                rtol=TOLERANCES["dop853_rtol"]/(10 if refine else 1),
                atol=TOLERANCES["dop853_atol"]/(10 if refine else 1),
                max_step=TOLERANCES["maximum_step_h"]/(2 if refine else 1), dense_output=True)
            assert sol.success
            times = sorted(set(sol.t.tolist()+np.linspace(float(a), float(b), 61).tolist()))
            sample = sol.sol(times)[:7].T
            for t, point in zip(times, sample):
                minimum = min(minimum, independent_physical_margin(point, command, model))
                if 100.*point[6] > queue_peak:
                    queue_peak, queue_peak_time = 100.*point[6], t
            maximum = minimize_scalar(lambda t: -100.*float(sol.sol(t)[6]), bounds=(float(a), float(b)),
                method="bounded", options=dict(xatol=1e-13, maxiter=120))
            if maximum.success and -maximum.fun > queue_peak:
                queue_peak, queue_peak_time = float(-maximum.fun), float(maximum.x)
            if overflow is None:
                for k in range(len(times)-1):
                    old, new = sample[k, 6]-model["queue_limit"], sample[k+1, 6]-model["queue_limit"]
                    if old <= 0 < new:
                        overflow = brentq(lambda t: float(sol.sol(t)[6])-model["queue_limit"], times[k], times[k+1], xtol=1e-13)
                        break
                    if old > 0:
                        overflow = times[k]; break
            mesh_time.extend(times[1:]); mesh_state.extend(sample[1:].tolist())
            y = sol.y[:, -1]
        checkpoints.append(dict(epoch=epoch, exact_time=str(right), state=y[:7].tolist(),
            tts_veh_h=float(y[7]), queue_veh_h=float(y[8]), region_1_veh_h=float(y[11]),
            region_2_veh_h=float(y[12]), oracle_task_cost_veh_h=float(y[13])))
        clock = right
    mass = 6200.+y[10]-100.*sum(y[[0, 1, 6]])-y[9]
    assert abs(mass) < TOLERANCES["mass_balance_vehicles"]
    assert abs(y[7]-y[8]-y[11]-y[12]) < 1e-7
    return dict(exact_end=str(clock), initial_state=BASE, final_state=y[:7].tolist(), checkpoints=checkpoints,
        tts_veh_h=float(y[7]), queue_veh_h=float(y[8]), departures_vehicles=float(y[9]), offered_vehicles=float(y[10]),
        region_1_veh_h=float(y[11]), region_2_veh_h=float(y[12]),
        oracle_task_cost_veh_h=float(y[13]) if include_task_score else None,
        development_runtime_formula_integral=float(y[13]) if not include_task_score else None,
        normative_score_eligible=include_task_score, deadline_vehicles=float(100.*y[0]),
        raw_numerical_service_satisfied=bool(clock == END and 100.*y[0] <= 2850.),
        near_service_boundary=bool(abs(100.*y[0]-2850.) < .01),
        maximum_queue_vehicles=float(queue_peak), queue_peak_time_minutes=float(60.*queue_peak_time),
        first_storage_overflow_minutes=None if overflow is None else float(60.*overflow),
        minimum_sampled_physical_margin=float(minimum) if schedule else None,
        conservation_residual_vehicles=float(mass), no_observation_center_resets=True,
        numerical_diagnostic_not_validated_reachability=True,
        sampled_trajectory=dict(time_h=mesh_time, state=mesh_state))


def audit_case(path, checked_sources, artifacts):
    started = time.perf_counter()
    case = read(path); config = case["configuration"]
    case_id = config["case_id"]; certified = config["certified"]
    version = 2 if path.parent.parent.name == "task_state_runtime_v2" else 1
    assert path.parent.name == case_id
    assert read(path.parent/"started.json") == config
    verify_sources(config["source_sha256"], checked_sources)
    verify_sources(config["kernel_source_sha256"], checked_sources)
    protocol_name = "TASK_STATE_RUNTIME_V2_PROTOCOL.md" if version == 2 else "TASK_STATE_RUNTIME_PROTOCOL.md"
    assert config["protocol_sha256"] == sha(ROOT/protocol_name)
    commissioning_path = ROOT/config["commissioning_source"]
    assert sha(commissioning_path) == config["commissioning_sha256"]
    commissioned_commands = read(commissioning_path)["commands"]
    contract = config["hard_contract"]; csha = descriptor_hash(contract)
    assert contract["immutable_source_sha256"] == config["kernel_source_sha256"]
    assert contract["online_protocol_sha256"] == config["kernel_source_sha256"]["STRUCTURE_ONLINE_RUNTIME_PROTOCOL.md"]
    assert contract["physical_half_widths"] == [.01]*7
    assert frac(contract["residual_bound"]) == frac(.6)
    assert contract["service_checks"] == [dict(exact_time="1", coordinate=0, upper=28.5)]
    assert contract["mission_start_exact_h"] == "0"
    assert contract["mission_expiry_exact_h"] == contract["deadline_exact_h"] == "1"
    assert list(map(Fraction, contract["pulse"])) == [frac(.3), Fraction(1), Fraction(1, 4), frac(1/12)]
    baseline = read(RESULTS/"mfd_incident_uncertainty_frontier_v1_r60_conservation_box_proof.json")
    assert contract["model"] == baseline["certificate"]["proof_descriptor"]["model_contract"]
    widths = contract["physical_half_widths"]
    state = BASE.copy(); assert case["trajectory"]["state"][0] == state
    active = None; accepted = []; executed = []; schedule = []; proofs = []; retentions = []
    chain = descriptor_hash(dict(genesis=csha))
    information_checks = []
    candidate_count = 0; no_candidate_count = 0
    pressure_candidates = []; compilation_records = []
    for epoch, event in enumerate(case["decisions"]):
        assert read(path.parent/f"epoch_{epoch:02}.json") == event
        now, following = epoch*PERIOD, (epoch+1)*PERIOD
        assert event["epoch"] == epoch and Fraction(event["exact_time_h"]) == now
        assert Fraction(event["exact_next_decision_h"]) == following
        assert event["initial_state"] == state == case["trajectory"]["state"][30*epoch]
        info = event["information_box"]
        assert Fraction(info["exact_time_h"]) == now and info["half_widths"] == widths
        assert info["center"] == state
        assert info["sensor_contract"] == contract["sensor_contract"]
        assert info["simulation_containment_audit"]["passed"]
        assert info["simulation_containment_audit"]["actual_state"] == state
        identity = {k: v for k, v in info.items() if k not in ("information_id", "simulation_containment_audit")}
        assert descriptor_hash(identity) == info["information_id"]
        errors = [abs(frac(x)-frac(c)) for x, c in zip(state, info["center"])]
        assert all(error <= frac(width) for error, width in zip(errors, widths))
        information_checks.append(dict(epoch=epoch, information_id=info["information_id"], exact_actual_containment=True))
        request = event["tool_request"]
        if event["request_valid"]:
            assert request is not None and request["objective"] in ROWS
            assert request["cost_row"] == ROWS[request["objective"]]
            assert type(request["remaining_minutes"]) is int and 0 <= request["remaining_minutes"] <= 180
            assert request["duration_minutes"] == request["remaining_minutes"]
        if config["provider"] == "oracle" and not config["nominal_counterfactual"]:
            assert request == oracle_request(epoch)
        if config["provider"] in ("network", "command_pressure"):
            assert request == dict(objective="network", remaining_minutes=0, duration_minutes=0, cost_row=ROWS["network"])
        commands = event.get("candidate_commands")
        if version == 2:
            compiled = audit_compilation(event)
            if compiled is not None:
                compilation_records.append(dict(epoch=epoch, **compiled))
        if commands is not None:
            candidate_count += 1
            durations = list(map(Fraction, event["candidate_exact_durations_h"]))
            assert durations == [d for d in (min(PERIOD, END-now), END-now-PERIOD) if d > 0]
            if epoch == 0:
                assert (event["raw_candidate_commands"] if version == 2 else commands) == commissioned_commands
            if config["provider"] == "command_pressure" and epoch in (4, 5, 6, 7):
                assert (event["raw_candidate_commands"] if version == 2 else commands) == [PRESSURE, PRESSURE]
                assert "deterministic command pressure" in event["adversary"]
                pressure_candidates.append(epoch)
        else:
            no_candidate_count += 1
        artifact = None
        reference = event["certificate_reference"]
        if reference is not None:
            assert certified and commands is not None
            pair = {k: reference[k] for k in ("path", "sha256")}
            assert descriptor_hash(pair) == reference["registry_id"]
            assert case["trusted_proof_registry"][reference["registry_id"]] == pair
            artifact_path = ROOT/reference["path"]
            assert sha(artifact_path) == reference["sha256"]
            artifacts[reference["path"]] = reference["sha256"]
            artifact = read(artifact_path)
            assert artifact["origin"] == "trusted_local_verifier" and artifact["method"] == "affine_v3"
            assert artifact["case_id"] == ("taskstate_v2_" if version == 2 else "taskstate_")+case_id and artifact["epoch"] == epoch
            assert artifact["exact_installation_candidate_h"] == str(now)
            assert artifact["current_contract_sha256"] == csha and artifact["information_box"] == info
            assert artifact["commands"] == commands and artifact["exact_durations_h"] == event["candidate_exact_durations_h"]
            assert artifact["certification_source_sha256"] == config["kernel_source_sha256"]
            assert artifact["online_protocol_sha256"] == contract["online_protocol_sha256"]
            cert = artifact["certificate"]
            if cert is not None:
                outcome = check_certificate(cert, now=now, center=info["center"], widths=widths, commands=commands,
                    residual_veh_h=60., service_vehicles=2850., expected_model=contract["model"])
                signed = cert["proof_descriptor"]["signed_generator_method"]
                for name, expected in signed["source_sha256"].items():
                    assert config["kernel_source_sha256"][name] == expected
                assert signed["protocol_sha256"] == config["kernel_source_sha256"]["reproducibility/results/mfd_affine_v3_protocol.json"]
                assert event["candidate_admitted"] == cert["accepted"]
                proofs.append(dict(**outcome, epoch=epoch, artifact_sha256=reference["sha256"], artifact_path=reference["path"]))
            else:
                assert not event["candidate_admitted"] and artifact["precondition_failure"] is not None
                if artifact["precondition_failure"]["kind"] == "invalid_initial_physical_information_set":
                    assert physical_box_minimum(info["center"], widths, commands[0]) < 0
        else:
            assert not certified or commands is None
        if event["candidate_admitted"]:
            if certified:
                assert artifact is not None and artifact["certificate"]["accepted"]
                installation = event["installation"]
                identity = {k: v for k, v in installation.items() if k != "installation_id"}
                assert descriptor_hash(identity) == installation["installation_id"]
                assert installation["method"] == "affine_v3"
                assert installation["original_proof_reference"] == reference
                assert installation["original_proof_id"] == artifact["certificate"]["proof_id"]
                assert installation["original_commands"] == commands
                assert installation["original_exact_durations_h"] == event["candidate_exact_durations_h"]
                assert installation["original_installation_exact_h"] == str(now)
                assert installation["original_expiry_exact_h"] == "1" and installation["original_epoch"] == epoch
                assert installation["information_id"] == info["information_id"]
                assert installation["current_contract_sha256"] == csha
                assert installation["executed_prefix_hash_at_installation"] == chain
                assert installation["accepted_history_prefix_hash"] == descriptor_hash(accepted)
                assert installation["predecessor_installation_id"] == (None if active is None else active["installation_id"])
                active = copy.deepcopy(installation); accepted.append(copy.deepcopy(installation))
            else:
                active = dict(original_commands=commands, original_epoch=epoch)
        else:
            assert event.get("installation") is None
        actual_command = event.get("executed_constant_command")
        if actual_command is None:
            assert certified and active is None and epoch == 0
            assert event["execution_source"] == "unresolved" and not case["metrics"]["complete"]
            continue
        assert active is not None
        if certified:
            cursor = Fraction(active["original_installation_exact_h"]); selected = None
            for index, (command, duration) in enumerate(zip(active["original_commands"], active["original_exact_durations_h"])):
                stop = cursor+Fraction(duration)
                if cursor <= now < stop and following <= stop:
                    selected = (index, command)
                cursor = stop
            assert cursor == END and selected is not None
            index, command = selected
            assert actual_command == command
            binding = dict(installation_id=active["installation_id"], proof_id=active["original_proof_id"],
                proof_reference=active["original_proof_reference"], original_command_index=index,
                original_epoch=active["original_epoch"], original_installation_exact_h=active["original_installation_exact_h"],
                original_expiry_exact_h="1", executed_prefix_hash=chain, exact_left_h=str(now), exact_right_h=str(following))
            assert event["execution_binding"] == binding
            committed = dict(**binding, command=command)
            committed["execution_record_hash"] = descriptor_hash(committed)
            assert event["committed_execution"] == committed
            chain = committed["execution_record_hash"]; executed.append(committed)
            assert event["outer_hull_diagnostic"]["passed"]  # Diagnostic only, never proof replacement.
            if event["candidate_admitted"]:
                assert event["execution_source"] == "verified_replacement"
            else:
                assert event["execution_source"] == "retained_original_certified_suffix"
                retentions.append(dict(epoch=epoch, original_epoch=active["original_epoch"],
                    original_proof_id=active["original_proof_id"], original_expiry="1"))
        else:
            assert actual_command == active["original_commands"][0 if active["original_epoch"] == epoch else -1]
            assert event["execution_source"] == ("nominal_replacement" if event["candidate_admitted"] else "retained_nominal_suffix")
        assert case["trajectory"]["command"][30*epoch:30*(epoch+1)] == [actual_command]*30
        assert event["final_state"] == case["trajectory"]["state"][30*(epoch+1)]
        feedback = event["execution_feedback"]
        assert feedback["accepted"] == event["candidate_admitted"]
        assert feedback["retained_previous_plan"] == (not event["candidate_admitted"])
        assert feedback["executed_gate_command"] == actual_command
        assert abs(feedback["time_minutes"]-float(following)*60.) < 1e-12
        assert feedback["certified_until_minutes"] == (60 if certified else None)
        state = event["final_state"]
        schedule.append(dict(epoch=epoch, left=str(now), right=str(following), command=actual_command))
    assert accepted == case["accepted_history"] and executed == case["executed_history"]
    if not certified:
        assert not case["trusted_proof_registry"]
    metrics = case["metrics"]
    assert metrics["complete"] == (len(schedule) == 12)
    assert metrics["accepted_plans"] == sum(e["candidate_admitted"] for e in case["decisions"])
    assert metrics["retained_epochs"] == len(retentions)
    if certified and metrics["complete"]:
        assert case["expiry_guard"]["passed"]
        assert "expired" in case["expiry_guard"]["reason"]
    if config["provider"] == "command_pressure" and metrics["complete"]:
        assert config["pressure_epochs"] == pressure_candidates == [4, 5, 6, 7]
    eligible = config["split"] != "development"
    independent = replay(schedule, contract["model"], include_task_score=eligible)
    refined = replay(schedule, contract["model"], include_task_score=eligible, refine=True)
    refinement_state = max((100.*max(abs(np.asarray(a["state"])-np.asarray(b["state"])))
                            for a, b in zip(independent["checkpoints"], refined["checkpoints"])), default=0.)
    refinement_fields = ("tts_veh_h", "queue_veh_h", "departures_vehicles", "region_1_veh_h", "region_2_veh_h",
                         "maximum_queue_vehicles", "deadline_vehicles")
    refinement_differences = {field: refined[field]-independent[field] for field in refinement_fields}
    if eligible:
        refinement_differences["oracle_task_cost_veh_h"] = refined["oracle_task_cost_veh_h"]-independent["oracle_task_cost_veh_h"]
    assert refinement_state < TOLERANCES["refinement_state_difference_vehicles"]
    assert max(map(abs, refinement_differences.values())) < TOLERANCES["refinement_integral_difference"]
    event_errors = [100.*max(abs(np.asarray(check["state"])-np.asarray(event["final_state"])))
                    for check, event in zip(independent["checkpoints"], [e for e in case["decisions"] if e.get("executed_constant_command") is not None])]
    max_error = max(event_errors, default=0.)
    assert max_error < TOLERANCES["epoch_state_difference_vehicles"], (case_id, max_error)
    replay_information = []
    for epoch, event in enumerate(case["decisions"]):
        point = BASE if epoch == 0 else independent["checkpoints"][epoch-1]["state"]
        center = event["information_box"]["center"]
        errors = [abs(frac(x)-frac(c)) for x, c in zip(point, center)]
        contained = all(error <= frac(width) for error, width in zip(errors, widths))
        assert contained, (case_id, epoch, "independent replay outside fresh observation box")
        replay_information.append(dict(epoch=epoch, independent_numerical_replay_contained=True,
            maximum_error_vehicles=float(100*max(errors)), numerical_diagnostic_not_sensor_validation=True))
    differences = {}; precision_warnings = []
    for field in ("tts_veh_h", "queue_veh_h", "region_1_veh_h", "region_2_veh_h", "departures_vehicles"):
        differences[field] = independent[field]-metrics[field]
        tolerance = TOLERANCES["departure_difference_vehicles"] if field == "departures_vehicles" else TOLERANCES["integral_difference_vehicle_hours"]
        if field == "departures_vehicles" and abs(differences[field]) >= tolerance:
            precision_warnings.append(dict(field=field, difference=differences[field], retained_guard=tolerance,
                interpretation="Original trapezoidal reporting integral exceeds the tight comparison guard; refined independent integral is the reported value. Execution-state, observation-containment and replay-refinement checks remain separate."))
        else:
            assert abs(differences[field]) < tolerance, (case_id, field, differences[field])
    independent_task = independent["oracle_task_cost_veh_h"] if eligible else independent["development_runtime_formula_integral"]
    differences["oracle_task_cost_veh_h"] = independent_task-metrics["oracle_task_cost_veh_h"]
    assert abs(differences["oracle_task_cost_veh_h"]) < TOLERANCES["task_integral_difference_vehicle_hours"]
    if certified and metrics["complete"]:
        assert independent["deadline_vehicles"] <= 2850.+.005
        assert independent["maximum_queue_vehicles"] <= 2000.+.005
        assert independent["minimum_sampled_physical_margin"] >= -1e-7
    if not eligible:
        performance_scope = "development diagnostic only; excluded from held-out task-quality comparisons"
    elif config["provider"] == "command_pressure":
        performance_scope = "deterministic malicious-command diagnostic, not an observed LLM output"
    elif config["nominal_counterfactual"]:
        performance_scope = "fixed semantic-request replay through fresh MPC on its own state, not a live episode"
    else:
        performance_scope = "held-out or corrupted-memory interaction under the common normative task score"
    return dict(case_id=case_id, runtime_version=version, passed=True, result_path=str(path.relative_to(ROOT)).replace("\\", "/"), result_sha256=sha(path),
        configuration=config, performance_scope=performance_scope, heldout_performance_eligible=eligible,
        proof_records=proofs, sound_information_checks=information_checks, retentions=retentions,
        independent_replay_information_checks=replay_information,
        decision_count=len(case["decisions"]), candidate_count=candidate_count, no_candidate_count=no_candidate_count,
        admitted_count=metrics["accepted_plans"], retained_count=len(retentions), command_pressure_epochs=pressure_candidates,
        complete=metrics["complete"], independent_replay=independent, original_reported_metrics=metrics,
        command_compilations=compilation_records,
        independent_replay_refinement=dict(passed=True, relative_and_absolute_tolerances_divided_by=10,
            maximum_step_divided_by=2, maximum_epoch_state_difference_vehicles=refinement_state,
            metric_differences=refinement_differences),
        numerical_differences=differences, maximum_epoch_state_difference_vehicles=max_error,
        original_reported_integrals_within_guards=not precision_warnings,
        original_reported_integral_precision_warnings=precision_warnings,
        exact_full_artifact_plan_clock_expiry_history_binding=True, arithmetic_kernel_recomputation=False,
        wall_seconds=time.perf_counter()-started)


def menu_status(cases, version=2):
    cases = [case for case in cases if case["runtime_version"] == version]
    def count(provider, split, certified, paraphrase=None, replayed=None):
        return sum(c["configuration"]["provider"] == provider and c["configuration"]["split"] == split
                   and c["configuration"]["certified"] == certified
                   and (paraphrase is None or c["configuration"]["paraphrase"] == paraphrase)
                   and (replayed is None or c["configuration"]["nominal_counterfactual"] == replayed) for c in cases)
    required = []
    for provider in ("oracle", "network"):
        required.append(dict(name=f"{provider}_reference", count=count(provider, "heldout", True, replayed=False), expected=1))
    if version == 1:
        for provider in ("original", "structured"):
            required.append(dict(name=f"development_{provider}", count=count(provider, "development", True), expected=1))
        for certified in (True, False):
            required.append(dict(name=f"command_pressure_certified_{certified}", count=sum(c["configuration"]["provider"] == "command_pressure"
                and c["configuration"]["certified"] == certified for c in cases), expected=1))
        return dict(complete=all(row["count"] == row["expected"] for row in required), expected_completed_results=6, categories=required,
                    interpretation="v1 development and references only; no v1 held-out calls")
    for provider in ("original", "structured"):
        for paraphrase in (0, 1):
            required.append(dict(name=f"heldout_{provider}_p{paraphrase}", count=count(provider, "heldout", True, paraphrase, False), expected=1))
    required.append(dict(name="poison_structured", count=count("structured", "poison", True, replayed=False), expected=1))
    required.append(dict(name="five_nominal_request_replays", count=sum(not c["configuration"]["certified"] and
                         c["configuration"]["nominal_counterfactual"] for c in cases), expected=5))
    for certified in (True, False):
        required.append(dict(name=f"command_pressure_certified_{certified}", count=sum(c["configuration"]["provider"] == "command_pressure"
                             and c["configuration"]["certified"] == certified for c in cases), expected=1))
    return dict(complete=all(row["count"] == row["expected"] for row in required), expected_completed_results=14, categories=required)


def matched_comparisons(cases):
    """Pair by normalized semantic requests, not by physical command equality."""
    out = []
    certified = [c for c in cases if c["configuration"]["certified"] and c["configuration"]["provider"] in ("original", "structured")
                 and c["configuration"]["split"] != "development"]
    nominal = [c for c in cases if c["configuration"]["nominal_counterfactual"] and not c["configuration"]["certified"]]
    for current in certified:
        original = read(ROOT/current["result_path"])
        requests = [(e["request_valid"], e["tool_request"]) for e in original["decisions"]]
        matches = []
        for other in nominal:
            if other["runtime_version"] != current["runtime_version"]:
                continue
            # The frozen runtime names its sole matched replay by this suffix.
            # Identical semantic traces in different episodes are not new pairs.
            if other["case_id"] != current["case_id"]+"_nominal":
                continue
            candidate = read(ROOT/other["result_path"])
            assert [(e["request_valid"], e["tool_request"]) for e in candidate["decisions"]] == requests
            matches.append(other)
        assert len(matches) <= 1
        for other in matches:
            a, b = current["independent_replay"], other["independent_replay"]
            out.append(dict(certified_case=current["case_id"], nominal_case=other["case_id"], exact_semantic_request_match=True,
                paired_by_frozen_case_id_suffix=True,
                tts_premium_veh_h=a["tts_veh_h"]-b["tts_veh_h"], tts_premium_percent=100.*(a["tts_veh_h"]-b["tts_veh_h"])/b["tts_veh_h"],
                oracle_task_cost_difference_veh_h=a["oracle_task_cost_veh_h"]-b["oracle_task_cost_veh_h"]))
    return out


def run(partial=False, version="2"):
    began = time.perf_counter(); checked = {}; artifacts = {}; cases = []
    versions = (1, 2) if version == "all" else (int(version),)
    compiler_checks = compiler_unit_checks() if 2 in versions else None
    paths = sorted(path for value in versions for path in
                   (RESULTS/("task_state_runtime_v2" if value == 2 else "task_state_runtime")).glob("*/result.json"))
    for path in paths:
        result = audit_case(path, checked, artifacts)
        cases.append(result)
        print(json.dumps(dict(case=result["case_id"], passed=True, certified=result["configuration"]["certified"],
            tts=result["independent_replay"]["tts_veh_h"], task_cost=result["independent_replay"]["oracle_task_cost_veh_h"],
            max_queue=result["independent_replay"]["maximum_queue_vehicles"], proofs=len(result["proof_records"]))), flush=True)
    status = {str(value): menu_status(cases, value) for value in versions}
    complete = all(item["complete"] for item in status.values())
    if not partial:
        assert complete, ("frozen menu incomplete; use --partial", status)
    comparisons = matched_comparisons(cases)
    if 2 in versions and status["2"]["complete"]:
        assert len(comparisons) == 5
    outcome = dict(passed=True, complete=complete, audit_source_sha256=sha(__file__),
        imported_audit_source_sha256=sha(ROOT/"reproducibility/audit_mfd_structure_online_runtime.py"), tolerances=TOLERANCES,
        protocol_menu=status, source_checks=list(checked.values()), artifact_checks=artifacts,
        deterministic_command_compiler_unit_checks=compiler_checks,
        cases=cases, matched_nominal_comparisons=comparisons,
        total_full_certificates=sum(len(c["proof_records"]) for c in cases),
        total_sound_information_sets=sum(len(c["sound_information_checks"]) for c in cases),
        total_retained_epochs=sum(c["retained_count"] for c in cases),
        maximum_epoch_state_difference_vehicles=max((c["maximum_epoch_state_difference_vehicles"] for c in cases), default=0.),
        maximum_TTS_difference_veh_h=max((abs(c["numerical_differences"]["tts_veh_h"]) for c in cases), default=0.),
        all_original_reported_integrals_within_guards=all(c["original_reported_integrals_within_guards"] for c in cases),
        original_reported_integral_precision_warnings=[dict(case_id=c["case_id"], **warning) for c in cases
            for warning in c["original_reported_integral_precision_warnings"]],
        complete_nonlinear_interval_recomputation=False, wall_seconds=time.perf_counter()-began)
    archive = RESULTS/"ar"/sha(__file__)[:12]
    archive.mkdir(parents=True, exist_ok=True)
    snapshot = archive/Path(__file__).name
    if snapshot.exists():
        assert sha(snapshot) == sha(__file__)
    else:
        shutil.copyfile(__file__, snapshot)
    outcome["audit_source_snapshot"] = str(snapshot.relative_to(ROOT)).replace("\\", "/")
    output = RESULTS/f"mfd_task_state_runtime_v{version}_independent_audit_results.json"
    outcome["output_path"] = str(output.relative_to(ROOT)).replace("\\", "/")
    output.write_text(json.dumps(outcome, indent=2, allow_nan=False), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--version", choices=("1", "2", "all"), default="2")
    args = parser.parse_args()
    result = run(args.partial, args.version)
    print(json.dumps(dict(passed=result["passed"], complete=result["complete"], cases=len(result["cases"]),
        output=result["output_path"]), indent=2), flush=True)
