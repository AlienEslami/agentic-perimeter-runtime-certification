"""Independent saved-feedback replay and commissioning audit of eight runs.

No gains are re-synthesized. All controller records and hashes are loaded from
the frozen results. DOP853 replay and dense checks are numerical audits, not
proof of robust safety or intersample constraint satisfaction.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from audit_mfd_incident_model import reference_field
from audit_mfd_incident_library_study import pulse


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
RUNS = RESULTS / "mfd_incident_operational_nominal_grid_results.json"
PROTOCOL = RESULTS / "mfd_incident_commissioned_library.json"
DISCOVERY = RESULTS / "mfd_incident_library_design_results.json"
OUTPUT = RESULTS / "mfd_incident_operational_nominal_audit_results.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def constraints(x, u, limit=20.):
    n1, n2, n11, n22, n10, n20, z = x
    state = np.r_[x, n1-n11-n10, n2-n22-n20, 100-n1, 100-n2, limit-z]
    control = np.r_[u, 1-u, .75-u[1]+u[0], .75+u[1]-u[0],
                    .75-u[3]+u[2], .75+u[3]-u[2]]
    return state, control


def commission_audit(protocol, payload, discovery):
    exact_text = json.dumps(protocol, sort_keys=True, indent=2)+"\n"
    protocol_hash = hashlib.sha256(exact_text.encode()).hexdigest()
    assert protocol_hash == payload["commissioned_protocol_sha256"]
    eligible = sorted((r for r in discovery["records"] if r.get("simulation", {}).get("eligible_for_cost_ranking")),
                      key=lambda r: r["simulation"]["total_time_spent_veh_h"])
    assert discovery["candidate_count"] == len(discovery["records"]) == 64
    assert discovery["eligible_count"] == len(eligible)
    assert discovery["ranked_eligible_names"] == [r["name"] for r in eligible]
    selected, seen = [], set()
    for record in eligible:
        target = (record["target_n1"], record["target_n2"])
        if target not in seen:
            selected.append(record)
            seen.add(target)
        if len(selected) == 3:
            break
    expected_choices = [[r["target_n1"], r["target_n2"], r["reference_arrival"], r["penalty"]] for r in selected]
    ranking_rule_reproduced = expected_choices == protocol["choices"]
    by_name = {candidate["name"]: candidate for candidate in eligible}
    coefficient_errors = []
    for installed in protocol["controller_library"]:
        candidate = by_name[installed["name"]]
        for key in ("center", "feedforward", "gain", "P"):
            coefficient_errors.append(float(np.max(abs(np.array(candidate[key])-np.array(installed[key])))))
    assert max(coefficient_errors) == 0.0
    development = set(map(tuple, protocol["development_cases"]))
    holdout = set(map(tuple, protocol["holdout_cases"]))
    assert development.isdisjoint(holdout)
    assert set((r["amplitude"], r["duration_h"]) for r in payload["runs"]) == development | holdout
    assert (discovery["configuration"]["amplitude"], discovery["configuration"]["duration_h"]) not in holdout
    return dict(protocol_sha256=protocol_hash, candidate_count=64, eligible_count=len(eligible),
                actual_selected_names=[r["name"] for r in protocol["controller_library"]],
                actual_selected_choices=protocol["choices"],
                expected_top_distinct_names=[r["name"] for r in selected],
                expected_top_distinct_choices=expected_choices,
                selection_rule_reproduced=ranking_rule_reproduced, distinct_target_count=len(seen),
                saved_coefficient_max_discrepancy=max(coefficient_errors),
                development_count=len(development), holdout_count=len(holdout),
                holdout_selection_scope="Excluded from the one primary-scenario commissioning ranking; same synthetic model.",
                limitation="Hashes and disjoint records verify content consistency, not independently attest wall-clock chronology.")


def replay(run, protocol):
    library = run["controller_library"]
    assert library == protocol["controller_library"]
    assert digest(library) == run["library_sha256"]
    assert all(event["library_sha256"] == run["library_sha256"] for event in run["events"])
    assert run["full_horizon_completed"] and not run["stopped_infeasible"]
    states, times, commands = np.array(run["states"]), np.array(run["times"]), np.array(run["commands"])
    assert len(times) == len(states) == len(commands)+1
    assert abs(times[-1]-2.) < 1e-12
    actual_parameters = run["model_parameters"]
    assert actual_parameters["boundary_capacity"] == 120.
    assert actual_parameters["release_rate"] == 60.
    assert actual_parameters["base_arrival"] == 60.
    assert actual_parameters["queue_limit"] == 20.
    assert actual_parameters["internal_demand"] == protocol["fixed_common_physical_demand"]
    amplitude, duration = run["amplitude"], run["duration_h"]
    current = np.r_[states[0], 0., 0., 0.]
    minimum_state = minimum_control = np.inf
    maximum_queue = 0.
    service_state = None
    command_errors, state_node_errors = [], []
    for index, event in enumerate(run["events"]):
        start = event["time_h"]
        end = run["events"][index+1]["time_h"] if index+1 < len(run["events"]) else times[-1]
        selected = library[event["executed_mode"]]
        gain, target, feedforward = map(np.asarray, (selected["gain"], selected["center"], selected["feedforward"]))
        def command(x):
            return feedforward+gain@(x-target)
        def rhs(t, y):
            derivative, exits = reference_field(y[:7], command(y[:7]), pulse(t, amplitude, duration))
            return np.r_[derivative, 100*sum(y[[0, 1, 6]]), 100*y[6], 100*exits]
        cuts = sorted(set([start, end] + [p for p in (.25, .25+1/12, .25+duration-1/12,
                                                     .25+duration, 1.) if start < p < end]))
        first_index = round(start*360)
        final_index = round(end*360)
        for j in range(first_index, final_index):
            command_errors.append(float(max(abs(commands[j]-command(states[j])))))
        for left, right in zip(cuts[:-1], cuts[1:]):
            solution = solve_ivp(rhs, (left, right), current, method="DOP853", dense_output=True,
                                 rtol=2e-12, atol=2e-13, max_step=1/240)
            assert solution.success
            dense_times = np.linspace(left, right, max(2, int(np.ceil((right-left)*3600))+1))
            samples = solution.sol(dense_times)[:7].T
            for sample in samples:
                state_margin, control_margin = constraints(sample, command(sample))
                minimum_state = min(minimum_state, float(min(state_margin)))
                minimum_control = min(minimum_control, float(min(control_margin)))
                maximum_queue = max(maximum_queue, float(sample[6]))
            current = solution.y[:, -1]
            if abs(right-1.) < 1e-12:
                service_state = current[:7].copy()
        state_node_errors.append(float(max(abs(current[:7]-states[final_index]))))
    assert service_state is not None
    offered = (sum(actual_parameters["internal_demand"])+60)*2+60*amplitude*(duration-1/12)
    balance = 100*(sum(states[0, [0, 1, 6]])+offered-sum(current[[0, 1, 6]]))-current[9]
    stored_balance = 100*(sum(states[0, [0, 1, 6]])+offered-sum(states[-1, [0, 1, 6]]))-run["departures"]
    service_limit = protocol["operator_service"]["n1_upper"]
    service_margin_vehicles = 100*(service_limit-service_state[0])
    result = dict(amplitude=amplitude, duration_h=duration, held_out=run["held_out"],
                  library_sha256=run["library_sha256"], command_binding_max_error=max(command_errors),
                  decision_state_replay_max_error=max(state_node_errors),
                  final_state=current[:7].tolist(), total_time_spent_veh_h=float(current[7]),
                  queue_time_veh_h=float(current[8]), completed_trips_vehicles=float(current[9]),
                  stored_minus_replay_tts_veh_h=float(run["total_time_spent_veh_h"]-current[7]),
                  stored_minus_replay_queue_time_veh_h=float(run["queue_time_veh_h"]-current[8]),
                  stored_minus_replay_completed_trips_vehicles=float(run["departures"]-current[9]),
                  independent_conservation_error_vehicles=float(balance),
                  stored_quadrature_conservation_error_vehicles=float(stored_balance),
                  deadline_n1_vehicles=100*float(service_state[0]),
                  stored_minus_replay_deadline_n1_vehicles=float(run["service_n1_vehicles"]-100*service_state[0]),
                  deadline_margin_vehicles=float(service_margin_vehicles),
                  minimum_dense_sampled_state_margin=minimum_state,
                  minimum_dense_sampled_input_margin=minimum_control,
                  maximum_dense_sampled_queue_vehicles=100*maximum_queue,
                  dense_sample_spacing_at_most_seconds=1.,
                  sampled_not_continuous_safety=True)
    assert result["held_out"] == ((amplitude, duration) in set(map(tuple, protocol["holdout_cases"])))
    assert max(command_errors) < 1e-12
    # Availability-branch crossings reduce fixed-step RK4's observed order;
    # this is a 0.02-vehicle numerical agreement test, not an error certificate.
    assert max(state_node_errors) < 2e-4, result
    assert abs(balance) < 1e-6
    assert abs(result["stored_minus_replay_tts_veh_h"]) < .1
    assert abs(result["stored_minus_replay_queue_time_veh_h"]) < .1
    assert abs(result["stored_minus_replay_completed_trips_vehicles"]) < .5
    assert abs(result["stored_minus_replay_deadline_n1_vehicles"]) < .02
    assert minimum_state >= -1e-10 and minimum_control >= -1e-10 and service_margin_vehicles >= 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.runs.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    selection = commission_audit(protocol, payload, discovery)
    cases = [replay(run, protocol) for run in payload["runs"]]
    status = "pass" if selection["selection_rule_reproduced"] else "numerical_replay_pass_selection_rule_mismatch"
    result = dict(status=status, source=str(args.runs), source_sha256=hashlib.sha256(args.runs.read_bytes()).hexdigest(),
                  not_a_robust_or_continuous_time_safety_certificate=True,
                  commissioning=selection, cases=cases,
                  summary=dict(cases=len(cases), maximum_command_binding_error=max(r["command_binding_max_error"] for r in cases),
                               maximum_state_replay_error=max(r["decision_state_replay_max_error"] for r in cases),
                               maximum_tts_difference_veh_h=max(abs(r["stored_minus_replay_tts_veh_h"]) for r in cases),
                               maximum_queue_time_difference_veh_h=max(abs(r["stored_minus_replay_queue_time_veh_h"]) for r in cases),
                               maximum_completion_difference_vehicles=max(abs(r["stored_minus_replay_completed_trips_vehicles"]) for r in cases),
                               minimum_dense_sampled_input_margin=min(r["minimum_dense_sampled_input_margin"] for r in cases),
                               minimum_deadline_margin_vehicles=min(r["deadline_margin_vehicles"] for r in cases)))
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=status, output=str(args.output), **result["summary"]), indent=2))


if __name__ == "__main__":
    main()
