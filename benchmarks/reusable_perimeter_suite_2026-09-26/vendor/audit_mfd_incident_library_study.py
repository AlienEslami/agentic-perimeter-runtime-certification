"""Independent replay/accounting audit of the original three-mode pilot.

Uses saved gains rather than the potentially re-commissioned default library.
This is numerical consistency evidence, not robust or continuous-time proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad, solve_ivp

from audit_mfd_incident_model import reference_field


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
INTERNAL_TOTAL = sum((74.85630, 37.42815, 18.714075, 44.85630, 7.42815, 0.748563))


def pulse(t, amplitude, duration, start=.25, ramp=1/12):
    s = t - start
    if s <= 0 or s >= duration:
        return 60.0
    return 60.0 * (1.0 + amplitude * min(1.0, s/ramp, (duration-s)/ramp))


def cumulative_shape(s, duration=1.0, ramp=1/12):
    if s <= 0:
        return 0.0
    if s <= ramp:
        return s*s/(2*ramp)
    if s <= duration-ramp:
        return s-ramp/2
    if s < duration:
        return duration-ramp-(duration-s)**2/(2*ramp)
    return duration-ramp


def shape_integral(s, duration=1.0, ramp=1/12):
    if s <= 0:
        return 0.0
    if s <= ramp:
        return s**3/(6*ramp)
    if s <= duration-ramp:
        return s*s/2-ramp*s/2+ramp*ramp/6
    if s < duration:
        return (duration-ramp)*(s-duration/2)+(duration-s)**3/(6*ramp)
    return (duration-ramp)*(s-duration/2)


def queue_capacity_time(amplitude, duration=1.0, start=.25, ramp=1/12,
                        initial_queue=2.0, limit=20.0):
    target = (limit-initial_queue)/(60*amplitude)
    if target > duration-ramp:
        return None
    if target <= ramp/2:
        local = np.sqrt(2*ramp*target)
    elif target <= duration-3*ramp/2:
        local = target+ramp/2
    else:
        local = duration-np.sqrt(2*ramp*(duration-ramp-target))
    return float(start+local)


def pulse_formula_audit():
    errors = []
    for s in (.02, 1/12, .25, .6, 11/12, .98, 1., 1.75):
        knots = sorted(set([0., min(s, 1/12), min(s, 11/12), min(s, 1.), s]))
        area = sum(quad(lambda v: (pulse(v+.25, 1., 1.)-60)/60, a, b,
                        epsabs=1e-12, epsrel=1e-12)[0] for a, b in zip(knots[:-1], knots[1:]))
        integral = sum(quad(cumulative_shape, a, b, epsabs=1e-12, epsrel=1e-12)[0]
                       for a, b in zip(knots[:-1], knots[1:]))
        errors.extend([abs(area-cumulative_shape(s)), abs(integral-shape_integral(s))])
    threshold = 18/(60*(1-1/12))
    crossing = queue_capacity_time(.4)
    assert abs(crossing-25/24) < 1e-14
    assert queue_capacity_time(.3) is None
    assert abs(2+24*cumulative_shape(crossing-.25)-20) < 1e-12
    assert max(errors) < 1e-12
    return dict(shape="trapezoid; duration includes both ramps", start_h=.25, duration_h=1., ramp_h=1/12,
                storage_limit_vehicles=2000, initial_queue_vehicles=200,
                overflow_amplitude_threshold=float(threshold),
                thirty_percent=dict(final_queue_vehicles=1850.0, capacity_reached=False,
                                    baseline_tts_veh_h=12400.0, excess_tts_veh_h=2062.5,
                                    total_tts_veh_h=14462.5),
                forty_percent=dict(capacity_reached_at_h=crossing,
                                   capacity_reached_minutes_after_pulse_start=47.5,
                                   capacity_reached_absolute_minutes=62.5,
                                   post_overflow_formula_is_not_admissible_performance=True),
                maximum_formula_vs_quadrature_error=max(errors),
                interpretation="Queue-blind fixed nominal command, not a performance headline comparator.")


def replay(run, saved_library):
    times = np.array(run["times"])
    states = np.array(run["states"])
    amplitude, duration = run["amplitude"], run["duration_h"]
    current = np.r_[states[0], 0., 0., 0.]
    independent_states = [current[:7].copy()]
    prediction_choice_errors = 0
    for index, event in enumerate(run["events"]):
        start = event["time_h"]
        stop = run["events"][index+1]["time_h"] if index+1 < len(run["events"]) else times[-1]
        saved = saved_library[event["executed_mode"]]
        gain, center, offset = map(np.asarray, (saved["gain"], saved["center"], saved["feedforward"]))
        def rhs(t, y):
            x = y[:7]
            u = offset+gain@(x-center)
            derivative, departures = reference_field(x, u, pulse(t, amplitude, duration))
            return np.r_[derivative, 100*sum(x[[0, 1, 6]]), 100*x[6], 100*departures]
        cuts = sorted(set([start, stop] + [v for v in (.25, .25+1/12, .25+duration-1/12, .25+duration)
                                          if start < v < stop]))
        for left, right in zip(cuts[:-1], cuts[1:]):
            solution = solve_ivp(rhs, (left, right), current, method="DOP853", rtol=2e-12, atol=2e-13,
                                 max_step=1/240)
            if not solution.success:
                raise RuntimeError(solution.message)
            current = solution.y[:, -1]
        independent_states.append(current[:7].copy())
        feasible = [candidate for candidate in event["nominal_candidates"] if candidate["nominal_feasible"]]
        if event["reason"] == "nominal_optimizer":
            prediction_choice_errors += int(not feasible or feasible[0]["pair"][0] != event["executed_mode"])
    tts_reconstructed = 100*np.trapezoid(np.sum(states[:, [0, 1, 6]], axis=1), times)
    queue_reconstructed = 100*np.trapezoid(states[:, 6], times)
    offered = (INTERNAL_TOTAL+60)*times[-1]+60*amplitude*(duration-1/12)
    complete_from_balance = 100*(sum(states[0, [0, 1, 6]])+offered-sum(states[-1, [0, 1, 6]]))
    independent_complete_from_balance = 100*(sum(states[0, [0, 1, 6]])+offered-sum(current[[0, 1, 6]]))
    result = dict(amplitude=amplitude,duration_h=duration,
                  tts_reconstruction_error_veh_h=float(tts_reconstructed-run["total_time_spent_veh_h"]),
                  queue_reconstruction_error_veh_h=float(queue_reconstructed-run["queue_time_veh_h"]),
                  stored_completion_balance_error_vehicles=float(complete_from_balance-run["departures"]),
                  independent_completion_balance_error_vehicles=float(independent_complete_from_balance-current[9]),
                  independent_replay_tts_veh_h=float(current[7]),
                  stored_minus_independent_tts_veh_h=float(run["total_time_spent_veh_h"]-current[7]),
                  terminal_state_replay_error=float(max(abs(states[-1]-current[:7]))),
                  nominal_argmin_choice_errors=prediction_choice_errors,
                  numerical_service_pass=run["service_pass"], fallback_count=run["fallbacks"])
    assert abs(result["tts_reconstruction_error_veh_h"]) < 1e-7
    assert abs(result["queue_reconstruction_error_veh_h"]) < 1e-7
    assert abs(result["stored_completion_balance_error_vehicles"]) < .1
    assert abs(result["independent_completion_balance_error_vehicles"]) < 1e-6
    assert abs(result["stored_minus_independent_tts_veh_h"]) < .01
    assert result["terminal_state_replay_error"] < 1e-7
    assert prediction_choice_errors == 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=RESULTS/"mfd_incident_library_nominal_results.json")
    parser.add_argument("--library-record", type=Path, default=RESULTS/"mfd_incident_model_pilot_results.json")
    args = parser.parse_args()
    payload = json.loads(args.runs.read_text(encoding="utf-8"))
    library = json.loads(args.library_record.read_text(encoding="utf-8"))["library"]
    cases = [replay(run, library) for run in payload["runs"]]
    result = dict(status="pass", numerical_audit_not_a_robust_certificate=True,
                  frozen_runs_source=str(args.runs), frozen_library_source=str(args.library_record),
                  saved_library=library, pulse_formula=pulse_formula_audit(), cases=cases,
                  scope="Original three-high-target-mode commissioning pilot; preserve before re-commissioning.")
    output = RESULTS/"mfd_incident_library_nominal_audit_results.json"
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status="pass", cases=len(cases), output=str(output),
                          maximum_terminal_state_error=max(c["terminal_state_replay_error"] for c in cases),
                          maximum_tts_difference_veh_h=max(abs(c["stored_minus_independent_tts_veh_h"]) for c in cases)), indent=2))


if __name__ == "__main__":
    main()
