"""Fresh closed-loop MPC proposals admitted by the incident plan verifier.

The optimization, plant, forecast, service deadline and numerical integration
are those of the archived MPC baseline. Certificates add the existing full-state
box and measurable arrival-residual contract. A rejected replacement leaves the
original certified suffix active only through its exact expiry. Computation is
measured but activation is synchronous, as in the preceding incident study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

import mfd_incident_model as model
import mfd_incident_mpc as mpc
import mfd_incident_operational_study as operating

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reproducibility/results'
PERIOD = Fraction(1, 12)
END = Fraction(2)
INITIAL_BOX = .01
RESIDUAL_BOUND = .6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(certified, amplitude=.3, duration=1.):
    baseline = mpc.configuration(amplitude, duration)
    baseline['proposal_certificate_scope'] = baseline.pop('certificate_scope')
    baseline['uncertainty'] = 'nominal realization; certified execution covers full-state observation box and bounded arrival residual' if certified else 'nominal realization only'
    return dict(**baseline,
                runtime='fresh actual-state optimization followed by exact-plan admission',
                certified=certified, initial_box_units=INITIAL_BOX,
                residual_bound_units_per_hour=RESIDUAL_BOUND,
                admission_metric='conservation_box', verifier_step_seconds=15.,
                uncertainty_contract='full-state box and all measurable bounded arrival residuals',
                failure_policy='execute unchanged previously accepted suffix until exact expiry; otherwise stop simulation',
                warm_start_after_rejection='last accepted tail, not rejected nominal winner',
                execution_certificate_scope='finite prescribed plan, outward interval physical and scheduled-service checks' if certified else 'none',
                activation='synchronous; wall time measured, physical computation delay not simulated',
                no_new_llm_calls=True)


def run(amplitude=.3, duration=1., *, certified=True, verify_fn=None,
        force_rejection_epochs=(), max_epochs=24, verbose=True):
    """Nominal realization with online reoptimization and retained-plan binding.

    force_rejection_epochs injects a verifier outage after actual verification.
    It tests authority restriction, not an adversarial physical disturbance.
    """
    if certified and verify_fn is None:
        from mfd_incident_mpc_certification import verify_command_plan
        verify_fn = verify_command_plan
    library = model.default_library()  # Same fixed optimization guesses as MPC.
    offered = lambda t: model.arrival(t, amplitude=amplitude, duration=duration)
    x = model.BASE_STATE.copy()
    times, states, commands, decisions = [0.], [x.copy()], [], []
    total = queue = departures = arrivals = 0.
    minimum = np.inf
    previous = None
    last_command = None
    stopped = None
    force_rejection_epochs = set(force_rejection_epochs)
    for epoch in range(min(max_epochs, 24)):
        exact_now = epoch*PERIOD
        exact_next = exact_now+PERIOD
        exact_end = min(END, exact_now+1)
        # Preserve the baseline's floating numerical clock; proof clocks are exact.
        now = epoch*mpc.DECISION
        began = time.perf_counter()
        selected, report = mpc.solve(x, now, library,
                                     None if previous is None else previous['tail'], offered)
        report.update(epoch=epoch, exact_time_h=str(exact_now),
                      exact_next_decision_h=str(exact_next), initial_state=x.tolist(),
                      certificate=None, certificate_seconds=0.,
                      candidate_admitted=False, forced_verifier_outage=False)
        accepted = False
        if selected is not None:
            candidate = selected[1].reshape(2, 4).copy()
            lengths = [min(PERIOD, exact_end-exact_now),
                       max(Fraction(0), exact_end-exact_now-PERIOD)]
            nonzero = [(u, length) for u, length in zip(candidate, lengths) if length > 0]
            report['candidate_commands'] = candidate.tolist()
            if certified:
                proof_started = time.perf_counter()
                cert = verify_fn(x, [u for u, _ in nonzero], [d for _, d in nonzero],
                                 initial_time_h=exact_now, amplitude=amplitude, duration=duration,
                                 initial_box=INITIAL_BOX, residual_bound=RESIDUAL_BOUND,
                                 step_seconds=15.)
                report['certificate_seconds'] = time.perf_counter()-proof_started
                report['certificate'] = cert
                accepted = bool(cert['accepted'])
                if epoch in force_rejection_epochs:
                    report['forced_verifier_outage'] = True
                    accepted = False
            else:
                accepted = True
            report['candidate_admitted'] = accepted
            if accepted:
                command = candidate[0].copy()
                previous = dict(tail=candidate[1].copy(), expires=exact_end,
                                origin=epoch, proof_id=report['certificate']['proof_id'] if certified else None)
                report['execution_source'] = 'verified_mpc_replacement' if certified else 'nominal_mpc_replacement'
        if not accepted:
            if previous is not None and previous['expires'] >= exact_next:
                command = previous['tail'].copy()
                report['execution_source'] = 'retained_certified_suffix' if certified else 'retained_nominal_suffix'
            else:
                report.update(execution_source='no_authorized_initial_plan_or_suffix',
                              executed_constant_command=None, executed_end_time_h=float(exact_now),
                              total_decision_seconds=time.perf_counter()-began)
                stopped = dict(exact_time_h=str(exact_now), reason='unresolved_continuation')
                decisions.append(report)
                break
        report.update(executed_constant_command=command.tolist(),
                      first_command_changed=last_command is None or not np.array_equal(command, last_command),
                      retained_plan_origin_epoch=previous['origin'],
                      retained_plan_exact_expiry_h=str(previous['expires']),
                      retained_plan_proof_id=previous['proof_id'],
                      total_decision_seconds=time.perf_counter()-began)
        last_command = command.copy()
        controller = lambda _x, u=command: u
        if verbose:
            print(f"a={amplitude:.2f} d={duration:.2f} epoch={epoch:02d} "
                  f"admitted={accepted} source={report['execution_source']} "
                  f"solve_s={report['solve_seconds']:.3f} proof_s={report['certificate_seconds']:.3f}", flush=True)
        for step in range(round(mpc.DECISION/mpc.EXECUTION_STEP)):
            t = now+step*mpc.EXECUTION_STEP
            margin = float(min(model.margins(x, command)))
            minimum = min(minimum, margin)
            if margin < -1e-6 or not np.all(np.isfinite(x)):
                stopped = dict(time_h=t, reason='physical_constraint_violation', margin=margin)
                break
            new = model.rk4(x, controller, t, mpc.EXECUTION_STEP, offered)
            new_margin = float(min(model.margins(new, command)))
            minimum = min(minimum, new_margin)
            total += mpc.EXECUTION_STEP/2*100*(mpc.COST_ROW@(x+new))
            queue += mpc.EXECUTION_STEP/2*100*(x[6]+new[6])
            departures += mpc.EXECUTION_STEP/2*100*(model.departure_rate(x, command)+model.departure_rate(new, command))
            arrivals += mpc.EXECUTION_STEP/2*100*(sum(model.INTERNAL_DEMAND)*2+offered(t)+offered(t+mpc.EXECUTION_STEP))
            commands.append(command.tolist())
            x = new
            states.append(x.copy())
            times.append(t+mpc.EXECUTION_STEP)
            if new_margin < -1e-6 or not np.all(np.isfinite(x)):
                stopped = dict(time_h=t+mpc.EXECUTION_STEP, reason='physical_constraint_violation', margin=new_margin)
                break
        report.update(executed_final_state=x.tolist(), executed_end_time_h=times[-1])
        decisions.append(report)
        if stopped is not None:
            break
    arrays = np.asarray(states)
    at_deadline = np.flatnonzero(np.isclose(times, mpc.DEADLINE, atol=1e-9, rtol=0))
    deadline_n1 = None if not len(at_deadline) else float(arrays[at_deadline[0], 0])
    complete = stopped is None and abs(times[-1]-float(END)) < 1e-10
    proofs = [d['certificate'] for d in decisions if d['certificate'] is not None]
    decision_times = [d['total_decision_seconds'] for d in decisions]
    metrics = dict(total_time_spent_veh_h=total, queue_time_veh_h=queue,
                   completed_trips_vehicles=departures,
                   maximum_queue_vehicles=100*float(max(arrays[:, 6])),
                   final_queue_vehicles=100*float(x[6]),
                   minimum_physical_margin=float(minimum) if np.isfinite(minimum) else None,
                   deadline_n1=deadline_n1, deadline_pass=deadline_n1 is not None and deadline_n1 <= mpc.TARGET_N1,
                   completed_horizon_h=times[-1], full_horizon_completed=complete, stopped=stopped,
                   conservation_residual_vehicles=100*(mpc.COST_ROW@model.BASE_STATE)+arrivals-100*(mpc.COST_ROW@x)-departures,
                   optimization_epochs=len(decisions), certificate_attempts=len(proofs),
                   successful_certificates=sum(c['accepted'] for c in proofs),
                   failed_certificates=sum(not c['accepted'] for c in proofs),
                   admitted_replacements=sum(d['candidate_admitted'] for d in decisions),
                   first_command_changes_including_initial=sum(d.get('first_command_changed', False) for d in decisions),
                   admitted_plans_with_unchanged_first_command=sum(d['candidate_admitted'] and not d.get('first_command_changed', False) for d in decisions),
                   retained_suffix_epochs=sum('retained_' in d['execution_source'] for d in decisions),
                   forced_verifier_outages=sum(d['forced_verifier_outage'] for d in decisions),
                   mean_total_decision_seconds=float(np.mean(decision_times)),
                   median_total_decision_seconds=float(np.median(decision_times)),
                   maximum_total_decision_seconds=float(max(decision_times)))
    result = dict(amplitude=amplitude, duration_h=duration, configuration=configuration(certified, amplitude, duration),
                  metrics=metrics, decisions=decisions,
                  trajectory=dict(time_h=times, state=arrays.tolist(), command=commands))
    if complete:
        result['independent_replay'] = mpc.independent_replay(decisions, metrics, offered)
    return result


def compare_archived(run, archived):
    if not run['metrics']['full_horizon_completed']:
        return dict(comparable_full_horizon=False)
    ours, theirs = run['trajectory'], archived['trajectory']
    return dict(comparable_full_horizon=True,
                archived_mpc_source_sha256=digest(RESULTS/'mfd_incident_mpc_grid_results.json'),
                maximum_command_difference=float(np.max(abs(np.asarray(ours['command'])-theirs['command']))),
                maximum_state_difference_units=float(np.max(abs(np.asarray(ours['state'])-theirs['state']))),
                total_time_spent_difference_veh_h=run['metrics']['total_time_spent_veh_h']-archived['metrics']['total_time_spent_veh_h'],
                adaptive_tts_difference_veh_h=run['independent_replay']['total_time_spent_veh_h']-archived['independent_replay']['total_time_spent_veh_h'],
                note='Fresh closed-loop solve, not execution of an archived proposal trace.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', action='store_true')
    parser.add_argument('--nominal', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = ((.3, 1.),) if args.primary else operating.DEVELOPMENT_CASES+operating.HOLDOUT_CASES
    output = args.output or RESULTS/f'mfd_incident_mpc_{"nominal_repeat" if args.nominal else "certified"}_{"primary" if args.primary else "grid"}_results.json'
    archived_path = RESULTS/'mfd_incident_mpc_grid_results.json'
    archived = json.loads(archived_path.read_text())['runs']
    payload = dict(scope='Fresh receding-horizon MPC with optional execution certification',
                   protocol=configuration(not args.nominal),
                   source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (
                       Path(__file__), ROOT/'reproducibility/mfd_incident_mpc.py',
                       ROOT/'reproducibility/mfd_incident_verifier.py',
                       ROOT/'reproducibility/mfd_incident_mpc_certification.py',
                       ROOT/'reproducibility/mfd_incident_model.py')},
                   archived_mpc_sha256=digest(archived_path), runs=[])
    for amplitude, duration in cases:
        started = time.perf_counter()
        rec = run(amplitude, duration, certified=not args.nominal)
        reference = next(r for r in archived if r['amplitude'] == amplitude and r['duration_h'] == duration)
        rec['comparison_to_archived_nominal_mpc'] = compare_archived(rec, reference)
        rec['wall_seconds'] = time.perf_counter()-started
        payload['runs'].append(rec)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('CASE', json.dumps(dict(amplitude=amplitude, duration_h=duration,
                                    **rec['metrics'], comparison=rec['comparison_to_archived_nominal_mpc'])), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
