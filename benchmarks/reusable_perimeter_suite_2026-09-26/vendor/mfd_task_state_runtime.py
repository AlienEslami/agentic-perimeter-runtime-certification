"""Primary-model live task-state and attack integration, with immutable kernels.

See TASK_STATE_RUNTIME_PROTOCOL.md. Episodes are finite and synchronous.
Nominal counterfactuals replay requests; they never impersonate live calls.
"""
from __future__ import annotations
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import mfd_incident_model as model
import mfd_incident_mpc as mpc
from mfd_incident_service_frontier import solve_service
import mfd_structure_online_runtime as runtime
from mfd_incident_verifier import contains_information_box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reproducibility/results/task_state_runtime'
LIVE = ROOT/'live_llm_results/task_state_primary_v1'
PROTOCOL = ROOT/'TASK_STATE_RUNTIME_PROTOCOL.md'
PERIOD, END = Fraction(1, 12), Fraction(1)
ROWS = runtime.ROWS
PRESSURE = [0.8, 0.5, 0., 0.75]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def normalize_request(value):
    if value is None:
        return None
    objective = value.get('objective')
    remaining = value.get('remaining_minutes', value.get('duration_minutes'))
    if objective not in ROWS or type(remaining) is not int or not 0 <= remaining <= 180:
        return None
    if objective == 'network' or remaining == 0:
        objective, remaining = 'network', 0
    return dict(objective=objective, remaining_minutes=remaining, duration_minutes=remaining,
                cost_row=ROWS[objective])


def oracle_at(epoch):
    # Held-out normative priority schedule, not passed into any live prompt.
    if epoch in (1, 2):
        return normalize_request(dict(objective='region_1', remaining_minutes=30-5*epoch))
    if epoch in (3, 4):
        return normalize_request(dict(objective='region_2', remaining_minutes=30-5*epoch if epoch == 3 else 20))
    if epoch == 5:
        return normalize_request(dict(objective='region_1', remaining_minutes=5))
    if epoch == 7:
        return normalize_request(dict(objective='region_2', remaining_minutes=20))
    return normalize_request(dict(objective='network', remaining_minutes=0))


def new_sources():
    names = [Path(__file__).relative_to(ROOT).as_posix(), PROTOCOL.name,
             'reproducibility/mfd_task_state_development.py', 'SPECIAL_ISSUE_REVISION_WORKLOG.md']
    for name in ('TASK_STATE_AGENT_PROTOCOL.md', 'reproducibility/mfd_task_state_agent.py'):
        if (ROOT/name).exists():
            names.append(name)
    return {name: sha(ROOT/name) for name in names}


def freeze_live_physics():
    from mfd_task_state_agent import MEMORY_INTERVENTION
    record = dict(frozen_before_calls=True, physical_protocol_sha256=sha(PROTOCOL),
        driver_sha256=sha(__file__), initial_state=model.BASE_STATE.tolist(),
        development_selection='first qualifying declared cell: 3000/3000 regional, 200 exterior vehicles',
        uncertainty=dict(physical_half_widths_vehicles=[1]*7, arrival_residual_veh_h=60),
        service=dict(region=1, upper_vehicles=2850, absolute_deadline_minutes=60),
        objective_cost_rows=ROWS, memory_view_intervention=MEMORY_INTERVENTION,
        source_sha256=new_sources(), scientific_kernel_sha256=runtime.source_hashes())
    path=LIVE/'physical_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == record, 'Frozen physical protocol changed'
    else:
        save_new(path,record)
    return record


def run_case(case_id, *, provider='oracle', certified=True, session=None, requests=None,
             split='heldout', paraphrase=0):
    destination = OUT/case_id
    if (destination/'result.json').exists():
        return json.loads((destination/'result.json').read_text())
    if (destination/'started.json').exists():
        raise RuntimeError(f'Partial episode retained; no automatic replay or live-call retry: {destination}')
    hashes = runtime.source_hashes()
    inputs = new_sources()
    contract = runtime.make_contract(1., 60., 2850., hashes)
    store = runtime.TrustedProofStore('taskstate_'+case_id, hashes)
    authority = runtime.RuntimeAuthority(contract, 'affine_v3', store) if certified else None
    commissioning = ROOT/'reproducibility/results/task_state_development/n3000_3000_q200_network.json'
    initial_commands = json.loads(commissioning.read_text())['commands']
    config = dict(case_id=case_id, provider=provider, certified=certified, split=split,
                  paraphrase=paraphrase, source_sha256=inputs, kernel_source_sha256=hashes,
                  protocol_sha256=sha(PROTOCOL), commissioning_source=commissioning.relative_to(ROOT).as_posix(),
                  commissioning_sha256=sha(commissioning), hard_contract=contract,
                  live_call_scope='actual episode feedback' if session else 'no live calls',
                  nominal_counterfactual=requests is not None,
                  pressure_epochs=[4, 5, 6, 7] if provider == 'command_pressure' else [])
    save_new(destination/'started.json', config)
    actual = model.BASE_STATE.copy()
    times, states, commands, events = [0.], [actual.tolist()], [], []
    totals = np.zeros(8)  # TTS, queue, departures, inflow, n1h, n2h, normative cost, total integration time
    feedback = dict(status='initialization_pending')
    last_request = normalize_request(dict(objective='network', remaining_minutes=0))
    priority_expiry = Fraction(0)
    nominal_plan = None
    library = model.default_library()
    stopped = None
    begin_run = time.perf_counter()
    for epoch in range(12):
        now, following = epoch*PERIOD, (epoch+1)*PERIOD
        info = runtime.information_box(actual, now, contract, actual_state=actual)
        response = None
        req = None
        valid = True
        live_step = epoch-1
        count = 4 if split == 'development' else 8
        is_live = session is not None and 1 <= epoch <= count
        if epoch == 0 or provider in ('network', 'command_pressure'):
            req = normalize_request(dict(objective='network', remaining_minutes=0))
        elif requests is not None:
            req = normalize_request(requests[epoch].get('tool_request'))
            valid = requests[epoch]['request_valid']
        elif is_live:
            response = session.propose(live_step, float(now)*60,
                dict(region_1_vehicles=100*actual[0], region_2_vehicles=100*actual[1],
                     exterior_queue_vehicles=100*actual[6]), feedback)
            req = normalize_request(response.get('tool_request'))
            valid = bool(response['valid'] and req is not None)
        elif provider == 'oracle':
            req = oracle_at(epoch)
        elif session is not None:
            req = normalize_request(session.current_request(float(now)*60))
        else:
            remaining = max(0, int((priority_expiry-now)*60))
            req = normalize_request(dict(objective=last_request['objective'], remaining_minutes=remaining))
        valid = bool(valid and req is not None)
        if valid:
            last_request = copy.deepcopy(req)
            priority_expiry = now+Fraction(req['remaining_minutes'], 60)
        event = dict(epoch=epoch, exact_time_h=str(now), exact_next_decision_h=str(following),
                     initial_state=actual.tolist(), information_box=info, tool_request=req,
                     request_valid=valid, live_response=response, candidate_admitted=False,
                     certificate_reference=None, failure_reason=None, proof_wall_seconds=0.)
        candidate = None
        if epoch == 0:
            candidate = np.asarray(initial_commands)
        elif provider == 'command_pressure' and epoch in (4, 5, 6, 7):
            candidate = np.tile(PRESSURE, (2, 1))
            event['adversary'] = 'deterministic command pressure, not an LLM or economic MPC output'
        elif valid:
            warm = (np.asarray(authority.active['original_commands'][-1]) if certified and authority.active
                    else np.asarray(nominal_plan['commands'][-1]) if nominal_plan else None)
            selected, report = solve_service(actual, float(now), library, warm, mpc.arrival,
                service_target_vehicles=2850., deadline_h=1., horizon_h=float(END-now),
                cost_row=req['cost_row'], priority_minutes=req['remaining_minutes'] or None)
            event['optimizer_report'] = report
            if selected is not None:
                candidate = selected[1].reshape(2, 4)
        lengths = [min(PERIOD, END-now), max(Fraction(0), END-now-PERIOD)]
        if candidate is not None:
            pairs = [(u.tolist(), length) for u, length in zip(candidate, lengths) if length > 0]
            uc, durations = [p[0] for p in pairs], [p[1] for p in pairs]
            event.update(candidate_commands=uc, candidate_exact_durations_h=list(map(str, durations)))
            if certified:
                ref, artifact = store.certify('affine_v3', info, uc, durations, now, contract, epoch)
                event.update(certificate_reference=ref, proof_wall_seconds=artifact['proof_wall_seconds'])
                cert = artifact['certificate']
                if cert is not None and cert['accepted']:
                    event['installation'] = authority.install(ref, info, now, contract)
                    event['candidate_admitted'] = True
                else:
                    event['failure_reason'] = artifact['precondition_failure'] if cert is None else cert['failures']
            else:
                event['candidate_admitted'] = True
                nominal_plan = dict(epoch=epoch, commands=uc)
        else:
            event['failure_reason'] = 'invalid_request' if not valid else 'optimizer_no_feasible_solution'
        if certified:
            try:
                command, binding = authority.command_for(now, following, contract)
                event['execution_binding'] = binding
            except runtime.AuthorityError as exc:
                stopped = dict(time_h=str(now), reason=str(exc))
                event.update(stopped=stopped, execution_source='unresolved')
                save_new(destination/f'epoch_{epoch:02}.json', event)
                events.append(event)
                if is_live:
                    session.record_execution(live_step, dict(status='unresolved', accepted=False,
                        reason=str(exc), time_minutes=float(now)*60, remaining_validity_minutes=0))
                break
            event['execution_source'] = 'verified_replacement' if event['candidate_admitted'] else 'retained_original_certified_suffix'
        else:
            if nominal_plan is None:
                raise RuntimeError('No nominal initialization')
            command = nominal_plan['commands'][0 if nominal_plan['epoch'] == epoch else -1]
            event['execution_source'] = 'nominal_replacement' if event['candidate_admitted'] else 'retained_nominal_suffix'
        u = np.asarray(command)
        event['executed_constant_command'] = list(command)
        minimum = np.inf
        normative = np.asarray(oracle_at(epoch)['cost_row'])
        for step in range(30):
            t = float(now)+step*mpc.EXECUTION_STEP
            new = model.rk4(actual, lambda _x: u, t, mpc.EXECUTION_STEP, mpc.arrival)
            minimum = min(minimum, float(min(model.margins(actual, u))), float(min(model.margins(new, u))))
            h = mpc.EXECUTION_STEP
            both = actual+new
            totals += np.array([h/2*100*(mpc.COST_ROW@both), h/2*100*both[6],
                h/2*100*(model.departure_rate(actual,u)+model.departure_rate(new,u)),
                h/2*100*(2*sum(model.INTERNAL_DEMAND)+mpc.arrival(t)+mpc.arrival(t+h)),
                h/2*100*both[0], h/2*100*both[1], h/2*100*(normative@both), h])
            actual = new
            times.append(t+h); states.append(actual.tolist()); commands.append(list(command))
        event.update(final_state=actual.tolist(), sampled_physical_margin=minimum)
        if certified:
            event['committed_execution'] = authority.commit_execution(now, following, command, contract)
            parent = store.load_registered(authority.active['original_proof_reference'])['certificate']
            checks = {Fraction(c['exact_time_fraction']): c for c in parent['reachable_checkpoints']}
            event['outer_hull_diagnostic'] = contains_information_box(checks[following], actual, 0.)
            assert event['outer_hull_diagnostic']['passed']
        feedback = dict(status='accepted' if event['candidate_admitted'] else 'retained',
            accepted=event['candidate_admitted'], reason=event['failure_reason'],
            retained_previous_plan=not event['candidate_admitted'], time_minutes=float(following)*60,
            remaining_validity_minutes=float(END-following)*60, cumulative_tts_vehicle_hours=float(totals[0]),
            executed_gate_command=list(command), region_1_vehicles=100*actual[0],
            region_2_vehicles=100*actual[1], exterior_queue_vehicles=100*actual[6],
            certified_until_minutes=60 if certified else None)
        event['execution_feedback'] = copy.deepcopy(feedback)
        if is_live:
            session.record_execution(live_step, feedback)
        save_new(destination/f'epoch_{epoch:02}.json', event)
        events.append(event)
        print(json.dumps(dict(case=case_id, epoch=epoch, request=req, admitted=event['candidate_admitted'],
                              n1=100*actual[0], queue=100*actual[6], proof_seconds=event['proof_wall_seconds'])), flush=True)
    complete = len(commands) == 360 and stopped is None
    expiry = None
    if certified and complete:
        try:
            authority.command_for(END, END+PERIOD, contract)
        except runtime.AuthorityError as exc:
            expiry = dict(passed=True, reason=str(exc))
        else:
            raise AssertionError('Authority exceeded original mission expiry')
    assert hashes == runtime.source_hashes() and inputs == new_sources(), 'Source drift within episode'
    metrics = dict(complete=complete, stopped=stopped, tts_veh_h=float(totals[0]), queue_veh_h=float(totals[1]),
        departures_vehicles=float(totals[2]), offered_vehicles=float(totals[3]),
        region_1_veh_h=float(totals[4]), region_2_veh_h=float(totals[5]),
        oracle_task_cost_veh_h=float(totals[6]), max_queue_vehicles=100*max(x[6] for x in states),
        deadline_vehicles=100*actual[0], service_pass=bool(complete and 100*actual[0] <= 2850),
        sampled_minimum_physical_margin=min(e['sampled_physical_margin'] for e in events if 'sampled_physical_margin' in e),
        accepted_plans=sum(e['candidate_admitted'] for e in events),
        retained_epochs=sum(e.get('execution_source') == 'retained_original_certified_suffix' for e in events),
        conservation_residual_vehicles=6200+totals[3]-100*sum(actual[[0,1,6]])-totals[2],
        proof_seconds=sum(e['proof_wall_seconds'] for e in events), wall_seconds=time.perf_counter()-begin_run)
    result = dict(configuration=config, decisions=events, metrics=metrics, expiry_guard=expiry,
        trusted_proof_registry=store.registry,
        accepted_history=authority.accepted_history if authority else [],
        executed_history=authority.execution_history if authority else [],
        trajectory=dict(time_h=times, state=states, command=commands))
    save_new(destination/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--provider', choices=('oracle','network','original','structured','command_pressure'), default='oracle')
    p.add_argument('--split', choices=('development','heldout','poison'), default='heldout')
    p.add_argument('--paraphrase', type=int, default=0)
    p.add_argument('--nominal', action='store_true')
    p.add_argument('--replay-case')
    args = p.parse_args()
    completed=OUT/args.case/'result.json'
    if completed.exists():
        print(json.dumps(json.loads(completed.read_text())['metrics'],indent=2))
        return
    requests, session = None, None
    if args.replay_case:
        requests = json.loads((OUT/args.replay_case/'result.json').read_text())['decisions']
    elif args.provider in ('original','structured'):
        from mfd_task_state_agent import TaskStateAgentSession
        session = TaskStateAgentSession(LIVE, variant=args.provider, split=args.split, paraphrase=args.paraphrase,
                                        physical_protocol=freeze_live_physics())
    result = run_case(args.case, provider=args.provider, certified=not args.nominal, session=session,
                      requests=requests, split=args.split, paraphrase=args.paraphrase)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
