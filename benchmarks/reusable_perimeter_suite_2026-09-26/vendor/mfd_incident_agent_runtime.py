"""Finite-mission queue-aware agent/MPC execution with exact-plan certification.

New experiment; no mutation of the archived planner, verifier or study files.
Live semantic requests can be replayed through nominal MPC on its own state.
Such replay is a matched-request comparison, not a second live feedback policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import mfd_incident_model as model
import mfd_incident_mpc as base
from mfd_incident_service_frontier import solve_service
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_verifier import contains_information_box
from audit_mfd_incident_model import reference_field

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reproducibility/results'
PERIOD = Fraction(1, 12)
END = Fraction(1)
ROWS = {'network': [1.,1.,0,0,0,0,1.], 'region_1': [4.,1.,0,0,0,0,1.],
        'region_2': [1.,4.,0,0,0,0,1.]}
GOLD = [('region_1',60),('region_1',60),('network',30),('region_2',90),
        ('region_2',90),('region_1',5),('region_2',90),('network',30)]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def request(objective, duration=60):
    return dict(objective=objective, duration_minutes=duration, cost_row=ROWS[objective])


def source_hashes():
    names = ['reproducibility/mfd_incident_agent_runtime.py',
             'reproducibility/mfd_incident_service_frontier.py',
             'reproducibility/mfd_incident_model.py', 'reproducibility/mfd_incident_mpc.py',
             'reproducibility/mfd_incident_verifier.py',
             'reproducibility/mfd_incident_mpc_certification.py',
             'OPERATING_RANGE_AGENT_STUDY_PROTOCOL.md', 'QUEUE_AGENT_PROTOCOL.md',
             'reproducibility/mfd_incident_live_agent.py']
    hashes = {}
    for name in names:
        source = ROOT/name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target = RESULTS/'ors'/digest[:16]/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            if os.name=='nt':
                shutil.copy2('\\\\?\\'+str(source.resolve()),'\\\\?\\'+str(target.resolve()))
            else:
                shutil.copy2(source,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        hashes[name]=digest
    return hashes


def run(*, certified=True, provider='oracle', requests=None, session=None,
        box_vehicles=1., residual_veh_h=60., service_vehicles=2850., checkpoint=None):
    """Nominal realization; enlarged contracts are fixed before initialization."""
    widths = np.full(7, box_vehicles/100.)
    widths[6] = min(box_vehicles, 10.)/100.
    offered = lambda t: model.arrival(t, amplitude=.3, duration=1.)
    library = model.default_library()
    x = model.BASE_STATE.copy()
    times, states, commands, decisions = [0.], [x.tolist()], [], []
    total = queue = departures = arrivals = 0.
    previous = None
    last_command = None
    last_request = request('network')
    priority_expiry = END
    feedback = {'status':'initialization_pending'}
    stopped = None
    config = dict(provider=provider, certified=certified, mission_minutes=60,
                  proposal_period_minutes=5, live_minutes=list(range(5,41,5)),
                  state_half_widths_vehicles=(100*widths).tolist(), residual_veh_h=residual_veh_h,
                  service_vehicles=service_vehicles, service_minute=60,
                  forecast='known primary30pct1h; original mean/ramp',
                  activation='synchronous; computation timing recorded separately',
                  cost_rows=ROWS, priority_duration='requested row then network row',
                  nominal_replay_scope='same recorded requests; independent state/reoptimization',
                  independent_contract=True, forced_outages=False)
    payload = dict(configuration=config, source_sha256=source_hashes(), decisions=decisions)
    for epoch in range(12):
        now, following = epoch*PERIOD, (epoch+1)*PERIOD
        began = time.perf_counter()
        response = None
        valid = True
        if epoch == 0:
            req = request('network')
        elif requests is not None:
            if epoch >= len(requests):
                stopped = dict(exact_time_h=str(now),reason='matched_request_record_ended')
                break
            item = requests[epoch]
            req, valid = item.get('tool_request'), item.get('request_valid', True)
        elif 1 <= epoch <= 8 and session is not None:
            response = session.propose(epoch-1, float(now)*60,
                {'region_1_vehicles':100*x[0], 'region_2_vehicles':100*x[1],
                 'exterior_queue_vehicles':100*x[6]}, feedback)
            valid = bool(response['valid'])
            req = response.get('tool_request')
        elif 1 <= epoch <= 8 and provider == 'oracle':
            req = request(*GOLD[epoch-1])
        elif provider == 'network':
            req = request('network')
        else:
            remaining = max(0, round(float(priority_expiry-now)*60))
            req = request(last_request['objective'], remaining) if remaining else request('network')
        if valid and req is not None:
            objective = req['objective']
            # Support a conventional balanced alias without accepting arbitrary weights.
            objective = 'network' if objective == 'balanced' else objective
            if objective not in ROWS:
                raise ValueError(f'Unrecognized structured objective: {objective}')
            req = request(objective, int(req['duration_minutes']))
            last_request = req.copy()
            priority_expiry = now+Fraction(req['duration_minutes'],60)
        else:
            req = None
        event = dict(epoch=epoch, exact_time_h=str(now), exact_next_decision_h=str(following),
                     time_h=float(now), initial_state=x.tolist(), tool_request=req,
                     request_valid=valid, live_response=response, certificate=None,
                     candidate_admitted=False, candidate_commands=None, solve_report=None,
                     proof_seconds=0., failure_reason=None)
        selected = None
        if valid:
            selected, report = solve_service(x,float(now),library,
                None if previous is None else previous['tail'],offered,
                service_target_vehicles=service_vehicles,deadline_h=1.,horizon_h=float(END-now),
                cost_row=ROWS[req['objective']],priority_minutes=req['duration_minutes'])
            event['solve_report'] = report
        accepted = False
        if selected is not None:
            candidate = selected[1].reshape(2,4).copy()
            lengths = [min(PERIOD,END-now), max(Fraction(0),END-now-PERIOD)]
            nonzero = [(u,d) for u,d in zip(candidate,lengths) if d>0]
            event['candidate_commands'] = [u.tolist() for u,d in nonzero]
            if certified or epoch == 0:
                pstart = time.perf_counter()
                cert = verify_command_plan(x,[u for u,d in nonzero],[d for u,d in nonzero],
                    initial_time_h=now,amplitude=.3,duration=1.,initial_box=widths,
                    residual_bound=residual_veh_h/100.,step_seconds=15.,
                    service_checks=[dict(time_h=END,coordinate=0,upper=service_vehicles/100.)])
                event['certificate'] = cert
                event['proof_seconds'] = time.perf_counter()-pstart
                accepted = bool(cert['accepted'])
                if not accepted:
                    event['failure_reason'] = [f['kind'] for f in cert['failures']]
            else:
                accepted = True
            if accepted:
                command = candidate[0].copy()
                previous = dict(tail=candidate[1].copy(),origin=epoch,expiry=END,
                                certificate=event['certificate'])
                event['execution_source'] = 'verified_replacement' if certified else 'nominal_replacement'
        else:
            event['failure_reason'] = 'invalid_request' if not valid else 'optimizer_no_feasible_solution'
        event['candidate_admitted'] = accepted
        if not accepted:
            if previous is not None and previous['expiry']>=following:
                command = previous['tail'].copy()
                event['execution_source'] = 'retained_certified_suffix' if certified else 'retained_nominal_suffix'
            else:
                stopped = dict(exact_time_h=str(now),reason='no_verified_initial_plan_or_continuation')
                event.update(execution_source='unresolved',executed_constant_command=None,
                             executed_end_time_h=float(now))
                if session is not None and 1<=epoch<=8:
                    session.record_execution(epoch-1,dict(accepted=False,status='unresolved',
                        reason=event['failure_reason'],time_minutes=float(now)*60,
                        remaining_validity_minutes=0,retained_previous_plan=False))
                decisions.append(event)
                break
        event.update(executed_constant_command=command.tolist(),
                     first_command_changed=last_command is None or not np.array_equal(last_command,command),
                     origin_epoch=previous['origin'],original_expiry_h=str(previous['expiry']),
                     execution_proof_id=None if previous['certificate'] is None else previous['certificate']['proof_id'])
        last_command = command.copy()
        controller = lambda _x,u=command:u
        minimum_margin = np.inf
        for step in range(30):
            t = float(now)+step*base.EXECUTION_STEP
            new = model.rk4(x,controller,t,base.EXECUTION_STEP,offered)
            minimum_margin = min(minimum_margin,float(min(model.margins(x,command))),
                                 float(min(model.margins(new,command))))
            total += base.EXECUTION_STEP/2*100*(base.COST_ROW@(x+new))
            queue += base.EXECUTION_STEP/2*100*(x[6]+new[6])
            departures += base.EXECUTION_STEP/2*100*(model.departure_rate(x,command)+model.departure_rate(new,command))
            arrivals += base.EXECUTION_STEP/2*100*(2*sum(model.INTERNAL_DEMAND)+offered(t)+offered(t+base.EXECUTION_STEP))
            x = new
            states.append(x.tolist()); times.append(t+base.EXECUTION_STEP); commands.append(command.tolist())
        event.update(executed_final_state=x.tolist(),executed_end_time_h=float(following),
                     sampled_physical_margin=minimum_margin,wall_seconds=time.perf_counter()-began)
        if certified:
            parent = previous['certificate']
            shapes = {Fraction(c['exact_time_fraction']):c for c in parent['reachable_checkpoints']}
            membership = contains_information_box(shapes[following],x,0.)
            event['endpoint_membership'] = membership
            if not membership['passed']:
                raise AssertionError(('sampled trajectory outside proof',epoch,membership))
        feedback = dict(status='accepted' if accepted else 'retained', accepted=accepted,
                        reason=event['failure_reason'],retained_previous_plan=not accepted,
                        time_minutes=float(following)*60,
                        remaining_validity_minutes=float(END-following)*60 if certified else None,
                        cumulative_tts_vehicle_hours=total,queue_vehicles=100*x[6],
                        executed_gate_command=command.tolist(),
                        region_1_vehicles=100*x[0],region_2_vehicles=100*x[1],
                        exterior_queue_vehicles=100*x[6],
                        certified_until_minutes=60 if certified else None)
        event['execution_feedback'] = feedback.copy()
        if session is not None and 1<=epoch<=8:
            session.record_execution(epoch-1,feedback)
        decisions.append(event)
        print(json.dumps(dict(epoch=epoch,provider=provider,certified=certified,
            objective=None if req is None else req['objective'],accepted=accepted,
            reason=event['failure_reason'],n1veh=round(100*x[0],2),
            seconds=round(event['wall_seconds'],2))),flush=True)
        if checkpoint:
            save(checkpoint,payload)
    complete = abs(times[-1]-1.)<1e-9 and stopped is None
    payload.update(metrics=dict(complete=complete,completed_horizon_h=times[-1],stopped=stopped,
        tts_veh_h=total,queue_veh_h=queue,departures_vehicles=departures,
        max_queue_vehicles=100*max(s[6] for s in states),deadline_vehicles=100*x[0] if complete else None,
        service_pass=bool(complete and 100*x[0]<=service_vehicles),
        conservation_residual_vehicles=6200+arrivals-100*sum(x[[0,1,6]])-departures,
        accepted_plans=sum(d['candidate_admitted'] for d in decisions),
        rejected_proposals=sum(not d['candidate_admitted'] for d in decisions),
        rejected_certificates=sum(d['certificate'] is not None and not d['certificate']['accepted'] for d in decisions),
        retained_epochs=sum(d['execution_source'].startswith('retained') for d in decisions),
        first_command_changes=sum(d.get('first_command_changed',False) for d in decisions)),
        trajectory=dict(time_h=times,state=states,command=commands))
    payload['independent_audit'] = audit(payload)
    return payload


def audit(case):
    """Reconstruct accepted command binding; independently replay class balances."""
    parents = {}
    y = np.r_[model.BASE_STATE,np.zeros(7)]
    maximum_error = 0.
    minimum = np.inf
    for event in case['decisions']:
        command = event.get('executed_constant_command')
        if command is None: continue
        now,following = Fraction(event['exact_time_h']),Fraction(event['exact_next_decision_h'])
        cert = event['certificate']
        if event['candidate_admitted'] and cert is not None:
            desc=cert['proof_descriptor']
            assert hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest()==cert['proof_id']
            assert cert['accepted'] and cert['verified_continuous_physical'] and cert['verified_scheduled_service']
            assert desc['initial_state']==event['initial_state']
            assert desc['residual_bound']==case['configuration']['residual_veh_h']/100.
            assert desc['service_checks']==[dict(exact_time='1',coordinate=0,upper=case['configuration']['service_vehicles']/100.)]
            assert cert['adapter_audit']['commands']==event['candidate_commands']
            assert Fraction(desc['initial_time'])+sum(map(Fraction,desc['durations']))==END
            assert min(cert['continuous_constraint_margins_lower'].values())>=0
            parents[event['epoch']]=event
        if case['configuration']['certified']:
            origin=parents[event['origin_epoch']]
            parent=origin['certificate']
            assert parent['proof_id']==event['execution_proof_id']
            assert event['original_expiry_h']=='1' and following<=END
            expected=origin['candidate_commands'][0] if origin['epoch']==event['epoch'] else origin['candidate_commands'][1]
            assert command==expected
            assert event['endpoint_membership']['passed']
        u=np.array(command)
        true_objective = GOLD[event['epoch']-1][0] if 1<=event['epoch']<=8 else 'network'
        true_row = np.array(ROWS[true_objective])
        def rhs(t,z):
            offered=model.arrival(t,amplitude=.3,duration=1.)
            dx,out=reference_field(z[:7],u,offered)
            return np.r_[dx,100*sum(z[[0,1,6]]),100*z[6],100*out,100*(sum(model.INTERNAL_DEMAND)+offered),
                         100*z[0],100*z[1],100*(true_row@z[:7])]
        sol=solve_ivp(rhs,(float(now),float(following)),y,method='DOP853',rtol=1e-11,atol=1e-12,
                      max_step=1/360,dense_output=True)
        assert sol.success
        samples=sol.sol(np.linspace(float(now),float(following),31))[:7].T
        saved=np.array(case['trajectory']['state'][30*event['epoch']:30*(event['epoch']+1)+1])
        maximum_error=max(maximum_error,float(np.max(abs(samples-saved))))
        for state in samples: minimum=min(minimum,float(min(model.margins(state,u))))
        y=sol.y[:,-1]
    balance=6200+y[10]-100*sum(y[[0,1,6]])-y[9]
    assert abs(balance)<1e-6 and maximum_error<2e-4
    assert abs(y[7]-case['metrics']['tts_veh_h'])<.1
    return dict(passed=True,independent_class_balance=True,accepted_bindings=len(parents),
                maximum_state_error_vehicles=100*maximum_error,
                conservation_residual_vehicles=float(balance),minimum_sampled_margin=float(minimum) if np.isfinite(minimum) else None,
                tts_veh_h=float(y[7]),queue_veh_h=float(y[8]),departures_vehicles=float(y[9]),
                region_1_veh_h=float(y[11]),region_2_veh_h=float(y[12]),
                true_priority_weighted_veh_h=float(y[13]),
                deadline_vehicles=100*float(y[0]) if case['metrics']['complete'] else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--provider',choices=['oracle','network','live'],default='oracle')
    parser.add_argument('--box',type=float,default=1.)
    parser.add_argument('--residual',type=float,default=60.)
    parser.add_argument('--service',type=float,default=2850.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace',type=int,default=0)
    parser.add_argument('--memoryless',action='store_true')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    session=None
    if args.provider=='live':
        if (args.box,args.residual,args.service)!=(1.,60.,2850.):
            raise ValueError('The frozen live protocol requires box1, residual60, service2850')
        from mfd_incident_live_agent import QueueAgentSession
        source_hashes()  # Complete filesystem preflight before creating live-session metadata.
        session=QueueAgentSession(ROOT/'live_llm_results/incident_agent/queue_semantic_agent_v1',
                                  variant='memoryless' if args.memoryless else 'stateful',
                                  paraphrase=args.trace,resume_empty=True)
    certified=run(certified=True,provider=args.provider,session=session,
                  box_vehicles=args.box,residual_veh_h=args.residual,service_vehicles=args.service,
                  checkpoint=args.output.with_suffix('.partial.json'))
    matched=run(certified=False,provider='matched_request_replay',requests=certified['decisions'],
                box_vehicles=args.box,residual_veh_h=args.residual,service_vehicles=args.service)
    complete=certified['metrics']['complete'] and matched['metrics']['complete']
    difference=None if not complete else certified['independent_audit']['tts_veh_h']-matched['independent_audit']['tts_veh_h']
    out=dict(protocol='OPERATING_RANGE_AGENT_STUDY_PROTOCOL.md',certified=certified,nominal_matched_requests=matched,
             comparison=dict(matched_complete_horizon=complete,incremental_tts_veh_h=difference,
                             incremental_tts_percent=None if difference is None else 100*difference/matched['independent_audit']['tts_veh_h']))
    save(args.output,out)
    print(json.dumps(out['comparison']),flush=True)


if __name__=='__main__': main()
