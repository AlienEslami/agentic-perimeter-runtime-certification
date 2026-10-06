"""Matched two-block finite-library economic selector for the queue incident.

Numerical prediction is not a certificate. The optional verifier is independently
responsible for continuous-time physical and service guarantees.
"""
from __future__ import annotations
import argparse
import itertools
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
import mfd_incident_model as m

PERIOD=1/12
EXACT_PERIOD=Fraction(1,12)
HORIZON=1.0
END=2.0
SERVICE_TIME=1.0
SERVICE_UPPER=28.5


def batch_field(x,u,arrival):
    n1,n2,n11,n22,n10,n20,z=x.T
    u12,u21,u02,u20=u.T
    h1=.0014877*n1*n1-.29815*n1+15.0912
    h2=.0014877*n2*n2-.29815*n2+15.0912
    q11=n11*h1;q12=(n1-n11-n10)*h1;q10=n10*h1
    q22=n22*h2;q21=(n2-n22-n20)*h2;q20=n20*h2
    admission=np.minimum(60*z,120*u02)
    d11,d12,d10,d22,d21,d20=m.INTERNAL_DEMAND
    return np.stack((-q11-(q12+q10)*u12+q21*u21+d11+d12+d10,
                     -q22+(q12+q10)*u12-q21*u21+admission-q20*u20+d22+d21+d20,
                     -q11+q21*u21+d11,-q22+q12*u12+.5*admission+d22,
                     -q10*u12+d10,-q20*u20+q10*u12+d20,arrival-admission),axis=1)


def batch_margins(x,u):
    return np.column_stack((x,x[:,0]-x[:,2]-x[:,4],x[:,1]-x[:,3]-x[:,5],
                            100-x[:,0],100-x[:,1],20-x[:,6],u,1-u,
                            .75-u[:,1]+u[:,0],.75+u[:,1]-u[:,0],
                            .75-u[:,3]+u[:,2],.75+u[:,3]-u[:,2]))


def predict_candidates(x,t,library,amplitude,duration,horizon=None,prediction_step=1/60):
    horizon=min(HORIZON,END-t) if horizon is None else horizon
    first=min(PERIOD,horizon)
    pairs=list(itertools.product(range(len(library)),repeat=2)) if horizon>PERIOD+1e-10 else [(k,k) for k in range(len(library))]
    gains=np.array([mode.gain for mode in library]); offsets=np.array([mode.feedforward-mode.gain@mode.center for mode in library])
    states=np.tile(x,(len(pairs),1)); costs=np.zeros(len(pairs)); minimum=np.full(len(pairs),np.inf)
    service=np.full(len(pairs),np.nan)
    cuts=[t,t+horizon,t+first]
    cuts.extend(z for z in (SERVICE_TIME,.25,.25+1/12,.25+duration-1/12,.25+duration) if t<z<t+horizon)
    grid=sorted(set([*cuts,*np.arange(t,t+horizon-1e-10,prediction_step)]))
    pairs=np.array(pairs)
    for left,right in zip(grid[:-1],grid[1:]):
        h=right-left; ids=pairs[:,0 if left<t+first-1e-10 else 1]
        def command(y):return offsets[ids]+np.einsum('nij,nj->ni',gains[ids],y)
        def field(y,s):return batch_field(y,command(y),m.arrival(s,amplitude,duration))
        minimum=np.minimum(minimum,np.min(batch_margins(states,command(states)),axis=1))
        k1=field(states,left);k2=field(states+h*k1/2,left+h/2);k3=field(states+h*k2/2,left+h/2);k4=field(states+h*k3,right)
        new=states+h*(k1+2*k2+2*k3+k4)/6
        costs+=100*h/2*(np.sum(states[:,[0,1,6]],axis=1)+np.sum(new[:,[0,1,6]],axis=1))
        minimum=np.minimum(minimum,np.min(batch_margins(new,command(new)),axis=1))
        states=new
        if abs(right-SERVICE_TIME)<1e-9:service=states[:,0].copy()
    records=[]
    for k,pair in enumerate(pairs):
        feasible=minimum[k]>=-1e-8 and (not np.isfinite(service[k]) or service[k]<=SERVICE_UPPER+1e-8)
        records.append(dict(pair=pair.tolist(),durations_h=[first,horizon-first],predicted_cost_veh_h=float(costs[k]),
                            minimum_predicted_margin=float(minimum[k]),
                            predicted_service_n1=float(service[k]) if np.isfinite(service[k]) else None,
                            nominal_feasible=bool(feasible),terminal_state=states[k].tolist()))
    return sorted(records,key=lambda z:z['predicted_cost_veh_h'])


def simulate(amplitude=.3,duration=1.,certified=False,residual_pattern=None,initial_error=None,prediction_step=1/60,
             observer_box=.01,residual_bound=.6,verify_step_seconds=15,library=None):
    library=m.default_library() if library is None else list(library)
    x=m.BASE_STATE.copy()
    if initial_error is not None:x+=initial_error
    states=[x.copy()];times=[0.];events=[];u_records=[];queue=tts=departures=0.;minimum=np.inf
    decision=0;previous=-1;stored_mode=None;stored_until=0.;fallbacks=0;verification_failures=0;all_verified=certified
    previous_certificate=None;stored_certificate=None;stored_event=None;stopped_infeasible=False
    stored_until_exact=Fraction(0)
    library_records=[mode.record() for mode in library]
    library_hash=hashlib.sha256(json.dumps(library_records,sort_keys=True).encode()).hexdigest()
    if certified:
        from mfd_incident_verifier import PlanVerifier
        verifier=PlanVerifier(library,metric_strategy='conservation_box')
        verify_plan=verifier.verify_plan
    for k in range(round(END/PERIOD)):
        exact_time=k*EXACT_PERIOD
        t=float(exact_time);tic=time.perf_counter()
        records=predict_candidates(x,t,library,amplitude,duration,prediction_step=prediction_step)
        prediction_seconds=time.perf_counter()-tic;selected=None;attempts=[]
        for candidate in records:
            if not candidate['nominal_feasible']:continue
            if not certified:selected=candidate;break
            pair=candidate['pair'];ds=candidate['durations_h']
            # Preserve the exact installed endpoint, including a deadline at1h.
            exact_start=exact_time
            exact_end=min(Fraction(2),exact_start+Fraction(1))
            first_length=min(EXACT_PERIOD,exact_end-exact_start)
            exact_lengths=[first_length,exact_end-exact_start-first_length]
            nonzero=[(q,interval) for q,interval in zip(pair,exact_lengths) if interval>0]
            services=[dict(time_h=SERVICE_TIME,coordinate=0,upper=SERVICE_UPPER)] if t<SERVICE_TIME-1e-10 and SERVICE_TIME<=t+sum(ds)+1e-10 else []
            future_cuts=[j*EXACT_PERIOD for j in range(k+1,25) if j*EXACT_PERIOD<=exact_end]
            cert=verify_plan(x,[library[q] for q,_ in nonzero],[interval for _,interval in nonzero],initial_time_h=exact_start,amplitude=amplitude,duration=duration,
                             residual_bound=residual_bound,initial_box=observer_box,step_seconds=verify_step_seconds,
                             service_checks=services,timecuts_h=future_cuts)
            passed=bool(cert.get('accepted',cert.get('verified',cert.get('verified_continuous',False))))
            attempts.append(dict(pair=pair,passed=passed,certificate=cert))
            if passed:selected=candidate;previous_certificate=cert;break
            verification_failures+=1
        if selected is None:
            fallbacks+=1
            if stored_mode is not None and stored_until_exact>=exact_time+EXACT_PERIOD:
                decision=stored_mode;reason='retained_previous_plan'
            else:
                if certified:
                    all_verified=False;stopped_infeasible=True
                    events.append(dict(time_h=t,reason='no_verified_initial_plan_or_suffix',nominal_candidates=records,
                                       verification_attempts=attempts,executed_mode=None))
                    break
                decision=1 if t<SERVICE_TIME else 0;reason='nominal_search_infeasible'
        else:
            decision=selected['pair'][0];stored_mode=selected['pair'][1];stored_until=t+sum(selected['durations_h'])
            stored_until_exact=min(Fraction(2),exact_time+Fraction(1));stored_until=float(stored_until_exact)
            if certified:
                stored_certificate=previous_certificate;stored_event=k
            reason='verified_replacement' if certified else 'nominal_optimizer'
        event=dict(time_h=t,executed_mode=decision,previous_mode=previous,changed=bool(decision!=previous),reason=reason,
                   exact_time_h=str(exact_time),exact_next_decision_h=str(exact_time+EXACT_PERIOD),
                   nominal_candidates=records,verification_attempts=attempts,prediction_seconds=prediction_seconds,
                   library_sha256=library_hash,retained_plan_origin_event=stored_event,
                   retained_plan_valid_until_h=stored_until,
                   retained_plan_exact_expiry_h=str(stored_until_exact),
                   retained_plan_certificate_verified=bool(stored_certificate and stored_certificate.get('accepted')) if certified else None,
                   total_decision_seconds=time.perf_counter()-tic)
        events.append(event);previous=decision;mode=library[decision]
        h=1/360
        for j in range(round(PERIOD/h)):
            left=float(exact_time+Fraction(j,360));u=mode.command(x);minimum=min(minimum,float(min(m.margins(x,u))))
            def offered(s):return m.arrival(s,amplitude,duration)+(0 if residual_pattern is None else residual_pattern(s))
            new=m.rk4(x,mode.command,left,h,offered)
            tts+=100*h/2*(sum(x[[0,1,6]])+sum(new[[0,1,6]]))
            queue+=100*h/2*(x[6]+new[6])
            departures+=100*h/2*(m.departure_rate(x,u)+m.departure_rate(new,mode.command(new)))
            x=new;times.append(left+h);states.append(x.copy());u_records.append(u.tolist())
    minimum=min(minimum,float(min(m.margins(x,library[decision].command(x)))))
    trajectory=np.array(states);service_index=int(np.argmin(abs(np.array(times)-SERVICE_TIME)))
    completed=not stopped_infeasible
    return dict(amplitude=amplitude,duration_h=duration,certified=certified,all_decisions_certified=all_verified,
                full_horizon_completed=completed,stopped_infeasible=stopped_infeasible,
                controller_library=library_records,library_sha256=library_hash,
                activation_source='balanced queue-aware controller at regional target(30,30); initial installation counts as a change',
                model_parameters=dict(internal_demand=m.INTERNAL_DEMAND.tolist(),boundary_capacity=m.CAPACITY,
                                      release_rate=m.RELEASE_RATE,base_arrival=m.BASE_ARRIVAL,queue_limit=m.QUEUE_LIMIT),
                total_time_spent_veh_h=tts,queue_time_veh_h=queue,departures=departures,
                service_n1_vehicles=100*trajectory[service_index,0],service_target_vehicles=100*SERVICE_UPPER,
                service_pass=bool(times[-1]>=SERVICE_TIME and trajectory[service_index,0]<=SERVICE_UPPER+1e-8),
                maximum_queue_vehicles=100*max(trajectory[:,6]),final_queue_vehicles=100*x[6],
                minimum_physical_margin=minimum,changes=sum(e.get('changed',False) for e in events),fallbacks=fallbacks,
                rejected_certificates=verification_failures,events=events,times=times,states=trajectory.tolist(),commands=u_records)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--certified',action='store_true');parser.add_argument('--grid',action='store_true')
    args=parser.parse_args();cases=list(itertools.product((.2,.3,.4),(.5,1.))) if args.grid else [(.3,1.)]
    runs=[]
    for a,d in cases:
        rec=simulate(a,d,args.certified);runs.append(rec)
        print({key:rec[key] for key in ('amplitude','duration_h','total_time_spent_veh_h','queue_time_veh_h','maximum_queue_vehicles','service_n1_vehicles','service_pass','changes','fallbacks','minimum_physical_margin')},flush=True)
    stem='certified' if args.certified else 'nominal'
    out=m.ROOT/f'reproducibility/results/mfd_incident_library_{stem}_results.json'
    out.write_text(json.dumps(dict(protocol='two-block finite-library selector; all failed candidates retained',runs=runs),indent=2)+'\n')
