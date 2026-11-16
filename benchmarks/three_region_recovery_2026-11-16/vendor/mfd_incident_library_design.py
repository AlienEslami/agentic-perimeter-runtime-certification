"""Declared economic operating-library commissioning sweep, not certification.

The seven-state plant and its internal demands are unchanged. Reference arrival
is a controller-design parameter; every simulation uses the same offered pulse.
All candidates and failures are retained; raw affine commands are never clipped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from scipy.integrate import solve_ivp

import mfd_incident_model as model

TARGETS = ((18.,20.),(20.,22.),(22.,24.),(24.,26.))
REFERENCE_ARRIVALS = (60.,72.,78.,84.)
PENALTIES = (3000.,30000.,300000.,3000000.)
WEIGHTS = (1.,1.,.1,.1,.1,.1,2.)
AMPLITUDE = .3
DURATION = 1.
HORIZON = 2.
STEP = 1/360
DEADLINE = 1.
SERVICE_N1 = 28.5


def arrival(t):
    return model.arrival(t, amplitude=AMPLITUDE, duration=DURATION)


def synthesize(n1,n2,reference_arrival,penalty):
    center,feedforward=model.equilibrium(n1,n2,arrival=reference_arrival)
    a=model.original.complex_step_jacobian(lambda x:model.field(x,feedforward,reference_arrival),center)
    b=model.original.complex_step_jacobian(lambda u:model.field(center,u,reference_arrival),feedforward)
    q=np.diag(WEIGHTS);r=penalty*np.eye(4)
    s=solve_continuous_are(a,b,q,r)
    gain=-np.linalg.solve(r,b.T@s)
    closed=a+b@gain
    p=solve_continuous_lyapunov(closed.T,-np.eye(7))
    name=f'n{n1:g}_{n2:g}_arrival{reference_arrival:g}_R{penalty:g}'
    return model.Mode(name,center,feedforward,gain,closed,(p+p.T)/2)


def geometry_record(mode,reference_arrival):
    """Exact affine ellipsoid supports; nonlinear flow is NOT verified here."""
    a,b=model.affine_constraints(mode.center,mode.gain,mode.feedforward)
    slack=b-a@mode.center
    inverse=np.linalg.inv(mode.P)
    denominator=np.einsum('ij,jk,ik->i',a,inverse,a)
    levels=np.where(denominator>1e-20,slack*slack/np.maximum(denominator,1e-20),np.inf)
    level=0. if min(slack)<0 else float(min(levels))
    # Smooth capacity-limited admission requires C*u02 <= release_rate*z.
    branch_a=model.CAPACITY*mode.gain[2].copy()
    branch_a[6]-=model.RELEASE_RATE
    branch_slack=model.RELEASE_RATE*mode.center[6]-model.CAPACITY*mode.feedforward[2]
    branch_denom=float(branch_a@inverse@branch_a)
    branch_level=branch_slack**2/branch_denom if branch_slack>=0 and branch_denom>0 else 0.
    initial_error=model.BASE_STATE-mode.center
    initial_v=float(initial_error@mode.P@initial_error)
    closed_lyapunov=mode.A.T@mode.P+mode.P@mode.A
    return dict(scope='local linear Lyapunov and exact affine-support geometry only; no nonlinear or robust certificate',
                equilibrium_residual_at_reference_arrival=float(max(abs(model.field(mode.center,mode.feedforward,reference_arrival)))),
                equilibrium_residual_at_actual_base_arrival=float(max(abs(model.field(mode.center,mode.feedforward,model.BASE_ARRIVAL)))),
                minimum_equilibrium_margin=float(min(model.margins(mode.center,mode.feedforward))),
                closed_loop_spectral_abscissa=float(max(np.real(np.linalg.eigvals(mode.A)))),
                minimum_p_eigenvalue=float(min(np.linalg.eigvalsh(mode.P))),
                linear_lyapunov_residual=float(np.max(abs(closed_lyapunov+np.eye(7)))),
                largest_affine_admissible_quadratic_level=level,
                largest_capacity_branch_quadratic_level=float(branch_level),
                smooth_affine_admissible_level=min(level,float(branch_level)),
                initial_quadratic_value=initial_v,
                initial_to_affine_level_ratio=None if level<=0 else initial_v/level,
                initial_to_smooth_affine_level_ratio=None if min(level,branch_level)<=0 else initial_v/min(level,branch_level))


def simulate(mode,include_trajectory=False):
    x=model.BASE_STATE.copy()
    times,states=[0.],[x.copy()]
    tts=queue=trips=offered=0.
    minimum=np.inf
    minimum_input=np.inf
    minimum_state=np.inf
    failure=None
    deadline_n1=None
    initial_command=mode.command(x)
    for k in range(round(HORIZON/STEP)):
        t=k*STEP;u=mode.command(x)
        margins=model.margins(x,u)
        minimum=min(minimum,float(min(margins)))
        minimum_state=min(minimum_state,float(min(margins[:12])))
        minimum_input=min(minimum_input,float(min(margins[12:])))
        if not np.all(np.isfinite(x)) or min(margins)<-1e-7:
            failure=dict(time_h=t,kind='raw_physical_constraint_violation',minimum_margin=float(min(margins)))
            break
        new=model.rk4(x,mode.command,t,STEP,arrival)
        nu=mode.command(new)
        tts+=100*STEP/2*sum((x+new)[[0,1,6]])
        queue+=100*STEP/2*(x[6]+new[6])
        trips+=100*STEP/2*(model.departure_rate(x,u)+model.departure_rate(new,nu))
        offered+=100*STEP/2*(2*sum(model.INTERNAL_DEMAND)+arrival(t)+arrival(t+STEP))
        x=new;times.append(t+STEP);states.append(x.copy())
        if abs(t+STEP-DEADLINE)<1e-9: deadline_n1=float(x[0])
    final_margins=model.margins(x,mode.command(x))
    minimum=min(minimum,float(min(final_margins)))
    minimum_state=min(minimum_state,float(min(final_margins[:12])))
    minimum_input=min(minimum_input,float(min(final_margins[12:])))
    if failure is None and (not np.all(np.isfinite(x)) or min(final_margins)<-1e-7):
        failure=dict(time_h=times[-1],kind='raw_physical_constraint_violation',minimum_margin=float(min(final_margins)))
    completed=abs(times[-1]-HORIZON)<1e-9
    physically_feasible=completed and failure is None
    deadline_pass=deadline_n1 is not None and deadline_n1<=SERVICE_N1+1e-7
    result=dict(initial_command=initial_command.tolist(),
                full_horizon_completed=completed,physical_sampled_feasible=physically_feasible,
                deadline_n1=deadline_n1,deadline_pass=deadline_pass,
                eligible_for_cost_ranking=physically_feasible and deadline_pass,
                total_time_spent_veh_h=tts,queue_time_veh_h=queue,completed_trips_vehicles=trips,
                completed_horizon_h=times[-1],failure=failure,
                minimum_physical_margin=minimum,minimum_state_margin=minimum_state,
                minimum_input_margin=minimum_input,
                maximum_queue_vehicles=100*max(s[6] for s in states),
                final_queue_vehicles=100*x[6],final_state=x.tolist(),
                conservation_residual_vehicles=float(100*sum(model.BASE_STATE[[0,1,6]])+offered-100*sum(x[[0,1,6]])-trips),
                cost_scope='full two-hour cost' if physically_feasible else 'censored partial cost; not eligible for comparison')
    if include_trajectory:result['trajectory']=dict(time_h=times,state=np.array(states).tolist())
    return result


def replay(mode,baseline):
    y0=np.r_[model.BASE_STATE,np.zeros(4)]
    def rhs(t,y):
        x=y[:7];u=mode.command(x)
        return np.r_[model.field(x,u,arrival(t)),100*sum(x[[0,1,6]]),100*x[6],
                     100*model.departure_rate(x,u),100*(sum(model.INTERNAL_DEMAND)+arrival(t))]
    sol=solve_ivp(rhs,(0,HORIZON),y0,method='DOP853',rtol=1e-11,atol=1e-12,
                  dense_output=True,max_step=STEP)
    if not sol.success:raise RuntimeError(sol.message)
    samples=sol.sol(np.linspace(0,HORIZON,round(HORIZON*1800)+1))[:7].T
    minimum=min(float(min(model.margins(x,mode.command(x)))) for x in samples)
    y=sol.y[:,-1]
    return dict(scope='independent numerical held-feedback replay; not robust proof',
                method='DOP853',rtol=1e-11,atol=1e-12,dense_sample_seconds=2,
                total_time_spent_veh_h=float(y[7]),queue_time_veh_h=float(y[8]),
                completed_trips_vehicles=float(y[9]),deadline_n1=float(sol.sol(DEADLINE)[0]),
                minimum_physical_margin=minimum,final_state=y[:7].tolist(),
                conservation_residual_vehicles=float(100*sum(model.BASE_STATE[[0,1,6]])+y[10]-100*sum(y[:7][[0,1,6]])-y[9]),
                tts_difference_from_primary=float(y[7]-baseline['total_time_spent_veh_h']))


def main():
    started=time.perf_counter()
    records=[];modes={}
    for n1,n2 in TARGETS:
        for reference in REFERENCE_ARRIVALS:
            for penalty in PENALTIES:
                record=dict(target_n1=n1,target_n2=n2,reference_arrival=reference,penalty=penalty)
                try:
                    mode=synthesize(n1,n2,reference,penalty)
                    modes[mode.name]=mode
                    record.update(name=mode.name,center=mode.center.tolist(),feedforward=mode.feedforward.tolist(),
                                  gain=mode.gain.tolist(),P=mode.P.tolist(),geometry=geometry_record(mode,reference),
                                  simulation=simulate(mode))
                    sim=record['simulation']
                    print(f"{mode.name} eligible={sim['eligible_for_cost_ranking']} "
                          f"TTS={sim['total_time_spent_veh_h']:.3f} margin={sim['minimum_physical_margin']:.5f} "
                          f"n1deadline={sim['deadline_n1']}",flush=True)
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                    record['synthesis_failure']=dict(type=type(exc).__name__,message=str(exc))
                    print('synthesis_failure',record,flush=True)
                records.append(record)
    ranked=sorted((r for r in records if r.get('simulation',{}).get('eligible_for_cost_ranking')),
                  key=lambda r:r['simulation']['total_time_spent_veh_h'])
    selected=[]
    for record in ranked[:5]:
        mode=modes[record['name']]
        selected.append(dict(name=mode.name,simulation=simulate(mode,include_trajectory=True),
                             independent_replay=replay(mode,record['simulation'])))
    out=dict(scope='operating-library discovery/commissioning pilot; no nonlinear or robust certification',
             configuration=dict(targets=TARGETS,reference_arrivals=REFERENCE_ARRIVALS,penalties=PENALTIES,
                                weights=WEIGHTS,amplitude=AMPLITUDE,duration_h=DURATION,horizon_h=HORIZON,
                                step_seconds=3600*STEP,deadline_h=DEADLINE,service_n1=SERVICE_N1,
                                actual_arrival='common known 30% one-hour trapezoid around60',
                                internal_demand=model.INTERNAL_DEMAND.tolist(),
                                objective='queue-aware TTS over two hours',
                                selection_rule='rank all full-horizon physically and deadline-feasible candidates by TTS; retain five best'),
             candidate_count=len(records),eligible_count=len(ranked),records=records,
             ranked_eligible_names=[r['name'] for r in ranked],selected=selected,
             wall_seconds=time.perf_counter()-started)
    path=Path(__file__).resolve().parent/'results/mfd_incident_library_design_results.json'
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('RANKED',[(r['name'],r['simulation']['total_time_spent_veh_h']) for r in ranked[:5]],flush=True)


if __name__=='__main__':main()
