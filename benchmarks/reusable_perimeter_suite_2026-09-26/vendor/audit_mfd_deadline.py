#!/usr/bin/env python3
"""Independent falsification audits of the deadline interval implementation.

High-precision finite test cases independently check interval containment.
Random ODE trajectories are implementation falsification tests, not a substitute
for the interval construction's continuous-time proof or an uncertainty sample
used to define a robust certificate.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import mfd_deadline_certificates as cert
import mfd_cubic_pilot as plant
import mfd_cubic_runtime_pilot as runtime
import mfd_cubic_nonstationary_study as study

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_deadline_audit_results.json"
SEED = 284613


def m(value):
    """Exact embedding of a binary64 datum in 100-digit arithmetic."""
    return mp.mpf(float(value))


def mm(value):
    return mp.matrix([[m(z) for z in row] for row in np.asarray(value)])


def contains(interval, exact):
    lo, hi = np.asarray(interval.lo).ravel(), np.asarray(interval.hi).ravel()
    if isinstance(exact, mp.matrix):
        values = [exact[i, j] for i in range(exact.rows) for j in range(exact.cols)]
    elif isinstance(exact, (tuple, list, np.ndarray)):
        values = list(np.asarray(exact, dtype=object).ravel())
    else:
        values = [exact]
    assert len(values) == len(lo), (len(values), len(lo))
    for index, (lower, upper, value) in enumerate(zip(lo, hi, values)):
        assert m(lower) <= value <= m(upper), {
            "index": index, "lo": float(lower), "hi": float(upper),
            "exact": mp.nstr(value, 40), "outside_below": mp.nstr(m(lower)-value, 8),
            "outside_above": mp.nstr(value-m(upper), 8)}


def high_precision_field(x, gain, demands=None):
    """Independent conservation equations, without certificate interval classes."""
    demand = [m(z) for z in plant.DEMAND] if demands is None else demands
    commands = [m(plant.U_STAR[i])+sum(m(gain[i,j])*(x[j]-m(plant.X_STAR[j])) for j in range(6)) for i in range(4)]
    n1,n2,n11,n22,n10,n20=x
    u12,u21,u02,u20=commands
    c1=m(plant.MFD_A)*n1**2+m(plant.MFD_B)*n1+m(plant.MFD_C)
    c2=m(plant.MFD_A)*n2**2+m(plant.MFD_B)*n2+m(plant.MFD_C)
    q11,q12,q10=n11*c1,(n1-n11-n10)*c1,n10*c1
    q22,q21,q20=n22*c2,(n2-n22-n20)*c2,n20*c2
    d11,d12,d10,d22,d21,d20=demand
    return mp.matrix([-q11-(q12+q10)*u12+q21*u21+d11+d12+d10,
        -q22+(q12+q10)*u12-q21*u21+(m(plant.Q01)+m(plant.Q02))*u02-q20*u20+d22+d21+d20,
        -q11+q21*u21+d11,-q22+q12*u12+m(plant.Q02)*u02+d22,
        -q10*u12+d10,-q20*u20+q10*u12+d20])


def primitive_audit(rng):
    count = 0
    for _ in range(150):
        a, b = rng.normal(size=2)*10.0**rng.uniform(-12, 12, 2)
        da, db = abs(a)*rng.uniform(0, .1), abs(b)*rng.uniform(0, .1)
        aa, bb = cert.I(a-da, a+da), cert.I(b-db, b+db)
        for x in (aa.lo, aa.hi):
            for y in (bb.lo, bb.hi):
                contains(aa+bb, m(x)+m(y)); contains(aa-bb, m(x)-m(y)); contains(aa*bb, m(x)*m(y))
                count += 3
    matrix_cases = 0
    for _ in range(12):
        a = rng.normal(size=(6,6)); b=rng.normal(size=(6,6)); v=rng.normal(size=6)
        contains(cert.imat(a,b), mm(a)*mm(b))
        contains(cert.imat(a,v), mm(a)*mp.matrix([m(z) for z in v]))
        assert m(cert.norm_upper(v)) >= mp.sqrt(sum(m(z)**2 for z in v))
        eigenvalues = mp.eigsy(mm(a).T*mm(a), eigvals_only=True)
        assert m(cert.operator_upper(a)) >= mp.sqrt(eigenvalues[eigenvalues.rows-1])
        invertible=a+6*np.eye(6)
        contains(cert.inverse_interval(invertible), mm(invertible)**-1)
        sym=(a+a.T)/2
        symmetric_values=mp.eigsy(mm(sym),eigvals_only=True)
        largest=symmetric_values[symmetric_values.rows-1]
        assert m(cert.symmetric_eigen_upper(cert.I(sym))) >= largest
        matrix_cases += 1
    return {"scalar_endpoint_operations": count, "matrix_cases": matrix_cases,
        "arithmetic_precision_decimal_digits": mp.mp.dps, "all_contained": True}


def field_and_jacobian_audit(model, rng):
    samples = 0; point_fields = 0; time_polynomial_points = 0
    for mode in (0,1):
        L=mm(model.L[mode]); T=L**-1
        for _ in range(12):
            middle=rng.normal(0, .10, 6); halfwidth=rng.uniform(.001,.025,6)
            box=cert.I(middle-halfwidth,middle+halfwidth)
            bound=cert.evaluate_jacobian_polynomial(model.jac_polynomials[mode],box)
            for location in (middle-halfwidth,middle+halfwidth,middle+rng.uniform(-1,1,6)*halfwidth):
                z=mp.matrix([m(v) for v in location]); physical=mp.matrix([m(v) for v in plant.X_STAR])+T*z
                def transformed(coordinates):
                    x=mp.matrix([m(v) for v in plant.X_STAR])+T*mp.matrix(coordinates)
                    return L*high_precision_field(x,model.gains[mode])
                derivative=mp.matrix(6,6)
                for j in range(6):
                    for i in range(6):
                        def component(t):
                            zz=list(z);zz[j]=t
                            return transformed(zz)[i]
                        derivative[i,j]=mp.diff(component,z[j])
                contains(bound,derivative)
                samples += 1
            x=np.array([float(value) for value in physical])
            interval_field=cert.field([cert.I(v) for v in x],model.gains[mode])
            exact=high_precision_field([m(v) for v in x],model.gains[mode])
            for i in range(6): contains(interval_field[i],exact[i])
            point_fields += 1
        for _ in range(6):
            x=plant.X_STAR+rng.normal(0,.1,6)
            v=rng.normal(0,3,6); acceleration=rng.normal(0,20,6)
            curve=[cert.J([x[k],v[k],acceleration[k]]) for k in range(6)]
            values=cert.field(curve,model.gains[mode])
            residual=[values[k]-cert.J([v[k],2*acceleration[k]]) for k in range(6)]
            assert max(len(z.c) for z in residual)==9
            for t in (0.,1./480,1./240):
                exact_x=[m(x[k])+m(v[k])*m(t)+m(acceleration[k])*m(t)**2 for k in range(6)]
                exact=high_precision_field(exact_x,model.gains[mode])-mp.matrix([m(v[k])+2*m(acceleration[k])*m(t) for k in range(6)])
                for k in range(6):
                    polynomial=residual[k]; power=cert.I(1.); evaluation=cert.I(0.)
                    for coefficient in polynomial.c:
                        evaluation=evaluation+coefficient*power;power=power*cert.I(t)
                    contains(evaluation,exact[k])
                time_polynomial_points += 1
    return {"transformed_jacobian_high_precision_points": samples,
        "point_field_high_precision_checks": point_fields,
        "degree_eight_predictor_residual_points": time_polynomial_points,"all_contained": True}


def matrix_function_audit(model, rng):
    exponential_cases=0; adjoint_points=0; max_scipy_midpoint_difference=0.
    for mode in (0,1):
        A=model.linear[mode]
        for h in (1./3600,15./3600,1./64):
            bound=cert.exponential_interval(A,h)
            reference=mp.expm(mm(A)*m(h))
            contains(bound,reference)
            scipy_value=expm(A*h)
            scipy_contained=bool(np.all(scipy_value>=bound.lo)&np.all(scipy_value<=bound.hi))
            assert scipy_contained
            max_scipy_midpoint_difference=max(max_scipy_midpoint_difference,float(np.max(abs(scipy_value-bound.mid()))))
            exponential_cases += 1
            row=rng.normal(size=(1,6)); envelope=cert.adjoint_over_step(row,A,h)
            for fraction in np.linspace(0.,1.,11):
                contains(envelope,mm(row)*mp.expm(mm(A)*m(h)*m(fraction)))
                adjoint_points += 1
    return {"exponential_high_precision_and_scipy_cases": exponential_cases,
        "adjoint_high_precision_points": adjoint_points,
        "maximum_scipy_exponential_midpoint_difference": max_scipy_midpoint_difference,
        "all_contained": True}


def channel_audit(model,rng):
    signs=[1,1,1,-1,-1,-1]
    d=[m(z) for z in plant.DEMAND]
    amplitude=m(model.amplitude)
    exact_channel=mp.matrix([d[0]+d[1]+d[2],-(d[3]+d[4]+d[5]),d[0],-d[3],d[2],-d[5]])*amplitude
    contains(model.disturbance,exact_channel)
    maximum_float_residual=0.
    for mode in (0,1):
        for _ in range(20):
            x=plant.X_STAR+rng.normal(0,.2,6)
            sign=rng.uniform(-1,1)
            perturbed=[d[i]+m(sign)*amplitude*signs[i]*d[i] for i in range(6)]
            difference=high_precision_field([m(v) for v in x],model.gains[mode],perturbed)-high_precision_field([m(v) for v in x],model.gains[mode],d)
            assert max(abs(difference[i]-m(sign)*exact_channel[i]) for i in range(6)) < mp.mpf('1e-95')
            direct=plant.mfd_rhs(x,plant.U_STAR+model.gains[mode]@(x-plant.X_STAR),plant.DEMAND+sign*model.amplitude*study.shift_direction())
            nominal=plant.mfd_rhs(x,plant.U_STAR+model.gains[mode]@(x-plant.X_STAR))
            maximum_float_residual=max(maximum_float_residual,float(np.max(abs(direct-nominal-sign*np.array([float(z) for z in exact_channel])))))
    return {"high_precision_state_mode_sign_cases":40,"exact_rank_one_channel_contained":True,
        "maximum_float_residual":maximum_float_residual,"channel": [float(z) for z in exact_channel]}


def simulate_disturbance(model,x0,mode,values,tight=False):
    x=x0.copy(); grid=np.linspace(0.,.25,len(values)+1)
    minimum_input=np.inf; maximum_occupancy=0.
    for start,end,sign in zip(grid[:-1],grid[1:],values):
        demand=plant.DEMAND+model.amplitude*float(sign)*study.shift_direction()
        result=solve_ivp(lambda _t,x: plant.mfd_rhs(x,plant.U_STAR+model.gains[mode]@(x-plant.X_STAR),demand),
            (start,end),x,method="DOP853",rtol=2.e-12 if tight else 1.e-10,
            atol=2.e-13 if tight else 1.e-12,max_step=(end-start)/(24 if tight else 12),
            t_eval=np.linspace(start,end,25))
        assert result.success
        for current in result.y.T:
            _,margins=plant.margin_components(current,mode,model.gains)
            minimum_input=min(minimum_input,float(np.min(margins[:10])))
            error=current-plant.X_STAR
            maximum_occupancy=max(maximum_occupancy,float(error@model.P[mode]@error/model.tube['common_level']))
        x=result.y[:,-1]
    return x,minimum_input,maximum_occupancy


def disturbance_sampling_audit(model,rng):
    x0=runtime.state_on_tube(runtime.structured_directions()['region_2_heavy'],.8,model.cert,model.tube,0)
    records=[model.schedule(x0,[mode],step_seconds=15.) for mode in (0,1)]
    assert all(record['verified_continuous_tube'] for record in records)
    patterns=np.where(rng.random((200,10))<.5,-1.,1.)
    patterns[0]=1.;patterns[1]=-1.;patterns[2]=np.tile([-1.,1.],5);patterns[3]=-patterns[2]
    results=[]; max_refinement_discrepancy=0.
    for mode in (0,1):
        terminal=records[mode]['records'][-1]
        center=np.array(terminal['center']); radius=terminal['radius']
        bound=terminal['absolute_imbalance_upper_vehicles']
        outcomes=[]; max_ball_ratio=0.; min_input=np.inf; max_occ=0.
        for index,pattern in enumerate(patterns):
            x,input_margin,occupancy=simulate_disturbance(model,x0,mode,pattern)
            absolute_output=100.*abs(x[0]-x[1]); ball_ratio=np.linalg.norm(model.L[mode]@(x-center))/radius
            assert absolute_output <= bound
            assert ball_ratio <= 1.
            assert input_margin > 0. and occupancy <=1.+1.e-9
            max_ball_ratio=max(max_ball_ratio,float(ball_ratio)); min_input=min(min_input,input_margin);max_occ=max(max_occ,occupancy)
            outcomes.append(float(100.*(x[0]-x[1])))
            if index<4:
                refined,_,_=simulate_disturbance(model,x0,mode,pattern,tight=True)
                max_refinement_discrepancy=max(max_refinement_discrepancy,100.*float(np.max(abs(x-refined))))
        results.append({'mode_one_based':mode+1,'sample_count':len(patterns),
            'certificate':records[mode],'signed_terminal_outputs_vehicles':outcomes,
            'maximum_sampled_absolute_output_vehicles':max(abs(z) for z in outcomes),
            'minimum_sampled_signed_output_vehicles':min(outcomes),'maximum_sampled_signed_output_vehicles':max(outcomes),
            'maximum_terminal_error_ball_ratio':max_ball_ratio,'minimum_sampled_input_margin':min_input,
            'maximum_sampled_tube_occupancy':max_occ,'all_sampled_trajectories_contained':True})
    return {'scope':'sampling falsification only; robust certificate is computed independently of samples',
        'draw_count':200,'trajectories_per_draw':2,'disturbance_blocks':10,'block_duration_seconds':90,
        'disturbance_value_set':[-.003,.003],'initial_state':x0.tolist(),
        'initial_imbalance_vehicles':100.*abs(x0[0]-x0[1]),
        'fixed_terminal_target_vehicles':.1*100.*abs(x0[0]-x0[1]),
        'disturbance_sign_patterns':patterns.tolist(),'maximum_refinement_difference_vehicles':max_refinement_discrepancy,
        'results':results}


def main():
    started=time.perf_counter();mp.mp.dps=100;rng=np.random.default_rng(SEED)
    model=cert.Encloser(.003)
    out={'scope':'independent finite implementation audits; sampling is not a proof', 'seed':SEED}
    out['interval_primitives']=primitive_audit(rng);print('primitive audit passed',flush=True)
    out['field_jacobian']=field_and_jacobian_audit(model,rng);print('field/Jacobian audit passed',flush=True)
    out['matrix_functions']=matrix_function_audit(model,rng);print('matrix-function audit passed',flush=True)
    out['disturbance_channel']=channel_audit(model,rng);print('rank-one channel audit passed',flush=True)
    out['disturbance_sampling_seed']=SEED+1
    out['disturbance_sampling']=disturbance_sampling_audit(model,np.random.default_rng(SEED+1));print('400 trajectory containment checks passed',flush=True)
    out['wall_seconds']=time.perf_counter()-started
    OUTPUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'wall_seconds':out['wall_seconds'],'interval_audits_passed':True,
        'maximum_refinement_difference_vehicles':out['disturbance_sampling']['maximum_refinement_difference_vehicles'],
        'mode_summary':[{'mode':z['mode_one_based'],'sample_max':z['maximum_sampled_absolute_output_vehicles'],
            'certified_max':z['certificate']['records'][-1]['absolute_imbalance_upper_vehicles'],
            'ball_ratio':z['maximum_terminal_error_ball_ratio']} for z in out['disturbance_sampling']['results']]},indent=2))


if __name__=='__main__': main()
