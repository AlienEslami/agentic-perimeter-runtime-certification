"""Independent arithmetic, provenance and numerical audit of positive enclosures.

No original experiment or certificate is modified. The numerical corner/adverse
replays are implementation diagnostics; proof checks are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import mfd_incident_model as model
from mfd_deadline_certificates import I,imat
from mfd_incident_mpc_certification import ConstantCommandVerifier,constant_command_mode,verify_command_plan
from mfd_incident_verifier import Pulse,contains_information_box
from mfd_incident_positive_reachability import PositiveVerifier,initial_positivity
from mfd_incident_queue_bounds import queue_step_bounds

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'reproducibility/results'
LOWER_NAMES=['n1_lower','n2_lower','n11_lower','n22_lower','n10_lower','n20_lower','exterior_queue_lower','n12_lower','n21_lower']
CLASS=np.array([[0,0,1,0,0,0,0],[1,0,-1,0,-1,0,0],[0,0,0,0,1,0,0],
                [0,0,0,1,0,0,0],[0,1,0,-1,0,-1,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1.]])


def rational(x): return F.from_float(float(x))
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dot(row,values): return sum(rational(a)*rational(b) for a,b in zip(row,values))
def exact_array_equal(a,b): return np.array_equal(np.asarray(a),np.asarray(b))


def grids(descriptor):
    pulse=Pulse(*map(F,descriptor['pulse']))
    cursor=F(descriptor['initial_time'])
    out=[]
    for segment,duration in enumerate(map(F,descriptor['durations'])):
        end=cursor+duration
        cuts=[cursor]+sorted(set(c for c in pulse.knots+list(map(F,descriptor['timecuts'])) if cursor<c<end))+[end]
        for left,right in zip(cuts[:-1],cuts[1:]):
            count=max(1,math.ceil(float((right-left)*3600)/descriptor['step_seconds']))
            out.extend((segment,left+(right-left)*F(k,count),left+(right-left)*F(k+1,count)) for k in range(count))
        cursor=end
    return pulse,out


def exact_assumptions(case,commands):
    certificate=case['certificate']; desc=certificate['proof_descriptor']
    recorded=desc['structural_positive_extension']['exact_assumptions']
    center=desc['initial_state']; widths=certificate['initial_box_units']
    a,b,c=map(rational,desc['model_contract']['MFD_coefficients'])
    delta=4*a*c-b*b
    lower=[dot(row,center)-sum(abs(rational(v))*rational(w) for v,w in zip(row,widths)) for row in CLASS]
    eta=rational(desc['residual_bound']); arrival_lower=rational(model.BASE_ARRIVAL)-eta
    assert a>0 and delta>0 and min(lower)>=0 and arrival_lower>=0
    assert all(rational(x)>=0 for x in desc['model_contract']['internal_demand'])
    assert all(min(map(rational,u))>=0 for u in commands)
    assert recorded['initial_class_lower_exact']==list(map(str,lower))
    assert recorded['speed_4ac_minus_b_squared_exact']==str(delta)
    assert recorded['global_speed_minimum_exact']==str(delta/(4*a))
    assert recorded['offered_arrival_lower_exact']==str(arrival_lower)
    assert recorded['lower_rows_certified']==LOWER_NAMES
    assert recorded['passed'] and recorded['zero_gain_nonnegative_commands']
    initial_L=desc['initial_L']; initial_radius=desc['initial_radius']
    for row,r in zip(initial_L,initial_radius):
        assert rational(r)>=sum(abs(rational(v))*rational(w) for v,w in zip(row,widths))
    return dict(global_speed_minimum_exact=str(delta/(4*a)),initial_class_lower_exact=list(map(str,lower)),
                initial_full_physical_box_validated=True,arrival_lower_exact=str(arrival_lower))


def contraction_audit(record,raw_center,raw_radius,L,queue,variant):
    saved=record['endpoint_contraction']
    lo,hi=list(map(rational,saved['raw_lower'])),list(map(rational,saved['raw_upper']))
    newlo,newhi=list(map(rational,saved['contracted_lower'])),list(map(rational,saved['contracted_upper']))
    for j,row in enumerate(L):
        center=dot(row,raw_center); radius=rational(raw_radius[j])
        assert lo[j]<=center-radius and hi[j]>=center+radius
        assert lo[j]<=newlo[j]<=newhi[j]<=hi[j]
    # Upper supports cannot be cut using jam/storage/service requirements.
    allowed_upper=hi.copy()
    if queue is not None: allowed_upper[6]=min(allowed_upper[6],rational(queue['endpoint'][1]))
    assert newhi==allowed_upper
    safe_lower=lo.copy()
    for j in ([0,2,5,6] if variant=='strip_queue' else [0,1,2,5,6]):
        safe_lower[j]=max(safe_lower[j],F(0))
    if variant=='strip_queue':
        safe_lower[1]=max(safe_lower[1],-2*hi[2])
        safe_lower[2]=max(safe_lower[2],-hi[1]/2)
    if queue is not None: safe_lower[6]=max(safe_lower[6],rational(queue['endpoint'][0]))
    for j in (3,4): safe_lower[j]=max(safe_lower[j],safe_lower[6]/2)
    assert all(newlo[j]<=safe_lower[j] for j in range(7)), 'Unjustified endpoint lower contraction'
    center=saved['returned_center']; radius=saved['returned_radius']
    for j,row in enumerate(L):
        c=dot(row,center); r=rational(radius[j])
        assert c-r<=newlo[j] and c+r>=newhi[j], 'Returned parallelotope misses contracted box'
    assert exact_array_equal(radius,record['radius'])
    return np.array(center),np.array(radius)


def audit_case(path):
    began=time.perf_counter(); case=read(path)
    if 'certificate' not in case:
        assert not case['accepted'] and case['classification']=='invalid_initial_information_set'
        return dict(file=path.name,accepted=False,classification=case['classification'],interval_steps=0)
    cert=case['certificate']; desc=cert['proof_descriptor']; extension=desc['structural_positive_extension']
    expected_hash=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest()
    assert cert['proof_id']==expected_hash
    for shape in cert['reachable_checkpoints']: assert shape['proof_id']==expected_hash
    snapshots=0
    for name,digest in case['source_sha256'].items():
        if name.endswith('_results.json'):
            assert sha(ROOT/name)==digest
        else:
            archived=RESULTS/'pr'/digest[:16]/Path(name).name
            assert archived.exists() and sha(archived)==digest
            snapshots+=1
    assert extension['source_sha256']==case['source_sha256']['reproducibility/mfd_incident_positive_reachability.py']
    assert extension['protocol_sha256']==case['source_sha256']['POSITIVE_REACHABILITY_PROTOCOL.md']
    variant=case['variant']; has_queue=variant in ('positive_queue','strip_queue')
    if has_queue: assert extension['queue_source_sha256']==case['source_sha256']['reproducibility/mfd_incident_queue_bounds.py']
    contracts=desc['model_contract']
    assert contracts['MFD_coefficients']==[model.original.MFD_A,model.original.MFD_B,model.original.MFD_C]
    assert contracts['internal_demand']==model.INTERNAL_DEMAND.tolist()
    assert contracts['queue_limit']==20. and contracts['regional_jam_limits']==[100.,100.]
    assert contracts['admission_capacity']==120. and contracts['release_rate']==60.
    commands=cert['adapter_audit']['commands']
    archive=read(RESULTS/'mfd_incident_mpc_grid_results.json')
    original=next(row for row in archive['runs'] if row['amplitude']==.3 and row['duration_h']==1.)['decisions'][0]
    assert commands==original['controls'] and desc['initial_state']==original['initial_state']
    assert desc['durations']==['1/12','11/12'] and desc['initial_time']=='0'
    assert desc['service_checks']==[dict(exact_time='1',coordinate=0,upper=28.5)]
    assumptions=exact_assumptions(case,commands)
    geometry=PositiveVerifier(variant)
    modes=[constant_command_mode(u,f'constant_block_{i}') for i,u in enumerate(commands)]
    for mode,fingerprint in zip(modes,desc['mode_fingerprints']):
        assert geometry.geometry(mode)['fingerprint']==fingerprint
    pulse,grid=grids(desc)
    center=np.array(desc['initial_state']); radius=np.array(desc['initial_radius'])
    queue_start=I(center[6])+I(-cert['initial_box_units'][6],cert['initial_box_units'][6])
    queue_state=[max(0.,float(queue_start.lo)),float(queue_start.hi)]
    checkpoint_map={F(item['exact_time_fraction']):item for item in cert['reachable_checkpoints']}
    initial_checkpoint=checkpoint_map[F(0)]
    assert initial_checkpoint['center']==center.tolist() and initial_checkpoint['radius']==radius.tolist()
    override_rows=contractions=queue_steps=0
    unchanged_keys=('defect_upper','initial_radius','apriori_radius','predictor_endpoint_rounding',
                    'log_norm_upper','comparison_method','active_branch','branch_gap_interval','attempts',
                    'constant_command_support_audit')
    for step,(segment,left,right) in zip(cert['interval_steps'],grid):
        assert step['segment']==segment and step['left_h']==float(left) and step['right_h']==float(right)
        assert exact_array_equal(step['initial_radius'],radius)
        # Recompute the old uncut first-exit step, not the new subclass step.
        raw_center,raw_radius,raw=ConstantCommandVerifier.step(geometry,center,radius,modes[segment],left,right,pulse,desc['residual_bound'])
        for key in unchanged_keys:
            assert step[key]==raw[key], (path.name,float(left),'underlying enclosure changed',key)
        for initial,endpoint,trial in zip(raw['initial_radius'],raw['attempts'][-1]['endpoint_comparison_radius'],raw['apriori_radius']):
            assert max(rational(initial),rational(endpoint))<rational(trial)
        margins=raw['continuous_constraint_margins_lower'].copy()
        positive=step['structural_positivity']
        assert positive['independently_verified'] and positive['replaced_rows']==LOWER_NAMES
        assert positive['raw_lower_supports']==margins[:9]
        override_rows+=sum(value<0 for value in margins[:9])
        margins[:9]=[max(0.,value) for value in margins[:9]]
        queue=None
        if has_queue:
            a,slope=pulse.affine(left,right)
            queue=queue_step_bounds(*queue_state,right-left,a,slope,desc['residual_bound'],commands[segment][2])
            assert queue==step['autonomous_queue_bound'], 'Autonomous queue chain/binding changed'
            queue_state=queue['endpoint']; queue_steps+=1
            margins[11]=max(margins[11],float((I(20.)-I(queue['continuous'][1])).lo))
        assert margins==step['continuous_constraint_margins_lower'], 'Unjustified physical/actuator support change'
        if variant=='structural_only':
            assert 'endpoint_contraction' not in step
            center,radius=raw_center,raw_radius
            assert step['radius']==raw['radius']
        else:
            center,radius=contraction_audit(step,raw_center,raw_radius,geometry.geometry(modes[segment])['L'],queue,variant)
            contractions+=1
        if right in checkpoint_map:
            cp=checkpoint_map[right]
            assert cp['center']==center.tolist() and cp['radius']==radius.tolist()
            assert cp['metric_L']==geometry.geometry(modes[segment])['L'].tolist()
    assert len(cert['interval_steps'])<=len(grid)
    assert len(cert['interval_steps'])==cert['completed_interval_steps']
    if case['accepted']:
        assert cert['certified_until_h']==1. and len(cert['interval_steps'])==len(grid)
        assert all(value>=0 for value in cert['continuous_constraint_margins_lower'].values())
        assert cert['service_check_results'][0]['passed'] and cert['service_check_results'][0]['upper']<=28.5
    return dict(file=path.name,source_sha256=sha(path),accepted=case['accepted'],variant=variant,
        state_width_vehicles=case['state_half_width_vehicles'],residual_vehicles_h=case['residual_vehicles_h'],
        interval_steps=len(cert['interval_steps']),strict_first_exit_steps=len(cert['interval_steps']),
        independently_recomputed_original_steps=len(cert['interval_steps']),
        negative_lower_supports_discharged=override_rows,exact_rational_endpoint_contractors=contractions,
        independently_recomputed_scalar_queue_steps=queue_steps,source_snapshots_checked=snapshots,
        assumptions=assumptions,proof_id=expected_hash,wall_seconds=time.perf_counter()-began,
        legacy_metadata_warning=('Early metadata describes upper supports/hull too strongly; actual operations independently audited' if 'upper_requirements_unchanged' not in extension else None))


def negative_tests():
    nominal=model.BASE_STATE.copy(); command=np.array([.5,.5,.5,.5])
    modes=[constant_command_mode(command)]
    bad=[]
    for name,coordinate,value in [('negative_stored_class',2,-.1),('negative_queue',6,-.1),('negative_derived_class',0,1.)]:
        state=nominal.copy(); state[coordinate]=value
        assert not initial_positivity(state,np.zeros(7),modes,.6)['passed']
        bad.append(name)
    assert not initial_positivity(nominal,np.full(7,2.001),modes,.6)['passed']
    assert not initial_positivity(nominal,np.zeros(7),modes,60.001)['passed']
    bad+=['full_information_box_contains_negative_queue','negative_arrival_envelope']
    negative=command.copy(); negative[2]=-.001
    assert not initial_positivity(nominal,np.zeros(7),[constant_command_mode(negative)],.6)['passed']
    bad.append('negative_gate_command')
    original_c=model.original.MFD_C
    try:
        model.original.MFD_C=-1.
        assert not initial_positivity(nominal,np.zeros(7),modes,.6)['passed']
    finally:
        model.original.MFD_C=original_c
    bad.append('nonpositive_speed_factor_memory_only_fixture')
    assert initial_positivity(nominal,np.full(7,2.),modes,.6)['passed']
    counter=np.array([100.,0.,100.,0.,0.,0.,20.]); zero=np.zeros(4)
    counter_results=[]
    for variant in ('structural_only','positive_rebox','positive_queue','strip_queue'):
        proof=verify_command_plan(counter,[zero],[F(1,240)],initial_box=0.,residual_bound=.6,
            verifier=PositiveVerifier(variant),retain_steps=True,service_checks=[],timecuts_h=[])
        assert not proof['accepted']
        counter_results.append(dict(variant=variant,accepted=proof['accepted'],
            failures=[{k:f[k] for k in ('kind','constraint','margin_lower') if k in f} for f in proof['failures']]))
    drift=model.field(counter,zero,60.)
    assert drift[0]>0 and drift[6]>0
    return dict(rejected_hypotheses=bad,exact_zero_initial_queue_support_allowed=True,
        upper_boundary_counterexample=counter_results,
        upper_boundary_drift=dict(region1=float(drift[0]),queue=float(drift[6])),
        model_coefficients_restored=model.original.MFD_C==original_c)


def inward_corner(center,widths,signs):
    out=[]
    for c,w,s in zip(center,widths,signs):
        target=rational(c)+int(s)*rational(w); near=float(target)
        if abs(rational(near)-rational(c))>rational(w): near=float(np.nextafter(near,c))
        assert abs(rational(near)-rational(c))<=rational(w)
        out.append(near)
    return np.array(out)


def high_precision_queue(initial,commands,pulse,eta,sign,checkpoints):
    """Independent 90-digit scalar reference, not the interval bound routine.

    The fixed audit mission contains constant arrivals and one rising ramp.
    Constant-input crossings have closed forms; a rising-ramp upward crossing
    is solved by 260-step high-precision bisection. This diagnostic never
    changes the numerical class trajectory or an archived certificate.
    """
    from mpmath import mp
    times=sorted(set([F(0),F(1,12),F(1)]+[k for k in pulse.knots if 0<k<1]+
                     [F(cp['exact_time_fraction']) for cp in checkpoints]))
    values={}
    with mp.workdps(90):
        convert=lambda x:mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(float(x))
        z=convert(rational(initial)); values[0.]=float(z)
        for left,right in zip(times[:-1],times[1:]):
            mid=(left+right)/2; base=rational(model.BASE_ARRIVAL)
            if mid<=pulse.knots[0] or mid>=pulse.knots[3]: a,b=base,F(0)
            elif mid<pulse.knots[1]:
                b=base*pulse.amplitude/pulse.ramp; a=base+b*(left-pulse.start)
            elif mid<=pulse.knots[2]: a,b=base*(1+pulse.amplitude),F(0)
            else: raise AssertionError('The frozen one-hour diagnostic unexpectedly contains the falling ramp')
            a+=sign*rational(eta)
            A,B,H=map(convert,(a,b,right-left))
            q=convert(F(120)*rational(commands[0 if left<F(1,12) else 1][2])); threshold=q/60
            def supply(start,t):
                decay=mp.exp(-60*t); phi1=-mp.expm1(-60*t)/60
                return start*decay+A*phi1+B*(t-phi1)/60
            if b==0:
                if z>=threshold:
                    crossing=mp.inf if A>=q else (z-threshold)/(q-A)
                    z=z+(A-q)*H if H<=crossing else A/60+(threshold-A/60)*mp.exp(-60*(H-crossing))
                elif A<=q:z=supply(z,H)
                else:
                    crossing=mp.log((A-60*z)/(A-q))/60
                    z=supply(z,H) if H<=crossing else threshold+(A-q)*(H-crossing)
            elif z>=threshold:
                assert A>=q, 'Unexpected downward ramp crossing in independent reference'
                z=z+(A-q)*H+B*H*H/2
            else:
                end=supply(z,H)
                if end<=threshold:z=end
                else:
                    low,high=mp.mpf(0),H
                    for _ in range(260):
                        midpoint=(low+high)/2
                        if supply(z,midpoint)<=threshold:low=midpoint
                        else:high=midpoint
                    crossing=(low+high)/2; tail=H-crossing
                    z=threshold+(A+B*crossing-q)*tail+B*tail*tail/2
            values[float(right)]=float(z)
    return values


def replay_case(path):
    case=read(path); cert=case['certificate']; desc=cert['proof_descriptor']
    assert case['accepted']
    x0=np.array(desc['initial_state']); widths=np.array(cert['initial_box_units'])
    commands=cert['adapter_audit']['commands']; eta=desc['residual_bound']
    pulse,_=grids(desc)
    checkpoints=cert['reachable_checkpoints']; checkpoint_times=sorted(set(float(F(cp['exact_time_fraction'])) for cp in checkpoints))
    cuts=sorted(set([0.,1/12,1.]+[float(knot) for knot in pulse.knots if 0<knot<1]+checkpoint_times))
    directions=[('all_high',[1]*7),('all_low',[-1]*7),
                ('derived_classes_low',[-1,-1,1,1,1,1,1]),('stored_classes_low',[1,1,-1,-1,-1,-1,-1])]
    trials=[('nominal',x0,0)]
    for name,signs in directions:
        point=inward_corner(x0,widths,signs)
        for sign in (-1,1): trials.append((name+('_positive' if sign>0 else '_negative'),point,sign))
    records=[]
    for label,initial,sign in trials:
        def integrate(tolerance,max_step):
            current=np.r_[CLASS@initial,np.zeros(3)]
            values={0.:current.copy()}
            for left,right in zip(cuts[:-1],cuts[1:]):
                u=np.array(commands[0 if left<1/12-1e-13 else 1])
                def rhs(t,y):
                    c11,c12,c10,c22,c21,c20,z=y[:7]
                    n1,n2=c11+c12+c10,c22+c21+c20
                    a,b,c=model.original.MFD_A,model.original.MFD_B,model.original.MFD_C
                    h1=a*n1*n1+b*n1+c; h2=a*n2*n2+b*n2+c
                    admitted=min(60*z,120*u[2]); arrival=model.arrival(t,amplitude=.3,duration=1.)+sign*eta
                    d11,d12,d10,d22,d21,d20=model.INTERNAL_DEMAND
                    rates=np.array([d11+u[1]*h2*c21-h1*c11,d12-u[0]*h1*c12,d10-u[0]*h1*c10,
                        d22+u[0]*h1*c12+.5*admitted-h2*c22,d21+.5*admitted-u[1]*h2*c21,
                        d20+u[0]*h1*c10-u[3]*h2*c20,arrival-admitted])
                    out=h1*c11+h2*c22+u[3]*h2*c20
                    return np.r_[rates,sum(model.INTERNAL_DEMAND)+arrival,out,n1+n2+z]
                sol=solve_ivp(rhs,(left,right),current,method='DOP853',rtol=tolerance,atol=tolerance/10,
                              max_step=max_step)
                assert sol.success
                current=sol.y[:,-1]; values[right]=current.copy()
            return values
        first=integrate(2e-11,1/240); second=integrate(2e-12,1/480)
        maximum_error=100*max(float(np.max(abs(first[t][:7]-second[t][:7]))) for t in checkpoint_times)
        exact_queue=high_precision_queue(initial[6],commands,pulse,eta,sign,checkpoints)
        minimum_slack=np.inf; maximum_ratio=0.; point_checks=0
        raw_overshoots=[]; queue_correction=0.
        for checkpoint in checkpoints:
            t=float(F(checkpoint['exact_time_fraction'])); c=second[t][:7]
            state=np.array([sum(c[:3]),sum(c[3:6]),c[0],c[3],c[2],c[5],c[6]])
            raw_membership=contains_information_box(checkpoint,state,0.)
            if not raw_membership['passed']:
                excess=np.array(raw_membership['transformed_absolute_error_upper'])-np.array(checkpoint['radius'])
                indices=np.flatnonzero(excess>0).tolist()
                raw_overshoots.append(dict(time_h=t,metric_rows=indices,
                    maximum_excess_times100=float(100*max(excess))))
                assert indices==[6] and 100*max(excess)<1e-7,(path.name,label,'numerical discrepancy outside tiny queue boundary',raw_membership)
            queue_correction=max(queue_correction,100*abs(state[6]-exact_queue[t]))
            state[6]=exact_queue[t]
            if t==0:state=initial.copy()
            membership=contains_information_box(checkpoint,state,0.)
            assert membership['passed'],(path.name,label,t,'90-digit queue reference still outside',membership)
            L=np.array(checkpoint['metric_L']); r=np.array(checkpoint['radius'])
            displacement=abs(L@(state-np.array(checkpoint['center'])))
            maximum_ratio=max(maximum_ratio,float(np.max(displacement/r)))
            if t>0: minimum_slack=min(minimum_slack,float(np.min(r-displacement)))
            point_checks+=1
        last=second[1.]; inventory0=float(np.sum(CLASS@initial))
        balance=100*(inventory0+last[7]-sum(last[:7])-last[8])
        assert abs(balance)<2e-7 and maximum_error<2e-5
        assert min(last[:7])>=0 and 100*sum(last[:3])<=2850.
        records.append(dict(trial=label,residual_sign=sign,initial_state=initial.tolist(),
            checkpoint_class_points_with_independent_90digit_queue_contained=True,checkpoint_count=point_checks,
            raw_double_ode_boundary_overshoots=raw_overshoots,
            maximum_double_ode_queue_reference_difference_vehicles=queue_correction,
            max_checkpoint_occupancy=maximum_ratio,min_postinitial_metric_slack=minimum_slack,
            independent_integrator_convergence_vehicles=maximum_error,conservation_error_vehicles=balance,
            deadline_region1_vehicles=100*float(sum(last[:3])),terminal_queue_vehicles=100*float(last[6])))
    return dict(file=path.name,proof_id=cert['proof_id'],trials=records,
                numerical_checks_are_not_reachability_proofs=True)


def run(paths,with_replays=True):
    start_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'reproducibility/mfd_incident_positive_reachability.py',ROOT/'reproducibility/mfd_incident_queue_bounds.py']}
    summaries=[]
    for path in paths:
        row=audit_case(path); summaries.append(row)
        print(json.dumps(dict(audited=path.name,steps=row['interval_steps'],accepted=row['accepted'])),flush=True)
    negatives=negative_tests()
    replays=[]
    if with_replays:
        preferred=[p for p in paths if read(p).get('accepted') and read(p)['variant']=='positive_queue']
        selected=[]
        for width,residual in ((1.,60.),(1.,300.)):
            candidates=[p for p in preferred if read(p)['state_half_width_vehicles']==width and read(p)['residual_vehicles_h']==residual]
            if candidates:selected.append(sorted(candidates,key=lambda p:('v4' in p.name,p.name))[-1])
        joints=[p for p in preferred if read(p)['state_half_width_vehicles']>1. and read(p)['residual_vehicles_h']>60.]
        if joints:selected.append(max(joints,key=lambda p:read(p)['state_half_width_vehicles']*read(p)['residual_vehicles_h']))
        for path in selected:
            replays.append(replay_case(path)); print(json.dumps(dict(replayed=path.name,trials=9)),flush=True)
    assert all(sha(ROOT/name)==value for name,value in start_hashes.items()), 'Source modified while audit was running'
    return dict(status='PASS',source_sha256=start_hashes,input_proofs=[str(p.relative_to(ROOT)) for p in paths],
        proof_count=len(summaries),interval_steps=sum(r['interval_steps'] for r in summaries),
        independently_recomputed_original_steps=sum(r.get('independently_recomputed_original_steps',0) for r in summaries),
        exact_rational_endpoint_contractors=sum(r.get('exact_rational_endpoint_contractors',0) for r in summaries),
        independently_recomputed_scalar_queue_steps=sum(r.get('independently_recomputed_scalar_queue_steps',0) for r in summaries),
        source_snapshot_checks=sum(r.get('source_snapshots_checked',0) for r in summaries),
        negative_tests=negatives,proof_audits=summaries,replays=replays,
        numerical_trajectories=sum(len(r['trials']) for r in replays),
        proof_and_diagnostic_scopes_separate=True,no_live_llm_calls=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files',nargs='*',type=Path)
    parser.add_argument('--skip-replays',action='store_true')
    parser.add_argument('--output',type=Path,default=RESULTS/'mfd_positive_reachability_independent_audit_results.json')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    paths=args.files or sorted(RESULTS.glob('mfd_positive_*_v[1234].json'))
    paths+=[] if args.files else sorted(RESULTS.glob('mfd_positive_*_axis_v4.json'))+sorted(RESULTS.glob('mfd_positive_*_joint_v4.json'))+sorted(RESULTS.glob('mfd_positive_*_joint_v5.json'))
    paths=sorted(set(p.resolve() for p in paths))
    if not paths: raise ValueError('No proof files selected')
    payload=run(paths,not args.skip_replays)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({k:payload[k] for k in ('status','proof_count','interval_steps','source_snapshot_checks','numerical_trajectories')},indent=2))


if __name__=='__main__':main()
