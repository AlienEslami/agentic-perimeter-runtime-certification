"""Structural positivity and outward conservation-box contraction, new variant.

The original continuous error enclosure is unchanged. Nonnegativity follows
independently from the compartment equations, exact positive speed polynomial,
nonnegative commands/demands/arrival and a nonnegative initial information set.
Only these independently certified lower rows may replace raw box supports.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import mfd_incident_model as model
from mfd_incident_mpc_certification import ConstantCommandVerifier, verify_command_plan
from mfd_incident_verifier import I, imat, inverse_interval, norm_upper, transformed_jacobian_record
from mfd_incident_uncertainty_frontier import initial_supports

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'reproducibility/results'
CLASS=np.array([[0,0,1,0,0,0,0],[1,0,-1,0,-1,0,0],[0,0,0,0,1,0,0],
                [0,0,0,1,0,0,0],[0,1,0,-1,0,-1,0],[0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1]],dtype=float)
INVERSE=np.array([[1,1,1,0,0,0,0],[0,0,0,1,1,1,-1],
                  [1,0,0,0,0,0,0],[0,0,0,1,0,0,-.5],
                  [0,0,1,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],dtype=float)


def frac(value): return Fraction.from_float(float(value))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_positivity(center,widths,commands,residual):
    a,b,c=map(frac,(model.original.MFD_A,model.original.MFD_B,model.original.MFD_C))
    discriminant=4*a*c-b*b
    lower=[sum(frac(v)*frac(x) for v,x in zip(row,center))-
           sum(abs(frac(v))*frac(w) for v,w in zip(row,widths)) for row in CLASS]
    demands=list(map(frac,model.INTERNAL_DEMAND))
    arrival_lower=frac(model.BASE_ARRIVAL)-frac(residual)
    signs=all(np.array_equal(m.gain,np.zeros((4,7))) and np.all(m.feedforward>=0)
              and np.all(np.isfinite(m.feedforward)) for m in commands)
    passed=a>0 and discriminant>0 and min(lower)>=0 and min(demands)>=0 and arrival_lower>=0 and signs
    return dict(passed=bool(passed),class_order=['n11','n12','n10','n22','n21','n20','queue'],
        initial_class_lower_exact=list(map(str,lower)),speed_4ac_minus_b_squared_exact=str(discriminant),
        global_speed_minimum_exact=str(discriminant/(4*a)),class_demand_nonnegative=min(demands)>=0,
        offered_arrival_lower_exact=str(arrival_lower),zero_gain_nonnegative_commands=signs,
        argument='Quasipositive class balances; total inventory derivative is bounded by total offered inflow. No jam/storage assumption.',
        lower_rows_certified=['n1_lower','n2_lower','n11_lower','n22_lower','n10_lower','n20_lower','exterior_queue_lower','n12_lower','n21_lower'])


class PositiveVerifier(ConstantCommandVerifier):
    def __init__(self,variant='structural_only'):
        super().__init__()
        if variant not in ('structural_only','positive_rebox','positive_queue','strip_queue'): raise ValueError(variant)
        self.variant=variant
        self.structural=None

    def geometry(self,mode):
        geo=super().geometry(mode)
        if self.variant=='strip_queue' and not geo.get('destination_strip'):
            L=geo['L'].copy();L[1]-=2*L[2]
            inverse=inverse_interval(L)
            geo.update(L=L,T=inverse,destination_strip=True,
                coordinate_support=np.array([norm_upper(inverse[i]) for i in range(7)]),
                support=np.array([norm_upper(imat(geo['rows'][i:i+1],inverse)) for i in range(len(geo['names']))]),
                constraint_box_map=imat(geo['rows'],inverse).mag(),
                disturbance_factor=norm_upper(I(L[:,6])),
                jacobian_record=transformed_jacobian_record(mode,L,inverse),
                metric_selection=dict(strategy='destination_strip_conservation_box',
                    coordinates=['n11','n12-2n10','n10','n22+queue/2','n21+queue/2','n20','queue']))
        return geo

    def verify_plan(self,initial_state,modes,durations_h=None,**kwargs):
        if kwargs.get('initial_shape') is not None:
            raise ValueError('This new first-plan adapter requires an explicit physical initial box')
        if kwargs.get('amplitude',.3)<0: raise ValueError('Nonnegative pulse amplitude required')
        widths=np.broadcast_to(np.asarray(kwargs.get('initial_box',.01),dtype=float),(7,))
        self.structural=initial_positivity(np.asarray(initial_state),widths,modes,kwargs.get('residual_bound',.6))
        if not self.structural['passed']: raise ValueError(('positivity assumptions not verified',self.structural))
        queue=I(float(initial_state[6]))+I(-widths[6],widths[6])
        self.queue_box=[max(0.,float(queue.lo)),float(queue.hi)]
        result=super().verify_plan(initial_state,modes,durations_h,**kwargs)
        result['proof_descriptor']['structural_positive_extension']=dict(variant=self.variant,
            source_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'POSITIVE_REACHABILITY_PROTOCOL.md'),
            exact_assumptions=self.structural,upper_requirements_unchanged=True,
            initial_physical_information=dict(set_type='axis_aligned_physical_box',
                center=np.asarray(initial_state,dtype=float).tolist(),half_widths=widths.tolist(),
                exact_center=[str(frac(v)) for v in initial_state],exact_half_widths=[str(frac(v)) for v in widths]),
            independently_tightened_upper_supports=['exterior_queue_upper'] if self.variant in ('positive_queue','strip_queue') else [],
            queue_source_sha256=sha(ROOT/'reproducibility/mfd_incident_queue_bounds.py') if self.variant in ('positive_queue','strip_queue') else None,
            destination_strip_addendum_sha256=sha(ROOT/'DESTINATION_STRIP_ADDENDUM.md') if self.variant=='strip_queue' else None,
            endpoint_rule='box intersect positive class cone and optional autonomous queue bound then outward rebox' if self.variant!='structural_only' else 'no contraction')
        result['proof_id']=hashlib.sha256(json.dumps(result['proof_descriptor'],sort_keys=True).encode()).hexdigest()
        for checkpoint in result['reachable_checkpoints']: checkpoint['proof_id']=result['proof_id']
        result['structural_positive_extension']=result['proof_descriptor']['structural_positive_extension']
        return result

    def step(self,center,radius,mode,left,right,pulse,residual_bound):
        new_center,new_radius,record=super().step(center,radius,mode,left,right,pulse,residual_bound)
        raw=record['continuous_constraint_margins_lower'][:9]
        record['continuous_constraint_margins_lower'][:9]=[max(0.,v) for v in raw]
        record['structural_positivity']=dict(independently_verified=True,raw_lower_supports=raw,
            replaced_rows=self.structural['lower_rows_certified'],regional_upper_supports_unchanged=True)
        queue=None
        if self.variant in ('positive_queue','strip_queue'):
            from mfd_incident_queue_bounds import queue_step_bounds
            a,slope=pulse.affine(left,right)
            queue=queue_step_bounds(*self.queue_box,right-left,a,slope,residual_bound,mode.feedforward[2])
            self.queue_box=queue['endpoint']
            record['autonomous_queue_bound']=queue
            index=self.geometry(mode)['names'].index('exterior_queue_upper')
            record['continuous_constraint_margins_lower'][index]=max(record['continuous_constraint_margins_lower'][index],
                float((I(model.QUEUE_LIMIT)-I(queue['continuous'][1])).lo))
        if self.variant!='structural_only':
            L=self.geometry(mode)['L']
            box=imat(L,I(new_center))+I(-new_radius,new_radius)
            lower,upper=box.lo.copy(),box.hi.copy()
            positive_indices=[0,2,5,6] if self.variant=='strip_queue' else [0,1,2,5,6]
            lower[positive_indices]=np.maximum(lower[positive_indices],0.)
            if self.variant=='strip_queue':
                lower[1]=max(lower[1],float((I(-2.)*I(upper[2])).lo))
                lower[2]=max(lower[2],float((I(-.5)*I(upper[1])).lo))
            if queue is not None:
                lower[6]=max(lower[6],queue['endpoint'][0]); upper[6]=min(upper[6],queue['endpoint'][1])
            # y3=n22+z/2 and y4=n21+z/2; both classes are nonnegative.
            lower[[3,4]]=np.maximum(lower[[3,4]],float((I(.5)*I(lower[6])).lo))
            if np.any(lower>upper): raise AssertionError('Empty intersection contradicts the established enclosure')
            contracted=I(lower,upper)
            inverse=INVERSE.copy()
            if self.variant=='strip_queue': inverse[0,2]=3.
            new_center=imat(inverse,I(contracted.mid())).mid()
            transformed_center=imat(L,I(new_center))
            new_radius=(contracted-transformed_center).mag()
            assert np.all(new_radius>=0)
            record['endpoint_contraction']=dict(raw_lower=box.lo.tolist(),raw_upper=box.hi.tolist(),
                contracted_lower=lower.tolist(),contracted_upper=upper.tolist(),
                returned_center=new_center.tolist(),returned_radius=new_radius.tolist(),
                inclusion_rule='Safe coordinate-wise outer contraction using the positive class cone; no nominal-radius reset')
            record['radius']=new_radius.tolist()
        return new_center,new_radius,record


def sources():
    names=['reproducibility/mfd_incident_positive_reachability.py','POSITIVE_REACHABILITY_PROTOCOL.md','DESTINATION_STRIP_ADDENDUM.md',
        'reproducibility/mfd_incident_model.py','reproducibility/mfd_cubic_pilot.py',
        'reproducibility/mfd_incident_verifier.py','reproducibility/mfd_incident_mpc_certification.py',
        'reproducibility/mfd_deadline_certificates.py','reproducibility/mfd_incident_queue_bounds.py',
        'reproducibility/results/mfd_incident_mpc_grid_results.json']
    out={}
    for name in names:
        path=ROOT/name; digest=sha(path); out[name]=digest
        if not name.endswith('_results.json'):
            snapshot=RESULTS/'pr'/digest[:16]/path.name
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists(): shutil.copy2(path,snapshot)
            assert sha(snapshot)==digest
    return out


def run(variant,width,residual,tag,step):
    output=RESULTS/f'mfd_positive_{variant}_v{width:g}_r{residual:g}_{tag}.json'
    if output.exists(): raise FileExistsError(output)
    hashes=sources()
    archive=json.loads((RESULTS/'mfd_incident_mpc_grid_results.json').read_text())
    primary=next(row for row in archive['runs'] if row['amplitude']==.3 and row['duration_h']==1.)['decisions'][0]
    center=np.array(primary['initial_state']); commands=primary['controls']
    widths=np.array([width]*6+[min(width,10.)])/100.
    initial=initial_supports(center,commands[0],widths)
    if not initial['valid']:
        result=dict(accepted=False,classification='invalid_initial_information_set',initial=initial)
    else:
        certificate=verify_command_plan(center,commands,[Fraction(1,12),Fraction(11,12)],
            initial_box=widths,residual_bound=residual/100.,step_seconds=step,
            verifier=PositiveVerifier(variant),retain_steps=True)
        result=dict(accepted=certificate['accepted'],certificate=certificate,initial=initial)
    result.update(variant=variant,tag=tag,state_half_width_vehicles=width,queue_half_width_vehicles=min(width,10.),
        residual_vehicles_h=residual,step_seconds=step,source_sha256=hashes)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    cert=result.get('certificate',{})
    print(json.dumps(dict(file=output.name,accepted=result['accepted'],until=cert.get('certified_until_h'),
        seconds=cert.get('wall_seconds'),service=cert.get('service_check_results'),
        failures=[{k:f[k] for k in ('kind','constraint','time_h') if k in f} for f in cert.get('failures',[])])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['structural_only','positive_rebox','positive_queue','strip_queue'],required=True)
    p.add_argument('--width',type=float,default=1.);p.add_argument('--residual',type=float,default=60.)
    p.add_argument('--step',type=float,default=15.);p.add_argument('--tag',default='v1')
    args=p.parse_args();run(args.variant,args.width,args.residual,args.tag,args.step)
