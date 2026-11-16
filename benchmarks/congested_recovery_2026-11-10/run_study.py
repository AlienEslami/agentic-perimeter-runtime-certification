"""Congested initial-state study. Existing model and verifier are imported unchanged."""
from pathlib import Path
import sys, json, time, hashlib
from fractions import Fraction
import numpy as np
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
SUITE=HERE.parent/'reusable_perimeter_suite_2026-09-26'
sys.path.insert(0,str(SUITE))
import common
from common import plant, mpc, save, compile_commands
from mfd_incident_service_frontier import ServicePrediction
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_affine_reachability import AffineGeneratorVerifier
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
X0=plant.BASE_STATE.copy()
X0[[0,2,4]]*=45/X0[0]
X0[[1,3,5]]*=45/X0[1]
X0[6]=5
WIDTH=np.array([.25]*6+[.10])
DUR=[Fraction(1,12),Fraction(11,12)]
LIMIT=3300.
def forecast(t): return plant.arrival(t,amplitude=.3,duration=1.)

def plan(name,robust=False):
    path=OUT/(name+'_planning.json')
    if path.exists(): return json.loads(path.read_text())
    # Planning clearance is a heuristic only. Certification uses the original limits.
    pred=ServicePrediction(X0+WIDTH if robust else X0,0.,1.,
        (lambda t:forecast(t)+3) if robust else forecast,
        service_target_vehicles=3000 if robust else LIMIT,deadline_h=1.,
        cost_row=np.array([1.,1. if robust else 4.,0,0,0,0,1.]))
    def constraint(v):
        c=pred.evaluate(v)['constraints'].copy()
        if robust:
            # Every 12-row state block has the exterior-storage row at index 11.
            # Initial 12 rows, then 8 input rows, then 4 state blocks per RK step.
            c[11]-=4
            c[20+11:-1:12]-=4
        return c
    starts=[np.tile(plant.BASE_COMMAND,2),np.tile([.8,.6,.65,.85],2),np.tile([.9,.4,.7,1.],2)]
    rows=[]
    for guess in starts:
        t=time.perf_counter()
        r=minimize(pred.objective,guess,jac=True,method='SLSQP',bounds=[(0.,1.)]*8,
            constraints=[dict(type='ineq',fun=constraint,jac=pred.constraint_jacobian)],
            options=dict(maxiter=150,ftol=1e-8))
        row=dict(success=bool(r.success),message=str(r.message),iterations=int(r.nit),
            minimum_margin=float(min(constraint(r.x))),cost_veh_h=float(100*pred.evaluate(r.x)['cost']),
            commands=r.x.reshape(2,4).tolist(),seconds=time.perf_counter()-t)
        rows.append(row); print(name,json.dumps(row),flush=True)
    valid=[r for r in rows if r['success'] and r['minimum_margin']>=-1e-6]
    best=min(valid,key=lambda r:r['cost_veh_h']) if valid else None
    rec=dict(attempts=rows,selected=best)
    if best:
        u,change=compile_commands(best['commands']);rec.update(commands=u.tolist(),compilation_change=change)
    save(path,rec);return rec

def certify(name,commands):
    path=OUT/(name+'_certificate.json')
    if path.exists(): return json.loads(path.read_text())
    print('Certifying '+name,flush=True)
    c=verify_command_plan(X0,np.asarray(commands),DUR,initial_time_h=Fraction(0),
        amplitude=.3,duration=1.,initial_box=WIDTH,residual_bound=3.,step_seconds=15.,
        verifier=AffineGeneratorVerifier(),retain_steps=True,
        service_checks=[dict(time_h=Fraction(1),coordinate=0,upper=LIMIT/100)])
    save(path,c)
    print(json.dumps(dict(name=name,accepted=c['accepted'],seconds=c['wall_seconds'],
        until=c['certified_until_h'],service=c['service_check_results'],failures=c['failures'])),flush=True)
    return c

def main():
    save(HERE/'protocol.json',dict(initial_state_vehicles=(100*X0).tolist(),
        state_order=['n1','n2','n11','n22','n10','n20','z'],initial_error_bounds_vehicles=(100*WIDTH).tolist(),
        arrival_error_bound_veh_h=300,forecast='6000 veh/h with 30% trapezoidal increase, starts minute 15, 5-minute ramp, 60-minute duration',
        horizon_minutes=60,block_minutes=[5,55],queue_capacity_vehicles=2000,regional_caps_vehicles=[10000,10000],
        service_requirement='n1(60 min) <= 3300 vehicles',proposal_objective='integral(n1+4*n2+z) dt',
        critical_accumulation_vehicles=3391.93080685,
        note='Exploratory new initial-state study. No live LLM. Existing polynomial, dynamics, physical constraints and certification code unchanged.',
        source_hashes=common.portable_source_hashes()))
    for name,robust in [('proposal',False),('backup',True)]:
        p=plan(name,robust)
        if p['selected']: certify(name,p['commands'])

if __name__=='__main__':main()
