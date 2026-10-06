"""Shared physical model, exact command compilation and ensemble evaluation."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, LinearConstraint, minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import mfd_incident_model as plant
import mfd_incident_mpc as mpc
from mfd_incident_service_frontier import ServicePrediction, solve_service
from mfd_incident_affine_reachability import AffineGeneratorVerifier
import mfd_incident_affine_reachability as affine_module
from mfd_incident_mpc_certification import verify_command_plan, exact_command_constraints

START = Fraction(1, 3)
END = Fraction(1)
DURATIONS = [Fraction(1, 12), Fraction(7, 12)]
PROFILES = ["zero", "constant_high", "constant_low", "alternating_5min", "sinusoidal_10min"]

def portable_source_hashes():
    """Replace only archived path lookup, preserving every numerical routine."""
    modules=("mfd_incident_affine_reachability.py","mfd_incident_model.py","mfd_cubic_pilot.py",
        "mfd_incident_verifier.py","mfd_incident_mpc_certification.py","mfd_deadline_certificates.py",
        "mfd_incident_positive_reachability.py","mfd_incident_queue_bounds.py","mfd_incident_uncertainty_frontier.py")
    paths=[HERE/"vendor"/name for name in modules]+[HERE/"PROTOCOL.md",Path(__file__),HERE/"anchor.json"]
    return {path.relative_to(HERE).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

affine_module.source_hashes=portable_source_hashes
affine_module.PROTOCOL_PATH=HERE/"PROTOCOL.md"

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

def fixture():
    return json.loads((HERE / "anchor.json").read_text(encoding="utf-8"))

def compile_commands(commands):
    """One compilation rule for every planning tool; leave numerical slack.

    The change is bounded by 2e-8 in a gate fraction and is recorded. Exact
    dyadic actuator checks then verify the commands actually simulated.
    """
    raw = np.asarray(commands, dtype=float).reshape(2, 4)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Non-finite command")
    compiled = np.round(np.clip(raw, 0, 1), 9)
    for row in compiled:
        for left, right in ((0, 1), (2, 3)):
            difference = row[right] - row[left]
            if abs(difference) >= .75 - 1e-9:
                middle = .5 * (row[right] + row[left])
                half = np.sign(difference) * (.375 - 1e-8)
                row[left], row[right] = middle - half, middle + half
    assert all(exact_command_constraints(row)["passed"] for row in compiled)
    return compiled, float(np.max(abs(compiled - raw)))

def certify(commands, contract):
    checks = ([] if contract["deadline_vehicles"] is None else
              [dict(time_h=END, coordinate=0, upper=contract["deadline_vehicles"] / 100)])
    record = verify_command_plan(np.asarray(fixture()["center"]), commands, DURATIONS,
        initial_time_h=START, amplitude=contract["amplitude"], duration=contract["duration_h"],
        initial_box=contract["width_vehicles"] / 100,
        residual_bound=contract["residual_veh_h"] / 100, step_seconds=15,
        verifier=AffineGeneratorVerifier(), service_checks=checks, retain_steps=False)
    # Retain all certificate diagnostics, including rejected supports.
    return record

def residual(t, bound, profile):
    if profile == "zero": return 0.0
    if profile == "constant_high": return bound
    if profile == "constant_low": return -bound
    if profile == "alternating_5min":
        return bound if int(math.floor((t - float(START) + 1e-10) * 12)) % 2 == 0 else -bound
    return bound * math.sin(12 * math.pi * (t - float(START)))

def arrival(t, contract):
    return plant.arrival(t, amplitude=contract["amplitude"], duration=contract["duration_h"])

def batch_field(x, u, offered):
    """Vectorized, algebraically identical seven-state conservation model."""
    n1,n2,n11,n22,n10,n20,z = x.T
    u12,u21,u02,u20 = np.asarray(u).T
    h1=.0014877*n1*n1-.29815*n1+15.0912
    h2=.0014877*n2*n2-.29815*n2+15.0912
    q11,q12,q10=n11*h1,(n1-n11-n10)*h1,n10*h1
    q22,q21,q20=n22*h2,(n2-n22-n20)*h2,n20*h2
    admitted=np.minimum(60*z,120*u02)
    d11,d12,d10,d22,d21,d20=plant.INTERNAL_DEMAND
    f=np.stack((-q11-(q12+q10)*u12+q21*u21+d11+d12+d10,
        -q22+(q12+q10)*u12-q21*u21+admitted-q20*u20+d22+d21+d20,
        -q11+q21*u21+d11,-q22+q12*u12+.5*admitted+d22,
        -q10*u12+d10,-q20*u20+q10*u12+d20,offered-admitted),axis=1)
    departure=q11+q22+q20*u20
    return f,departure

def ensemble(commands, contract, *, step_seconds=10, signs=None):
    corners=np.asarray(list(itertools.product((-1.,1.),repeat=7)) if signs is None else signs)
    x=np.tile(np.asarray(fixture()["center"]),(len(corners)*5,1))
    x+=np.repeat(corners,5,axis=0)*contract["width_vehicles"]/100
    initial=x.copy()
    totals=np.zeros((len(x),4))  # TTS, queue, completions, priority
    minimum=np.full(len(x),np.inf)
    peak=x[:,6].copy()
    h=step_seconds/3600
    steps=round(float(END-START)/h)
    profile_index=np.tile(np.arange(5),len(corners))
    offered_integral=np.zeros(len(x))
    for k in range(steps):
        t=float(START)+k*h
        u=commands[0 if k*h < float(DURATIONS[0])-1e-10 else 1]
        def rhs(y,clock):
            forcing_clock=min(clock,np.nextafter(t+h,t))
            r=np.asarray([residual(t+h/2 if p=="alternating_5min" else forcing_clock,
                contract["residual_veh_h"]/100,p) for p in PROFILES])[profile_index]
            offer=arrival(clock,contract)+r
            f,dep=batch_field(y,u,offer)
            c=np.stack((y[:,0]+y[:,1]+y[:,6],y[:,6],dep,
                        y[:,0]+contract["weight"]*y[:,1]+y[:,6]),axis=1)*100
            return f,c,100*(sum(plant.INTERNAL_DEMAND)+offer)
        f1,c1,a1=rhs(x,t); x2=x+h*f1/2
        f2,c2,a2=rhs(x2,t+h/2); x3=x+h*f2/2
        f3,c3,a3=rhs(x3,t+h/2); x4=x+h*f3
        f4,c4,a4=rhs(x4,t+h)
        new=x+h/6*(f1+2*f2+2*f3+f4)
        totals+=h/6*(c1+2*c2+2*c3+c4)
        offered_integral+=h/6*(a1+2*a2+2*a3+a4)
        for y in (x,x2,x3,x4,new):
            minimum=np.minimum(minimum,np.min(y@mpc.STATE_A.T+mpc.STATE_B,axis=1))
            peak=np.maximum(peak,y[:,6])
        x=new
    balance=100*((x[:,0]+x[:,1]+x[:,6])-(initial[:,0]+initial[:,1]+initial[:,6]))-offered_integral+totals[:,2]
    result=[]
    for i in range(len(x)):
        limit=contract["deadline_vehicles"]
        result.append(dict(corner="".join("+" if s>0 else "-" for s in corners[i//5]),
            profile=PROFILES[i%5],tts_veh_h=float(totals[i,0]),queue_veh_h=float(totals[i,1]),
            completed_trips=float(totals[i,2]),priority_cost=float(totals[i,3]),
            peak_queue_vehicles=float(100*peak[i]),deadline_n1_vehicles=float(100*x[i,0]),
            physical_pass=bool(minimum[i]>=-1e-7),minimum_state_margin=float(minimum[i]),
            service_pass=None if limit is None else bool(100*x[i,0]<=limit+1e-6),
            conservation_residual_vehicles=float(balance[i])))
    return result

def summarize(rows):
    def stats(key):
        values=np.asarray([r[key] for r in rows])
        return dict(minimum=float(values.min()),median=float(np.median(values)),maximum=float(values.max()))
    return dict(runs=len(rows),physical_passes=sum(r["physical_pass"] for r in rows),
        service_passes=None if rows[0]["service_pass"] is None else sum(r["service_pass"] for r in rows),
        **{key:stats(key) for key in ("tts_veh_h","queue_veh_h","completed_trips","priority_cost",
                    "peak_queue_vehicles","deadline_n1_vehicles")},
        maximum_conservation_error_vehicles=max(abs(r["conservation_residual_vehicles"]) for r in rows))

class ClfQueue:
    def __init__(self):
        self.mode=plant.synthesize("region1_protection",27,32,(4,1,.1,.1,.1,.1,2),3000)
    def command(self,x,offered):
        reference=self.mode.feedforward
        u=self.mode.command(x)
        error=100*(x[:2]-self.mode.center[:2])
        drift=100/3600*plant.field(x,reference,offered)[:2]
        n1,n2,n11,n22,n10,n20,_=x
        q12=plant.original.class_flow(n1-n11,n1)
        q21=plant.original.class_flow(n2-n22-n20,n2)
        directions=100/3600*np.array([[-q12,q21],[q12,-q21]])
        alpha=float(error@drift); beta=directions.T@error; squared=float(beta@beta)
        correction=np.zeros(2) if squared<=np.finfo(float).eps else -(alpha+np.sqrt(alpha*alpha+squared*squared))/(squared*(1+np.sqrt(1+squared)))*beta
        u[:2]=reference[:2]+np.clip(correction,-reference[:2],1-reference[:2])
        if abs(u[1]-u[0])>.75:
            middle=.5*(u[0]+u[1]); half=.375*np.sign(u[1]-u[0]); u[:2]=[middle-half,middle+half]
        return u

def cbf(x,nominal,offered,bound,alpha=6):
    candidates=[]; ratio=60*x[6]/120
    for branch in ("capacity","supply"):
        if branch=="supply" and ratio>=1: continue
        ref=np.array([.5,.5,max(0,min(.5,.5*ratio)) if branch=="capacity" else min(1,ratio+.25),.5])
        f,_,fu=mpc.field_jacobians(x,ref,offered)
        constant=f-fu@ref
        cbranch=np.array([0,0,-1 if branch=="capacity" else 1,0.])
        dbranch=-ratio if branch=="capacity" else ratio
        adjustment=np.zeros(12); adjustment[-1]=-bound
        C=np.vstack((mpc.STATE_A@fu,-mpc.PAIR_A,cbranch))
        d=np.r_[-mpc.STATE_A@constant-alpha*(mpc.STATE_A@x+mpc.STATE_B)-adjustment,np.full(4,-.75),dbranch]
        r=minimize(lambda u:.5*float((u-nominal)@(u-nominal)),np.clip(nominal,0,1),
            jac=lambda u:u-nominal,bounds=Bounds(np.zeros(4),np.ones(4)),
            constraints=[LinearConstraint(C,d,np.inf)],method="SLSQP",options=dict(ftol=1e-12,maxiter=100))
        if r.success and min(C@r.x-d)>=-1e-8: candidates.append((r.fun,r.x))
    return None if not candidates else min(candidates,key=lambda r:r[0])[1]

def feedback_run(commands,contract,kind,*,profile="constant_high",signs=None):
    signs=np.ones(7) if signs is None else np.asarray(signs)
    x=np.asarray(fixture()["center"])+signs*contract["width_vehicles"]/100
    initial=x.copy(); cost=np.zeros(4); peak=x[6]; minimum=np.inf; interventions=0; compute=0.; offered_sum=0.
    clf=ClfQueue() if kind=="clf" else None
    h=1/360
    for k in range(240):
        t=float(START)+k*h; nominal=commands[0 if k<30 else 1]
        forecast=arrival(t,contract)
        begin=time.perf_counter()
        u=clf.command(x,forecast) if clf else cbf(x,nominal,forecast,contract["residual_veh_h"]/100)
        compute+=time.perf_counter()-begin
        if u is None: return dict(completed=False,failed_at_minutes=60*t,reason="CBF_QP_infeasible",compute_seconds=compute)
        interventions+=int(max(abs(u-nominal))>1e-7)
        def rhs(y,clock):
            forcing_clock=min(clock,np.nextafter(t+h,t))
            offer=arrival(clock,contract)+residual(t+h/2 if profile=="alternating_5min" else forcing_clock,
                contract["residual_veh_h"]/100,profile)
            f,dep=batch_field(y[None,:],u,offer)
            c=100*np.array([y[0]+y[1]+y[6],y[6],dep[0],y[0]+contract["weight"]*y[1]+y[6]])
            return f[0],c,100*(sum(plant.INTERNAL_DEMAND)+offer)
        f1,c1,a1=rhs(x,t); x2=x+h*f1/2
        f2,c2,a2=rhs(x2,t+h/2); x3=x+h*f2/2
        f3,c3,a3=rhs(x3,t+h/2); x4=x+h*f3
        f4,c4,a4=rhs(x4,t+h)
        new=x+h/6*(f1+2*f2+2*f3+f4); cost+=h/6*(c1+2*c2+2*c3+c4)
        offered_sum+=h/6*(a1+2*a2+2*a3+a4)
        minimum=min(minimum,*[float(min(plant.margins(y,u))) for y in (x,x2,x3,x4,new)])
        peak=max(peak,x2[6],x3[6],x4[6],new[6]); x=new
    limit=contract["deadline_vehicles"]
    balance=100*((x[0]+x[1]+x[6])-(initial[0]+initial[1]+initial[6]))-offered_sum+cost[2]
    return dict(completed=True,corner="".join("+" if s>0 else "-" for s in signs),profile=profile,
        tts_veh_h=float(cost[0]),queue_veh_h=float(cost[1]),completed_trips=float(cost[2]),priority_cost=float(cost[3]),
        peak_queue_vehicles=float(100*peak),deadline_n1_vehicles=float(100*x[0]),physical_pass=bool(minimum>=-1e-7),
        service_pass=None if limit is None else bool(100*x[0]<=limit+1e-6),
        conservation_residual_vehicles=float(balance),minimum_state_margin=float(minimum),
        interventions=interventions,compute_seconds=compute)

def adaptive_replay(commands,contract,*,signs=None,profile="constant_high"):
    signs=np.ones(7) if signs is None else np.asarray(signs)
    initial=np.asarray(fixture()["center"])+signs*contract["width_vehicles"]/100
    y=np.r_[initial,np.zeros(5)]; minimum=np.inf; peak=y[6]
    # Split known command, pulse and residual discontinuities exactly.
    cuts=sorted(set([float(START),float(START+DURATIONS[0]),1.,.5,.75,5/6]+[float(START)+k/12 for k in range(1,8)]))
    cuts=[t for t in cuts if float(START)<=t<=1]
    for left,right in zip(cuts[:-1],cuts[1:]):
        u=commands[0 if left<float(START+DURATIONS[0])-1e-10 else 1]
        def rhs(t,yy):
            # one-sided forcing at right endpoints of alternating intervals
            tt=min(t,right-1e-9) if profile=="alternating_5min" else t
            offer=arrival(tt,contract)+residual(tt,contract["residual_veh_h"]/100,profile)
            f=plant.field(yy[:7],u,offer)
            return np.r_[f,100*(yy[0]+yy[1]+yy[6]),100*yy[6],100*plant.departure_rate(yy[:7],u),
                100*(yy[0]+contract["weight"]*yy[1]+yy[6]),100*(sum(plant.INTERNAL_DEMAND)+offer)]
        sol=solve_ivp(rhs,(left,right),y,method="DOP853",rtol=1e-11,atol=1e-12,max_step=1/360,dense_output=True)
        assert sol.success,sol.message
        samples=sol.sol(np.linspace(left,right,round((right-left)*1800)+1))[:7].T
        minimum=min(minimum,float(np.min(samples@mpc.STATE_A.T+mpc.STATE_B)))
        peak=max(peak,float(max(samples[:,6]))); y=sol.y[:,-1]
    return dict(tts_veh_h=float(y[7]),queue_veh_h=float(y[8]),completed_trips=float(y[9]),priority_cost=float(y[10]),
        peak_queue_vehicles=100*peak,deadline_n1_vehicles=100*float(y[0]),minimum_state_margin=minimum,
        conservation_residual_vehicles=float(100*((y[0]+y[1]+y[6])-(initial[0]+initial[1]+initial[6]))-y[11]+y[9]))
