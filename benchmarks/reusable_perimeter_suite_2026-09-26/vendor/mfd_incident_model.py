"""Seven-state queue-aware MFD commissioning model, distinct from old benchmark.

Units: 100 vehicles and hours. Every policy shares INTERNAL_DEMAND and offered
arrival histories. Only an external staging queue is added; it is disjoint from
regional accumulations. Queue admission is availability/capacity limited.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
import mfd_cubic_pilot as original

ROOT=Path(__file__).resolve().parents[1]
CAPACITY=120.0
RELEASE_RATE=60.0
BASE_ARRIVAL=60.0
QUEUE_TARGET=2.0
QUEUE_LIMIT=20.0
STATE_ORDER=(*original.STATE_ORDER,'exterior_queue')
INTERNAL_DEMAND=original.DEMAND.copy()
INTERNAL_DEMAND[3]-=29.0
INTERNAL_DEMAND[4]-=29.0
BASE_STATE=np.r_[original.X_STAR,QUEUE_TARGET]
BASE_COMMAND=original.U_STAR.copy()


def field(x,u,arrival=BASE_ARRIVAL,capacity_scale=1.0):
    """Physical dynamics; do not clip states or replace offered with admitted flow."""
    n1,n2,n11,n22,n10,n20,z=x
    u12,u21,u02,u20=u
    n12=n1-n11-n10; n21=n2-n22-n20
    q11,q12,q10=[original.class_flow(v,n1)*capacity_scale for v in (n11,n12,n10)]
    q22,q21,q20=[original.class_flow(v,n2)*capacity_scale for v in (n22,n21,n20)]
    supply=RELEASE_RATE*z; service=CAPACITY*u02
    # Branch selection also supports complex-step differentiation away from ties.
    admission=supply if np.real(supply)<np.real(service) else service
    d11,d12,d10,d22,d21,d20=INTERNAL_DEMAND
    return np.asarray([
        -q11-(q12+q10)*u12+q21*u21+d11+d12+d10,
        -q22+(q12+q10)*u12-q21*u21+admission-q20*u20+d22+d21+d20,
        -q11+q21*u21+d11,
        -q22+q12*u12+.5*admission+d22,
        -q10*u12+d10,
        -q20*u20+q10*u12+d20,
        arrival-admission,
    ])


def departure_rate(x,u,capacity_scale=1.0):
    return capacity_scale*(original.class_flow(x[2],x[0])+
                           original.class_flow(x[3],x[1])+
                           original.class_flow(x[5],x[1])*u[3])


def equilibrium(n1,n2,arrival=BASE_ARRIVAL,queue=QUEUE_TARGET,u20=.52):
    """Analytic target construction with the same fixed INTERNAL_DEMAND.

    Admission equals offered arrival at equilibrium. A different arrival is a
    declared operating regime, never a mode-dependent choice of physical demand.
    """
    d11,d12,d10,d22,d21,d20=INTERNAL_DEMAND
    h1=original.class_flow(1.,n1); h2=original.class_flow(1.,n2)
    q11=d11+d21+.5*arrival
    q22=d22+d12+.5*arrival
    n11=q11/h1; n22=q22/h2
    u12=(d12+d10)/(original.mfd_outflow(n1)-q11)
    n10=d10/(u12*h1); n20=(d10+d20)/(u20*h2)
    n21=n2-n22-n20
    u21=(d21+.5*arrival)/(n21*h2)
    return np.array([n1,n2,n11,n22,n10,n20,queue]),np.array([u12,u21,arrival/CAPACITY,u20])


def margins(x,u):
    """All physical affine constraints; the admission branch is not a constraint."""
    n1,n2,n11,n22,n10,n20,z=x
    return np.r_[x,[n1-n11-n10,n2-n22-n20,100-n1,100-n2,QUEUE_LIMIT-z],
                 u,1-u,.75-(u[1]-u[0]),.75+(u[1]-u[0]),
                 .75-(u[3]-u[2]),.75+(u[3]-u[2])]


def affine_constraints(center,gain,feedforward):
    """Return a,b with a@x<=b for full physical and feedback command constraints."""
    rows=[]; bounds=[]
    for j in range(7):
        row=np.zeros(7); row[j]=-1; rows.append(row);bounds.append(0.)
    for indices in ((0,2,4),(1,3,5)):
        row=np.zeros(7);row[indices[0]]=-1;row[indices[1]]=row[indices[2]]=1
        rows.append(row);bounds.append(0.)
    for j,limit in ((0,100.),(1,100.),(6,QUEUE_LIMIT)):
        row=np.zeros(7);row[j]=1;rows.append(row);bounds.append(limit)
    offset=feedforward-gain@center
    for j in range(4):
        rows.extend([gain[j],-gain[j]]);bounds.extend([1-offset[j],offset[j]])
    for j,k in ((1,0),(3,2)):
        rows.extend([gain[j]-gain[k],gain[k]-gain[j]])
        bounds.extend([.75-offset[j]+offset[k],.75-offset[k]+offset[j]])
    return np.array(rows),np.array(bounds)


@dataclass
class Mode:
    name:str
    center:np.ndarray
    feedforward:np.ndarray
    gain:np.ndarray
    A:np.ndarray
    P:np.ndarray

    def command(self,x): return self.feedforward+self.gain@(x-self.center)
    def record(self):
        reference_arrival=float(min(RELEASE_RATE*self.center[6],CAPACITY*self.feedforward[2]))
        return dict(name=self.name,center=self.center.tolist(),feedforward=self.feedforward.tolist(),
                    gain=self.gain.tolist(),A=self.A.tolist(),P=self.P.tolist(),
                    reference_arrival=reference_arrival,
                    spectral_abscissa=float(np.max(np.real(np.linalg.eigvals(self.A)))),
                    equilibrium_residual=float(np.max(abs(field(self.center,self.feedforward,reference_arrival)))),
                    residual_under_base_arrival=float(np.max(abs(field(self.center,self.feedforward,BASE_ARRIVAL)))),
                    minimum_equilibrium_margin=float(min(margins(self.center,self.feedforward))))


def synthesize(name,n1,n2,weights=(1.,1.,.1,.1,.1,.1,2.),penalty=3000.):
    center,feedforward=equilibrium(n1,n2)
    A=original.complex_step_jacobian(lambda x:field(x,feedforward),center)
    B=original.complex_step_jacobian(lambda u:field(center,u),feedforward)
    Q=np.diag(weights);R=np.eye(4)*penalty
    S=solve_continuous_are(A,B,Q,R)
    K=-np.linalg.solve(R,B.T@S)
    closed=A+B@K
    P=solve_continuous_lyapunov(closed.T,-np.eye(7))
    return Mode(name,center,feedforward,K,closed,(P+P.T)/2)


def default_library(penalty=3000.):
    return [synthesize('balanced',30.,30.,penalty=penalty),
            synthesize('region1_protection',27.,32.,(4.,1.,.1,.1,.1,.1,2.),penalty),
            synthesize('region2_protection',32.,27.,(1.,4.,.1,.1,.1,.1,2.),penalty)]


def pulse_multiplier(t,amplitude=.3,duration=.5,start=.25,ramp=1/12):
    """Trapezoid: rise+plateau+fall, total support duration, five-minute ramps."""
    local=t-start
    if local<=0 or local>=duration:return 0.
    return amplitude*min(1.,local/ramp,(duration-local)/ramp)


def arrival(t,amplitude=.3,duration=.5,**kwargs):
    return BASE_ARRIVAL*(1+pulse_multiplier(t,amplitude,duration,**kwargs))


def rk4(x,u_fun,t,h,arrival_fun,capacity_scale=1.):
    def f(y,s):return field(y,u_fun(y),arrival_fun(s),capacity_scale)
    k1=f(x,t);k2=f(x+h*k1/2,t+h/2);k3=f(x+h*k2/2,t+h/2);k4=f(x+h*k3,t+h)
    return x+h*(k1+2*k2+2*k3+k4)/6


def simulate_held(mode,amplitude=.3,duration=.5,horizon=2.,initial=None,step=1/360):
    x=BASE_STATE.copy() if initial is None else np.array(initial).copy()
    states=[x.copy()]; times=[0.]; total=queue=departures=0.;margin=np.inf
    for k in range(round(horizon/step)):
        t=k*step;u=mode.command(x)
        margin=min(margin,float(min(margins(x,u))))
        new=rk4(x,mode.command,t,step,lambda s:arrival(s,amplitude,duration))
        total+=step/2*100*(sum(x[[0,1,6]])+sum(new[[0,1,6]]))
        queue+=step/2*100*(x[6]+new[6])
        departures+=step/2*100*(departure_rate(x,u)+departure_rate(new,mode.command(new)))
        x=new;states.append(x.copy());times.append(t+step)
    margin=min(margin,float(min(margins(x,mode.command(x)))))
    a=np.array(states)
    return dict(mode=mode.name,amplitude=amplitude,duration=duration,total_time_spent_veh_h=total,
                queue_time_veh_h=queue,departures=departures,maximum_queue_vehicles=100*max(a[:,6]),
                minimum_margin=margin,final_state=x.tolist(),times=times,states=a.tolist())


if __name__=='__main__':
    library=default_library()
    records=[simulate_held(m,amplitude=a,duration=d) for a in (.2,.3,.4) for d in (.5,1.) for m in library]
    out=ROOT/'reproducibility/results/mfd_incident_model_pilot_results.json'
    out.write_text(json.dumps(dict(model='distinct seven-state boundary-staging commissioning model',
                                  internal_demand=INTERNAL_DEMAND.tolist(),library=[m.record() for m in library],
                                  discovery=True,records=records),indent=2)+'\n')
    for rec in records:
        print(rec['mode'],rec['amplitude'],rec['duration'],round(rec['total_time_spent_veh_h'],2),
              round(rec['maximum_queue_vehicles'],2),round(rec['minimum_margin'],5),flush=True)
