"""Native four-state regulation benchmark with matched analytic-gradient MPC."""
from __future__ import annotations
import dataclasses,itertools,json,time
import numpy as np
from scipy.optimize import minimize
from common import HERE,save
import zhong_model as zhong

TOTAL=np.array([[1.,1,0,0],[0,0,1,1]])
CONSTRAINT=np.vstack((np.eye(4),-TOTAL))
OFFSET=np.r_[np.zeros(4),100.,100.]

def rhs(x,u,demand):
    n=TOTAL@x
    h=.0014877*n*n-.29815*n+15.0912
    hp=.0029754*n-.29815
    q=x*np.repeat(h,2)
    dq=np.diag(np.repeat(h,2))+x[:,None]*np.repeat(hp,2)[:,None]*np.repeat(TOTAL,2,axis=0)
    transfer=np.array([[-1,0,u[1],0],[0,-u[0],0,0],[0,0,-u[1],0],[0,u[0],0,-1.]])
    f=transfer@q+36*demand
    a=transfer@dq
    b=np.array([[0,q[2]],[-q[1],0],[0,-q[2]],[q[1],0]])
    return f,a,b

class Prediction:
    def __init__(self,x,now,demand):
        self.x=x.copy(); self.now=now; self.demand=demand; self.key=None
    def evaluate(self,v):
        if self.key is not None and np.array_equal(v,self.key):return self.rec
        self.key=v.copy(); x=self.x.copy(); s=np.zeros((4,4)); cost=0.; grad=np.zeros(4)
        cs=[CONSTRAINT@x+OFFSET]; js=[np.zeros((6,4))]
        h=1/60
        for k in range(30):
            t=self.now+k*h; block=0 if k<5 else 1; u=v.reshape(2,2)[block]
            embed=np.zeros((2,4)); embed[:,2*block:2*block+2]=np.eye(2)
            def f(y,sy,clock):
                ff,a,b=rhs(y,u,self.demand(clock)); return ff,a@sy+b@embed
            def c(y,sy):
                e=(TOTAL@y-30)/10
                return float(e@e)+.001*float((u-.5267)@(u-.5267)), 2*e/10@TOTAL@sy+.002*(u-.5267)@embed
            f1,s1=f(x,s,t); x2=x+h*f1/2; ss2=s+h*s1/2
            f2,s2=f(x2,ss2,t+h/2); x3=x+h*f2/2; ss3=s+h*s2/2
            f3,s3=f(x3,ss3,t+h/2); x4=x+h*f3; ss4=s+h*s3
            f4,s4=f(x4,ss4,t+h)
            c1,g1=c(x,s);c2,g2=c(x2,ss2);c3,g3=c(x3,ss3);c4,g4=c(x4,ss4)
            cost+=h/6*(c1+2*c2+2*c3+c4);grad+=h/6*(g1+2*g2+2*g3+g4)
            x+=h/6*(f1+2*f2+2*f3+f4);s+=h/6*(s1+2*s2+2*s3+s4)
            for y,sy in ((x2,ss2),(x3,ss3),(x4,ss4),(x,s)):
                cs.append(CONSTRAINT@y+OFFSET);js.append(CONSTRAINT@sy)
        e=(TOTAL@x-30)/10
        cost+=.5*float(e@e);grad+=e/10@TOTAL@s
        self.rec=dict(cost=cost,gradient=grad,constraints=np.concatenate(cs),jacobian=np.vstack(js))
        return self.rec
    def objective(self,v):r=self.evaluate(v);return r["cost"],r["gradient"]
    def constraints(self,v):return self.evaluate(v)["constraints"]
    def jacobian(self,v):return self.evaluate(v)["jacobian"]

def audit():
    x=np.array([15.,14.,13.,17.]);u=np.array([.45,.6]);d=np.full(4,1.6)
    f,a,b=rhs(x,u,d)
    assert max(abs(f-36*zhong.field(100*x,u,d)))<1e-10
    eps=1e-6
    aj=np.column_stack([(rhs(x+np.eye(4)[k]*eps,u,d)[0]-rhs(x-np.eye(4)[k]*eps,u,d)[0])/(2*eps) for k in range(4)])
    bj=np.column_stack([(rhs(x,u+np.eye(2)[k]*eps,d)[0]-rhs(x,u-np.eye(2)[k]*eps,d)[0])/(2*eps) for k in range(2)])
    p=Prediction(x,0,lambda t:d);v=np.tile(u,2);r=p.evaluate(v)
    grad=r["gradient"].copy();jac=r["jacobian"].copy()
    ng=np.array([(p.evaluate(v+np.eye(4)[k]*eps)["cost"]-p.evaluate(v-np.eye(4)[k]*eps)["cost"])/(2*eps) for k in range(4)])
    nj=np.column_stack([(p.evaluate(v+np.eye(4)[k]*eps)["constraints"]-p.evaluate(v-np.eye(4)[k]*eps)["constraints"])/(2*eps) for k in range(4)])
    errors=dict(state_jacobian=max(abs(a-aj).ravel()),input_jacobian=max(abs(b-bj).ravel()),
        cost_gradient=max(abs(grad-ng)),constraint_gradient=max(abs(jac-nj).ravel()))
    assert max(errors.values())<1e-4,errors
    return {k:float(v) for k,v in errors.items()}

def run(totals,split,bounds,profile,policy):
    shares=(zhong.CASE_1.reported_equilibrium/np.repeat(zhong.CASE_1.target_totals,2) if split=="equilibrium" else np.full(4,.5))
    x=shares*np.repeat(totals,2)/100; initial=x.copy(); lower,upper=bounds
    demand=lambda t:np.full(4,1.6)*(1 if profile=="nominal" else 1+.1*np.sin(2*np.pi*t/.5))
    times=[];states=[];commands=[];solver=[];cost=0.;tts=0.;arrivals=0.;departures=0.;minimum=np.inf;compute=0.
    warm=np.tile(zhong.CASE_1.steady_gate,2);selected=None
    h=1/720  # five-second physical step, feedback continuous at RK4 stages
    for k in range(1080):
        t=k*h
        if policy=="tracking_mpc" and k%60==0:
            p=Prediction(x,t,demand);begin=time.perf_counter();candidates=[]
            for start in (warm,np.tile(zhong.CASE_1.steady_gate,2)):
                r=minimize(p.objective,start,method="SLSQP",jac=True,bounds=[(lower,upper)]*4,
                    constraints=[dict(type="ineq",fun=p.constraints,jac=p.jacobian)],options=dict(ftol=1e-9,maxiter=100))
                margin=float(min(p.constraints(r.x)));ok=bool(r.success and margin>=-1e-7)
                solver.append(dict(time_h=t,success=bool(r.success),feasible=ok,iterations=int(r.nit),minimum_margin=margin))
                if ok:candidates.append((r.fun,r.x.copy()))
            compute+=time.perf_counter()-begin
            if not candidates:
                return dict(completed=False,failed_at_h=t,reason="tracking_MPC_unavailable",solver=solver)
            warm=min(candidates,key=lambda v:v[0])[1];selected=warm[:2]
        def command(y,clock):
            if policy=="tracking_mpc": return selected
            case=dataclasses.replace(zhong.CASE_1,demand=demand(clock))
            return zhong.almost_smooth_gate(100*y,case,lower,upper)
        def f(y,clock):
            u=command(y,clock);return rhs(y,u,demand(clock))[0],u
        begin=time.perf_counter();f1,u1=f(x,t);x2=x+h*f1/2
        f2,u2=f(x2,t+h/2);x3=x+h*f2/2
        f3,u3=f(x3,t+h/2);x4=x+h*f3
        f4,u4=f(x4,t+h);new=x+h/6*(f1+2*f2+2*f3+f4)
        if policy=="clf":compute+=time.perf_counter()-begin
        for y,u in ((x,u1),(x2,u2),(x3,u3),(x4,u4),(new,u4)):
            minimum=min(minimum,float(min(CONSTRAINT@y+OFFSET)),float(min(u-lower)),float(min(upper-u)))
        def costs(y,clock):
            e=100*(TOTAL@y-30);n=TOTAL@y;out=y[0]*(.0014877*n[0]**2-.29815*n[0]+15.0912)+y[3]*(.0014877*n[1]**2-.29815*n[1]+15.0912)
            return np.array([e@e,100*sum(n),3600*sum(demand(clock)),100*out])
        accumulated=h/6*(costs(x,t)+2*costs(x2,t+h/2)+2*costs(x3,t+h/2)+costs(x4,t+h))
        cost+=accumulated[0];tts+=accumulated[1];arrivals+=accumulated[2];departures+=accumulated[3]
        x=new
        if k%12==11:times.append(t+h);states.append((100*x).tolist());commands.append(u1.tolist())
    errors=np.max(abs(np.asarray(states)@TOTAL.T-3000),axis=1)
    settled=None
    for i in range(len(errors)):
        if max(errors[i:])<=25:settled=60*times[i];break
    return dict(completed=True,tracking_integral_veh2_h=float(cost),tts_veh_h=float(tts),completed_trips=float(departures),
        final_totals_vehicles=(100*TOTAL@x).tolist(),settling_to_25vehicles_minutes=settled,
        sampled_physical_pass=bool(minimum>=-1e-7),minimum_margin=float(minimum),compute_seconds=compute,
        conservation_error_vehicles=float(100*sum(x-initial)-arrivals+departures),solver=solver,
        trajectory=dict(time_h=times,state_vehicles=states,gates=commands))

def main():
    save(HERE/"results/regulation_audit.json",audit())
    cases=list(itertools.product(((1800,3100),(2400,3600),(3500,2800)),("equilibrium","symmetric"),((.1,.9),(.4,.6)),("nominal","sinusoidal")))
    for index,(totals,split,bounds,profile) in enumerate(cases):
        output=HERE/"results/regulation"/f"R{index:03d}.json"
        if output.exists():print(index,"cached",flush=True);continue
        result=dict(case=dict(id=f"R{index:03d}",totals=totals,split=split,bounds=bounds,profile=profile),policies={})
        for policy in ("clf","tracking_mpc"):
            result["policies"][policy]=run(totals,split,bounds,profile,policy)
        save(output,result)
        print(index,{p:dict(completed=r["completed"],error=r.get("tracking_integral_veh2_h"),safe=r.get("sampled_physical_pass")) for p,r in result["policies"].items()},flush=True)

if __name__=="__main__":main()
