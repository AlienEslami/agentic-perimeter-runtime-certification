"""Stronger comparator: all 128 corners and five residual profiles in planning.

This supplements the predeclared five-scenario comparator, preserving its
results. All new solver outcomes and independently checked plans are retained.
"""
from __future__ import annotations
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
from common import *
from run_contracts import contract_grid,evaluate_plan

class FullPrediction:
    def __init__(self,contract):
        self.c=contract;center=np.asarray(fixture()["center"])
        corners=np.asarray(list(itertools.product((-1.,1.),repeat=7)))
        self.x=np.vstack((center,np.tile(center,(640,1))+np.repeat(corners,5,axis=0)*contract["width_vehicles"]/100))
        self.profile=np.r_[0,np.tile(np.arange(5),128)]
        self.row=np.array([1.,contract["weight"],0,0,0,0,1.]);self.key=None;self.calls=0
    def evaluate(self,v):
        if self.key is not None and np.array_equal(self.key,v):return self.record
        self.key=v.copy();self.calls+=1;x=self.x.copy();s=np.zeros((641,7,8));cost=0.;gradient=np.zeros(8)
        cs=[.75+mpc.CONTROL_A@v];js=[mpc.CONTROL_A]
        h=1/60
        def support(y,sy):
            margins=y@mpc.STATE_A.T+mpc.STATE_B
            indices=np.argmin(margins,axis=0)
            grads=np.einsum('ij,njk->nik',mpc.STATE_A,sy)
            # One vehicle of fixed planning clearance for finite-grid discretization.
            cs.append(margins[indices,np.arange(12)]-.01)
            js.append(grads[indices,np.arange(12)])
        for k in range(40):
            t=float(START)+k*h;block=0 if k<5 else 1;u=v.reshape(2,4)[block]
            embed=np.zeros((4,8));embed[:,4*block:4*block+4]=np.eye(4)
            def rhs(y,sy,clock):
                forcing=np.asarray([residual(t+h/2 if p=="alternating_5min" else min(clock,t+h-1e-9),
                    self.c["residual_veh_h"]/100,p) for p in PROFILES])[self.profile]
                offered=arrival(clock,self.c)+forcing
                f,_=batch_field(y,u,offered)
                n1,n2,n11,n22,n10,n20,z=y.T
                h1=.0014877*n1*n1-.29815*n1+15.0912;h2=.0014877*n2*n2-.29815*n2+15.0912
                hp1=.0029754*n1-.29815;hp2=.0029754*n2-.29815
                q11,q12,q10=n11*h1,(n1-n11-n10)*h1,n10*h1
                q22,q21,q20=n22*h2,(n2-n22-n20)*h2,n20*h2
                a11=sy[:,2]*h1[:,None]+(n11*hp1)[:,None]*sy[:,0]
                a12=(sy[:,0]-sy[:,2]-sy[:,4])*h1[:,None]+((n1-n11-n10)*hp1)[:,None]*sy[:,0]
                a10=sy[:,4]*h1[:,None]+(n10*hp1)[:,None]*sy[:,0]
                a22=sy[:,3]*h2[:,None]+(n22*hp2)[:,None]*sy[:,1]
                a21=(sy[:,1]-sy[:,3]-sy[:,5])*h2[:,None]+((n2-n22-n20)*hp2)[:,None]*sy[:,1]
                a20=sy[:,5]*h2[:,None]+(n20*hp2)[:,None]*sy[:,1]
                adm=np.where((60*z<120*u[2])[:,None],60*sy[:,6],120*embed[2][None,:])
                u12,u21,u02,u20=u
                ds=np.stack((-a11-(a12+a10)*u12+a21*u21-(q12+q10)[:,None]*embed[0]+q21[:,None]*embed[1],
                    -a22+(a12+a10)*u12-a21*u21+adm-a20*u20+(q12+q10)[:,None]*embed[0]-q21[:,None]*embed[1]-q20[:,None]*embed[3],
                    -a11+a21*u21+q21[:,None]*embed[1],
                    -a22+a12*u12+.5*adm+q12[:,None]*embed[0],
                    -a10*u12-q10[:,None]*embed[0],
                    -a20*u20+a10*u12-q20[:,None]*embed[3]+q10[:,None]*embed[0],-adm),axis=1)
                return f,ds
            f1,s1=rhs(x,s,t);x2=x+h*f1/2;ss2=s+h*s1/2
            f2,s2=rhs(x2,ss2,t+h/2);x3=x+h*f2/2;ss3=s+h*s2/2
            f3,s3=rhs(x3,ss3,t+h/2);x4=x+h*f3;ss4=s+h*s3
            f4,s4=rhs(x4,ss4,t+h)
            cost+=h/6*float(self.row@(x[0]+2*x2[0]+2*x3[0]+x4[0]))
            gradient+=h/6*(self.row@(s[0]+2*ss2[0]+2*ss3[0]+ss4[0]))
            new=x+h/6*(f1+2*f2+2*f3+f4);ss=s+h/6*(s1+2*s2+2*s3+s4)
            for y,sy in ((x2,ss2),(x3,ss3),(x4,ss4),(new,ss)):support(y,sy)
            x,s=new,ss
        i=np.argmax(x[:,0]);cs.append(np.array([self.c["deadline_vehicles"]/100-x[i,0]-.01]));js.append(-s[[i],0,:])
        self.record=dict(cost=cost,gradient=gradient,constraints=np.concatenate(cs),jacobian=np.vstack(js))
        return self.record
    def objective(self,v):r=self.evaluate(v);return r["cost"],r["gradient"]
    def constraints(self,v):return self.evaluate(v)["constraints"]
    def jacobian(self,v):return self.evaluate(v)["jacobian"]

def audit():
    c=next(c for c in contract_grid() if c["id"]=="D027")
    # Distinct interior gates avoid tied worst-corner supports. At a tie the
    # selected active-corner derivative is a valid nonsmooth support derivative,
    # and a centered difference can average two different active derivatives.
    predictor=FullPrediction(c);v=np.array([.58,.64,.62,.76,.49,.55,.68,.73])
    r=predictor.evaluate(v);gradient=r["gradient"].copy();jac=r["jacobian"].copy();eps=1e-6
    ng=np.array([(predictor.evaluate(v+np.eye(8)[k]*eps)["cost"]-predictor.evaluate(v-np.eye(8)[k]*eps)["cost"])/(2*eps) for k in range(8)])
    nj=np.column_stack([(predictor.evaluate(v+np.eye(8)[k]*eps)["constraints"]-predictor.evaluate(v-np.eye(8)[k]*eps)["constraints"])/(2*eps) for k in range(8)])
    errors=dict(cost_gradient=float(max(abs(gradient-ng))),constraint_gradient=float(max(abs(jac-nj).ravel())))
    assert max(errors.values())<1e-4,errors
    return errors

def run_case(c):
    path=HERE/"results/full_scenario"/(c["id"]+".json")
    if path.exists():return c["id"],"cached"
    source=HERE/"results/contracts"/(c["id"]+".json")
    if not source.exists():return c["id"],"source_pending"
    original=json.loads(source.read_text());policies=original["policies"]
    if "nominal_mpc" not in policies:
        save(path,dict(contract=c,status="nominal_start_unavailable"));return c["id"],"no_start"
    backup=np.asarray(policies["continuation"]["commands"]);nominal=np.asarray(policies["nominal_mpc"]["commands"])
    predictor=FullPrediction(c);begin=time.perf_counter();attempts=[];candidates=[]
    for guess in (nominal.ravel(),backup.ravel(),np.tile(plant.BASE_COMMAND,2)):
        r=minimize(predictor.objective,guess,method="SLSQP",jac=True,bounds=[(0.,1.)]*8,
            constraints=[dict(type="ineq",fun=predictor.constraints,jac=predictor.jacobian)],options=dict(maxiter=100,ftol=1e-8))
        margin=float(min(predictor.constraints(r.x)));feasible=bool(r.success and margin>=-1e-7)
        attempts.append(dict(success=bool(r.success),feasible=feasible,iterations=int(r.nit),status=int(r.status),
            message=str(r.message),minimum_margin=margin,cost_veh_h=100*float(r.fun)))
        if feasible:candidates.append((float(r.fun),r.x.reshape(2,4).copy()))
    result=dict(contract=c,attempts=attempts,planning_scenarios=641,planning_margin_vehicles=1.,
        solve_seconds=time.perf_counter()-begin,rollout_evaluations=predictor.calls,policies={})
    if candidates:
        plan,delta=compile_commands(min(candidates,key=lambda r:r[0])[1]);certificate=certify(plan,c)
        result["compilation_max_change"]=delta
        result["policies"]["full_scenario_mpc"]=evaluate_plan(plan,c,certificate)
        result["certified_source"]="full_scenario_mpc" if certificate["accepted"] else None
        result["repair_attempts"]=[]
        if not certificate["accepted"] and original["backup_available"]:
            for theta in np.arange(1,21)/20:
                repaired,_=compile_commands((1-theta)*plan+theta*backup)
                cert=certify(repaired,c)
                result["repair_attempts"].append(dict(theta=float(theta),certificate=cert))
                if cert["accepted"]:
                    result["policies"]["full_scenario_repair"]=evaluate_plan(repaired,c,cert)
                    result["selected_theta"]=float(theta);result["certified_source"]="full_scenario_repair";break
        result["status"]="completed"
    else:result["status"]="solver_unavailable"
    result["wall_seconds"]=time.perf_counter()-begin;save(path,result)
    return c["id"],dict(status=result["status"],accepted=result["policies"].get("full_scenario_mpc",{}).get("certificate",{}).get("accepted"),
                        theta=result.get("selected_theta"),seconds=round(result["wall_seconds"],1))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2);parser.add_argument("--audit",action="store_true");parser.add_argument("--only");args=parser.parse_args()
    if args.audit:save(HERE/"results/full_scenario_gradient_audit.json",audit());print("All-corner predictor gradients passed.");return
    grid=[c for c in contract_grid() if c["task"]=="D" and (HERE/"results/contracts"/(c["id"]+".json")).exists()]
    if args.only:grid=[c for c in grid if c["id"] in args.only.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(run_case,c):c["id"] for c in grid}
        for future in as_completed(futures):print(future.result(),flush=True)

if __name__=="__main__":main()
