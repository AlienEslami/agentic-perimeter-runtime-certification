"""Complete fixed-grid economic/deadline replacement sweeps, checkpointed."""
from __future__ import annotations
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
from common import *

def contract_grid():
    rows=[]
    for a,d in itertools.product((.2,.3,.4,.5),(.5,.75,1.)):
        rows.append(dict(task="Q",split="development",amplitude=a,duration_h=d,
            width_vehicles=1.,residual_veh_h=300.,deadline_vehicles=None,weight=1.))
    for w,r,b,p in itertools.product((1.,10.,25.,50.),(60.,150.,300.),(2750.,2850.,2950.),(2.,4.)):
        rows.append(dict(task="D",split="development",amplitude=.3,duration_h=1.,
            width_vehicles=w,residual_veh_h=r,deadline_vehicles=b,weight=p))
    for w,r,b in itertools.product((5.,20.),(100.,225.),(2800.,2900.)):
        rows.append(dict(task="D",split="held_out",amplitude=.3,duration_h=1.,
            width_vehicles=w,residual_veh_h=r,deadline_vehicles=b,weight=3.))
    for index,row in enumerate(rows): row["id"]=f"{row['task']}{index:03d}"
    return rows

def scenario_plan(contract,nominal,backup):
    x=np.asarray(fixture()["center"]); width=contract["width_vehicles"]/100
    row=np.array([1.,contract["weight"],0,0,0,0,1.])
    predictors=[ServicePrediction(x,float(START),float(END-START),lambda t:arrival(t,contract),
        service_target_vehicles=contract["deadline_vehicles"],cost_row=row)]
    for sign,res in itertools.product((-1,1),(-1,1)):
        predictors.append(ServicePrediction(x+sign*width,float(START),float(END-START),
            lambda t,res=res:arrival(t,contract)+res*contract["residual_veh_h"]/100,
            service_target_vehicles=contract["deadline_vehicles"],cost_row=row))
    def constraints(v): return np.concatenate([p.constraints(v) for p in predictors])
    def jac(v): return np.vstack([p.constraint_jacobian(v) for p in predictors])
    attempts=[]; candidates=[]; begin=time.perf_counter()
    for guess in (nominal.ravel(),backup.ravel(),np.tile(plant.BASE_COMMAND,2)):
        result=minimize(predictors[0].objective,guess,method="SLSQP",jac=True,bounds=[(0.,1.)]*8,
            constraints=[dict(type="ineq",fun=constraints,jac=jac)],options=dict(maxiter=100,ftol=1e-8))
        margin=float(min(constraints(result.x)))
        ok=bool(result.success and margin>=-1e-7 and min(result.x)>=-1e-7 and max(result.x)<=1+1e-7)
        attempts.append(dict(success=bool(result.success),sampled_feasible=ok,status=int(result.status),
            iterations=int(result.nit),message=str(result.message),cost_veh_h=float(100*result.fun),minimum_margin=margin))
        if ok: candidates.append((float(result.fun),result.x.reshape(2,4).copy()))
    selected=None if not candidates else min(candidates,key=lambda r:r[0])[1]
    return selected,dict(attempts=attempts,solve_seconds=time.perf_counter()-begin,scenario_count=5,
        feasible_solution_found=selected is not None)

def evaluate_plan(commands,contract,certificate=None):
    rows=ensemble(commands,contract)
    return dict(commands=np.asarray(commands).tolist(),certificate=certificate,summary=summarize(rows),realizations=rows)

def run_case(contract):
    output=HERE/"results"/"contracts"/(contract["id"]+".json")
    if output.exists(): return contract["id"],"cached"
    begin=time.perf_counter(); center=np.asarray(fixture()["center"])
    backup,backup_change=compile_commands(fixture()["backup_commands"])
    row=np.array([1.,contract["weight"],0,0,0,0,1.])
    selected,solver=solve_service(center,float(START),arrival_fun=lambda t:arrival(t,contract),
        service_target_vehicles=contract["deadline_vehicles"],horizon_h=float(END-START),cost_row=row)
    payload=dict(contract=contract,nominal_solver=solver,policies={},repair_attempts=[],
        formal_scope="fixed minute-20 replacement through the absolute one-hour deadline")
    backup_certificate=certify(backup,contract)
    payload["backup_available"]=bool(backup_certificate["accepted"])
    payload["backup_compilation_max_change"]=backup_change
    payload["policies"]["continuation"]=evaluate_plan(backup,contract,backup_certificate)
    if selected is None:
        payload["status"]="nominal_planner_unavailable"
        payload["wall_seconds"]=time.perf_counter()-begin; save(output,payload)
        return contract["id"],payload["status"]
    nominal,delta=compile_commands(selected[1])
    cert=certify(nominal,contract)
    payload["compilation_max_change"]=delta
    payload["policies"]["nominal_mpc"]=evaluate_plan(nominal,contract,cert)
    if cert["accepted"]:
        payload["certified_retention_source"]="nominal_mpc"
        payload["certified_repair_source"]="nominal_mpc"
        payload["selected_theta"]=0.
    elif backup_certificate["accepted"]:
        payload["certified_retention_source"]="continuation"
        payload["certified_repair_source"]="continuation"
        payload["selected_theta"]=1.
        # Ascending finite grid; every preceding point is actually checked.
        for theta in np.arange(1,20)/20:
            commands,change=compile_commands((1-theta)*nominal+theta*backup)
            certificate=certify(commands,contract)
            payload["repair_attempts"].append(dict(theta=float(theta),compilation_max_change=change,certificate=certificate))
            if certificate["accepted"]:
                payload["policies"]["grid_repair"]=evaluate_plan(commands,contract,certificate)
                payload["certified_repair_source"]="grid_repair"; payload["selected_theta"]=float(theta)
                break
    else:
        payload["certified_retention_source"]=None
        payload["certified_repair_source"]=None
        payload["selected_theta"]=None
    if contract["task"]=="D":
        scenario,report=scenario_plan(contract,nominal,backup)
        payload["scenario_solver"]=report
        if scenario is not None:
            scenario,change=compile_commands(scenario)
            certificate=certify(scenario,contract)
            payload["policies"]["scenario_mpc"]=evaluate_plan(scenario,contract,certificate)
            payload["scenario_compilation_max_change"]=change
            payload["certified_scenario_source"]=("scenario_mpc" if certificate["accepted"] else
                                                  "continuation" if backup_certificate["accepted"] else None)
        # CBF is a physical-filter diagnostic, one common adverse corner per contract.
        payload["cbf_adverse_realization"]=feedback_run(nominal,contract,"cbf")
    else:
        signs=np.asarray(list(itertools.product((-1.,1.),repeat=7)))[::8]
        rows=[feedback_run(nominal,contract,"clf",profile=p,signs=s) for s in signs for p in PROFILES]
        completed=[r for r in rows if r["completed"]]
        payload["clf_extension"]=dict(realizations=rows,summary=summarize(completed) if completed else None,
            completed_count=len(completed),published_controller_unchanged=False,
            extension="queue and exit LQR channels plus shared gate-direction projection")
    payload["status"]="completed"; payload["wall_seconds"]=time.perf_counter()-begin
    save(output,payload)
    return contract["id"],dict(backup=payload["backup_available"],nominal_admitted=cert["accepted"],
                               theta=payload["selected_theta"],seconds=round(payload["wall_seconds"],1))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--only"); args=parser.parse_args()
    grid=contract_grid(); save(HERE/"contracts.json",grid)
    if args.only: grid=[c for c in grid if c["id"] in args.only.split(",") or c["task"]==args.only]
    if args.workers==1:
        for c in grid: print(run_case(c),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures={executor.submit(run_case,c):c["id"] for c in grid}
            for future in as_completed(futures):
                try: print(future.result(),flush=True)
                except Exception as exc:
                    print("ERROR",futures[future],type(exc).__name__,str(exc),flush=True)
                    raise

if __name__=="__main__": main()
