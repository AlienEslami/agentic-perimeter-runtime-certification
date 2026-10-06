"""Precompute a verified reserve when the old continuation proof does not close.

This is a separate extension, preserving the full original benchmark outcomes.
It grants no authority until the entire same-contract reserve plan is certified.
"""
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from common import *
from run_contracts import contract_grid,evaluate_plan

def run_case(c):
    output=HERE/"results/reserve_refresh"/(c["id"]+".json")
    if output.exists():return c["id"],"cached"
    original=json.loads((HERE/"results/contracts"/(c["id"]+".json")).read_text())
    if original["backup_available"]:return c["id"],"original_available"
    center=np.asarray(fixture()["center"]);begin=time.perf_counter();attempts=[];reserve=None;reserve_cert=None
    for name,row in (("network",np.array([1.,1,0,0,0,0,1.])),("moderate_region2",np.array([1.,2,0,0,0,0,1.]))):
        selected,report=solve_service(center,float(START),arrival_fun=lambda t:arrival(t,c),
            service_target_vehicles=c["deadline_vehicles"],horizon_h=float(END-START),cost_row=row)
        attempt=dict(objective=name,solver=report)
        if selected is not None:
            commands,change=compile_commands(selected[1]);cert=certify(commands,c)
            attempt.update(certificate=cert,commands=commands.tolist(),compilation_max_change=change)
            if cert["accepted"]:reserve,reserve_cert=commands,cert;attempts.append(attempt);break
        attempts.append(attempt)
    result=dict(contract=c,original_continuation_certified=False,reserve_attempts=attempts,policies={},repair_attempts=[],
        reserve_available=reserve is not None)
    if reserve is not None:
        result["policies"]["fresh_reserve"]=evaluate_plan(reserve,c,reserve_cert)
        candidate=np.asarray(original["policies"]["nominal_mpc"]["commands"])
        candidate_cert=original["policies"]["nominal_mpc"]["certificate"]
        if candidate_cert["accepted"]:
            result["certified_source"]="original_nominal_mpc";result["selected_theta"]=0.
        else:
            for theta in np.arange(1,21)/20:
                commands,_=compile_commands((1-theta)*candidate+theta*reserve);cert=certify(commands,c)
                result["repair_attempts"].append(dict(theta=float(theta),certificate=cert))
                if cert["accepted"]:
                    result["policies"]["reserve_repair"]=evaluate_plan(commands,c,cert)
                    result["certified_source"]="reserve_repair";result["selected_theta"]=float(theta);break
    result["wall_seconds"]=time.perf_counter()-begin;save(output,result)
    return c["id"],dict(reserve=result["reserve_available"],theta=result.get("selected_theta"),seconds=round(result["wall_seconds"],1))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2);args=parser.parse_args()
    rows=[]
    for c in contract_grid():
        path=HERE/"results/contracts"/(c["id"]+".json")
        if c["task"]=="D" and path.exists() and not json.loads(path.read_text())["backup_available"]:rows.append(c)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for future in as_completed([ex.submit(run_case,c) for c in rows]):print(future.result(),flush=True)

if __name__=="__main__":main()
