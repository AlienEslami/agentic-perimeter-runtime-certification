"""Select the lowest nominal task cost among the available certified tools.

Selection uses the center forecast, not the replay outcomes. It preserves the
same initial-information, model, service and expiry contract for all candidates.
"""
import json
import numpy as np
from common import *

def main():
    rows=[]
    for path in sorted((HERE/"results/contracts").glob("D*.json")):
        original=json.loads(path.read_text());c=original["contract"];candidates=[]
        strong_path=HERE/"results/full_scenario"/path.name
        reserve_path=HERE/"results/reserve_refresh"/path.name
        records=[("original",original)]
        if strong_path.exists():records.append(("full_scenario",json.loads(strong_path.read_text())))
        if reserve_path.exists():records.append(("reserve_refresh",json.loads(reserve_path.read_text())))
        predictor=ServicePrediction(np.asarray(fixture()["center"]),float(START),float(END-START),lambda t:arrival(t,c),
            service_target_vehicles=c["deadline_vehicles"],cost_row=np.array([1.,c["weight"],0,0,0,0,1.]))
        for label,record in records:
            for name,p in record.get("policies",{}).items():
                if p["certificate"]["accepted"]:
                    cost=100*float(predictor.evaluate(np.asarray(p["commands"]).ravel())["cost"])
                    candidates.append((cost,label,name,p))
        if not candidates:rows.append(dict(contract=c,available=False));continue
        cost,label,name,selected=min(candidates,key=lambda r:r[0])
        assert selected["summary"]["physical_passes"]==640 and selected["summary"]["service_passes"]==640
        rows.append(dict(contract=c,available=True,source=label,policy=name,nominal_priority_cost=cost,
            proof_id=selected["certificate"]["proof_id"],commands=selected["commands"],summary=selected["summary"],
            candidates=[dict(nominal_priority_cost=a,source=b,policy=k) for a,b,k,_ in candidates]))
    save(HERE/"results/certified_tool_selection.json",dict(selection_rule="minimum nominal center-forecast task cost among independently certified same-contract tool plans",
        contracts=len(rows),available=sum(r["available"] for r in rows),rows=rows,
        prospective=True,replay_outcomes_not_used_in_selection=True))
    print("Certified tool portfolio:",sum(r["available"] for r in rows),"/",len(rows))

if __name__=="__main__":main()
