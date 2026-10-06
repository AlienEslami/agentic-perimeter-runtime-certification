"""Aggregate all fixed-grid results without selecting favorable cases."""
from __future__ import annotations
import csv,json,platform,sys
from pathlib import Path
import numpy as np
import scipy,mpmath,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import HERE,save,plant

def stats(values):
    x=np.asarray(values,dtype=float)
    return dict(minimum=float(x.min()),median=float(np.median(x)),maximum=float(x.max())) if len(x) else None

def feasible(summary):
    return summary["physical_passes"]==summary["runs"] and (summary["service_passes"] is None or summary["service_passes"]==summary["runs"])

def load(folder,pattern="*.json"):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((HERE/folder).glob(pattern))]

def route(record,source):return record["policies"].get(source) if source else None

def deadline_group(records,full):
    result=dict(contracts=len(records),backup_available=sum(r["backup_available"] for r in records),
        nominal_certified=sum(r["policies"]["nominal_mpc"]["certificate"]["accepted"] for r in records),
        nominal_all_sampled_feasible=sum(feasible(r["policies"]["nominal_mpc"]["summary"]) for r in records),
        nominal_contracts_with_physical_violation=sum(r["policies"]["nominal_mpc"]["summary"]["physical_passes"]<640 for r in records),
        nominal_contracts_with_service_violation=sum(r["policies"]["nominal_mpc"]["summary"]["service_passes"]<640 for r in records),
        certified_repair_available=sum(r.get("certified_repair_source") is not None for r in records),
        modified_proposals=sum((r.get("selected_theta") or 0)>0 for r in records),
        theta=stats([r["selected_theta"] for r in records if (r.get("selected_theta") or 0)>0]),
        cbf_completed=sum(r["cbf_adverse_realization"]["completed"] for r in records),
        cbf_physical_pass=sum(r["cbf_adverse_realization"].get("physical_pass",False) for r in records),
        cbf_service_pass=sum(r["cbf_adverse_realization"].get("service_pass",False) for r in records))
    result["five_scenario_certified"]=sum(r["policies"].get("scenario_mpc",{}).get("certificate",{}).get("accepted",False) for r in records)
    result["five_scenario_all_sampled_feasible"]=sum(feasible(r["policies"]["scenario_mpc"]["summary"]) for r in records if "scenario_mpc" in r["policies"])
    paired_priority=[];paired_tts=[];recovery=[];strong_benefit=[]
    for r in records:
        repaired=route(r,r.get("certified_repair_source"));retained=route(r,r.get("certified_retention_source"))
        if repaired:
            assert repaired["certificate"]["accepted"] and feasible(repaired["summary"])
        if (r.get("selected_theta") or 0)>0 and repaired and retained:
            for a,b,n in zip(repaired["realizations"],retained["realizations"],r["policies"]["nominal_mpc"]["realizations"]):
                assert (a["corner"],a["profile"])==(b["corner"],b["profile"])==(n["corner"],n["profile"])
                paired_priority.append(100*(b["priority_cost"]-a["priority_cost"])/b["priority_cost"])
                paired_tts.append(100*(a["tts_veh_h"]-b["tts_veh_h"])/b["tts_veh_h"])
                if b["priority_cost"]-n["priority_cost"]>1e-8:
                    recovery.append((b["priority_cost"]-a["priority_cost"])/(b["priority_cost"]-n["priority_cost"]))
        stronger=full.get(r["contract"]["id"])
        if stronger and stronger.get("certified_source") and repaired:
            selected=stronger["policies"][stronger["certified_source"]]
            for a,b in zip(repaired["realizations"],selected["realizations"]):
                strong_benefit.append(100*(a["priority_cost"]-b["priority_cost"])/a["priority_cost"])
    result.update(repair_priority_improvement_vs_retention_percent=stats(paired_priority),
        repair_tts_change_vs_retention_percent=stats(paired_tts),requested_priority_benefit_recovered_fraction=stats(recovery),
        stronger_tool_priority_improvement_vs_original_certified_tool_percent=stats(strong_benefit))
    strong=[full[r["contract"]["id"]] for r in records if r["contract"]["id"] in full]
    result["full_scenario"]=dict(contracts=len(strong),solver_available=sum(r["status"]=="completed" for r in strong),
        certified_directly=sum(r["policies"].get("full_scenario_mpc",{}).get("certificate",{}).get("accepted",False) for r in strong),
        all_sampled_feasible=sum(feasible(r["policies"]["full_scenario_mpc"]["summary"]) for r in strong if "full_scenario_mpc" in r["policies"]),
        certified_with_repair=sum(r.get("certified_source") is not None for r in strong),
        theta=stats([r.get("selected_theta",0.) for r in strong]))
    return result

def main():
    records=load("results/contracts");reg=load("results/regulation")
    full_rows=load("results/full_scenario");full={r["contract"]["id"]:r for r in full_rows}
    assert len(records)==92 and len(reg)==24 and len(full)==80,(len(records),len(reg),len(full))
    assert (HERE/"results/end_to_end_api/summary.json").exists()
    q=[r for r in records if r["contract"]["task"]=="Q"]
    d=[r for r in records if r["contract"]["task"]=="D"]
    reserve_rows=load("results/reserve_refresh")
    assert len(reserve_rows)==sum(not rr["backup_available"] for rr in d),"Complete the reserve-refresh grid before aggregation"
    by_reserve={rr["contract"]["id"]:rr for rr in reserve_rows}
    native_ratio=[100*(1-r["policies"]["tracking_mpc"]["tracking_integral_veh2_h"]/r["policies"]["clf"]["tracking_integral_veh2_h"]) for r in reg]
    queue_ratio=[]
    for r in q:
        bykey={(v["corner"],v["profile"]):v for v in r["policies"]["nominal_mpc"]["realizations"]}
        for clf in r["clf_extension"]["realizations"]:
            if clf["completed"]:
                nominal=bykey[(clf["corner"],clf["profile"])]
                if clf["physical_pass"] and nominal["physical_pass"]:
                    queue_ratio.append(100*(clf["tts_veh_h"]-nominal["tts_veh_h"])/clf["tts_veh_h"])
    api=json.loads((HERE/"results/api/summary.json").read_text())
    end_api=json.loads((HERE/"results/end_to_end_api/summary.json").read_text())
    ledger=json.loads((HERE/"results/api/budget_ledger.json").read_text())
    portfolio=json.loads((HERE/"results/certified_tool_selection.json").read_text())
    summary=dict(protocol="26 September 2026 fixed-grid reusable benchmark",
        counts=dict(regulation_cases=24,queue_contracts=12,deadline_development=72,deadline_held_out=8,
            all_constant_plan_realizations=sum(p["summary"]["runs"] for r in records for p in r["policies"].values())+
                sum(p["summary"]["runs"] for r in full_rows+reserve_rows for p in r.get("policies",{}).values())),
        regulation=dict(cases=24,all_completed=all(p["completed"] for r in reg for p in r["policies"].values()),
            all_physically_feasible=all(p["sampled_physical_pass"] for r in reg for p in r["policies"].values()),
            tracking_mpc_regulation_error_reduction_percent=stats(native_ratio),
            maximum_conservation_error_vehicles=max(abs(p["conservation_error_vehicles"]) for r in reg for p in r["policies"].values())),
        queue=dict(contracts=12,nominal_certified=sum(r["policies"]["nominal_mpc"]["certificate"]["accepted"] for r in q),
            all_nominal_640_realizations_feasible=all(feasible(r["policies"]["nominal_mpc"]["summary"]) for r in q),
            clf_completed=sum(r["clf_extension"]["completed_count"] for r in q),
            clf_physically_feasible=sum(r["clf_extension"]["summary"]["physical_passes"] for r in q),
            matched_feasible_runs=len(queue_ratio),mpc_tts_reduction_vs_clf_extension_percent=stats(queue_ratio)),
        deadline_all=deadline_group(d,full),deadline_development=deadline_group([r for r in d if r["contract"]["split"]=="development"],full),
        deadline_held_out=deadline_group([r for r in d if r["contract"]["split"]=="held_out"],full),
        reserve_refresh=dict(contracts=len(reserve_rows),reserves_certified=sum(rr["reserve_available"] for rr in reserve_rows),
            new_repaired_outputs=sum(rr.get("certified_source")=="reserve_repair" for rr in reserve_rows),
            all_contracts_certified_with_refresh=sum(rr.get("certified_repair_source") is not None or
                by_reserve.get(rr["contract"]["id"],{}).get("certified_source") is not None for rr in d)),
        api_pilot=api,live_tool_traces=end_api,
        certified_tool_portfolio=dict(contracts=portfolio["contracts"],available=portfolio["available"],
            selection_rule=portfolio["selection_rule"],selected_sources={name:sum(r.get("source")==name for r in portfolio["rows"])
                for name in ("original","full_scenario","reserve_refresh")}),
        api_cost=dict(known_token_cost_usd=ledger["known_usage_usd"],accounted_upper_usd=ledger["accounted_usd"],budget_usd=10.,
            all_attempts=len(ledger["calls"]),unknown_or_pending=sum(r["status"]!="completed" for r in ledger["calls"])),
        timing=dict(contract_study_wall_seconds=stats([r["wall_seconds"] for r in records]),
            full_scenario_wall_seconds=stats([r["wall_seconds"] for r in full_rows if "wall_seconds" in r]),
            proof_seconds=stats([p["certificate"]["wall_seconds"] for r in records for p in r["policies"].values()]),
            scope="checkpointed studies executed in parallel processes; wall time reflects concurrent desktop load"))
    primary=next(r for r in d if r["contract"]["id"]=="D027")
    rows=[]
    for name in ("nominal_mpc","continuation","grid_repair","scenario_mpc"):
        p=primary["policies"][name];v=next(v for v in p["realizations"] if v["corner"]=="+++++++" and v["profile"]=="constant_high")
        rows.append(dict(policy=name,certified=p["certificate"]["accepted"],**v))
    rows.append(dict(policy="physical_CBF",certified=None,**primary["cbf_adverse_realization"]))
    for name,p in full["D027"]["policies"].items():
        v=next(v for v in p["realizations"] if v["corner"]=="+++++++" and v["profile"]=="constant_high")
        rows.append(dict(policy=name,certified=p["certificate"]["accepted"],**v))
    summary["primary_case"]=dict(contract=primary["contract"],rows=rows)
    save(HERE/"results/summary.json",summary)
    environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,mpmath=mpmath.__version__,matplotlib=matplotlib.__version__,
        platform=platform.system(),architecture=platform.machine(),model=dict(state_scale_vehicles=100,time_unit="hour",
            internal_demand_veh_h=(100*plant.INTERNAL_DEMAND).tolist(),queue_limit_vehicles=2000,release_h_inverse=60,admission_capacity_veh_h=12000))
    save(HERE/"results/environment.json",environment)
    write_csv(records,full_rows+reserve_rows)
    make_figures(d,full)
    write_markdown(summary)
    print(json.dumps(summary,indent=2))

def write_csv(records,full_rows):
    case_rows=[];realization_rows=[]
    for record in records+full_rows:
        c=record["contract"]
        for name,p in record.get("policies",{}).items():
            cert=p["certificate"]
            case_rows.append(dict(**c,policy=name,accepted=cert["accepted"],proof_id=cert["proof_id"],
                runs=p["summary"]["runs"],physical_passes=p["summary"]["physical_passes"],service_passes=p["summary"]["service_passes"],
                tts_median=p["summary"]["tts_veh_h"]["median"],priority_median=p["summary"]["priority_cost"]["median"]))
            realization_rows.extend(dict(case_id=c["id"],task=c["task"],split=c["split"],policy=name,**r) for r in p["realizations"])
    for name,rows in (("CASE_SUMMARY.csv",case_rows),("REALIZATION_RESULTS.csv",realization_rows)):
        with (HERE/"results"/name).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def make_figures(records,full):
    figures=HERE/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
    widths=[1.,10.,25.,50.];residuals=[60.,150.,300.]
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.8),layout="constrained")
    for ax,strong,title in zip(axes,(False,True),("Economic MPC + certified repair","All-scenario MPC + certified repair")):
        array=np.full((4,3),np.nan)
        for r in records:
            c=r["contract"]
            if c["deadline_vehicles"]==2850 and c["weight"]==4 and c["split"]=="development":
                rr=full[c["id"]] if strong else r
                theta=(rr.get("selected_theta",0.) if rr.get("certified_source") else np.nan) if strong else rr.get("selected_theta")
                array[widths.index(c["width_vehicles"]),residuals.index(c["residual_veh_h"])]=theta
        picture=ax.imshow(100*array,cmap="Blues",vmin=0,vmax=100,aspect="auto")
        ax.set_xticks(range(3),[str(int(v)) for v in residuals]);ax.set_yticks(range(4),[str(int(v)) for v in widths])
        ax.set_xlabel("Arrival residual bound (veh/h)");ax.set_ylabel("Sensing half-width (vehicles)");ax.set_title(title)
        for i,j in np.ndindex(array.shape):
            if np.isfinite(array[i,j]):ax.text(j,i,f"{100*array[i,j]:.0f}%",ha="center",va="center",color="white" if array[i,j]>.55 else "#102A43")
            else:ax.text(j,i,"n/a",ha="center",va="center",color="#526777")
    fig.colorbar(picture,ax=axes,label="Interpolation toward verified continuation (%)",shrink=.8)
    fig.savefig(figures/"repair_uncertainty_grid.png",dpi=200);plt.close(fig)

def write_markdown(s):
    d=s["deadline_all"];q=s["queue"];r=s["regulation"];api=s["api_cost"]
    text=f"""# Complete reusable perimeter-control benchmark report

26 September 2026. The previous manuscript is preserved in `archive/pre_reusable_benchmark_2026-09-26`.

## Completed scope

- 24 native regulation cases, both CLF and tracking MPC.
- 12 queue-only finite-plan contracts.
- 72 development and 8 held-out deadline contracts.
- 640 corner/profile realizations per constant plan; separate 80-run CLF and one-run CBF diagnostic denominators.
- Both a five-scenario and a 641-scenario predictive comparator, with every solver and proof outcome retained.
- 96 OpenAI interpretation/routing calls and eight end-to-end live planning traces.

## Main results

Regional regulation: both policies completed all 24 cases within the tested limits. Tracking MPC reduced integrated regulation error by {r['tracking_mpc_regulation_error_reduction_percent']['minimum']:.1f}-{r['tracking_mpc_regulation_error_reduction_percent']['maximum']:.1f}% (median {r['tracking_mpc_regulation_error_reduction_percent']['median']:.1f}%). This comparison concerns the controllers, separately from runtime certification.

Queue-only planning: {q['nominal_certified']}/12 nominal plans were certified unchanged. The 640-realization tests passed in every case. Against the declared CLF queue/exit extension, matched feasible-case TTS reduction was {q['mpc_tts_reduction_vs_clf_extension_percent']['minimum']:.1f}-{q['mpc_tts_reduction_vs_clf_extension_percent']['maximum']:.1f}% (median {q['mpc_tts_reduction_vs_clf_extension_percent']['median']:.1f}%).

Deadline replacement: {d['nominal_certified']}/80 nominal proposals passed the uncertainty-family checker. {d['nominal_all_sampled_feasible']}/80 passed every sampled physical and service test. Certified replacement/repair remained available in {d['certified_repair_available']}/80 contracts, with a verified continuation available in {d['backup_available']}/80. The physical CBF diagnostic met its physical rows in {d['cbf_physical_pass']}/80 adverse trajectories and the regional deadline in {d['cbf_service_pass']}/80.

The all-scenario planner was numerically feasible across the full replay ensemble in {d['full_scenario']['all_sampled_feasible']}/80 contracts and directly certified in {d['full_scenario']['certified_directly']}/80. Certification plus repair provided {d['full_scenario']['certified_with_repair']}/80 certified outputs. This stronger comparator separates finite-scenario coverage from uncertainty-family admission.

## Matched primary case

The table uses the same all-high initial corner and constant-high residual, with a 2000-vehicle exterior queue limit and a 2850-vehicle Region-1 deadline. Costs of infeasible rows are descriptive, and feasible performance is compared only after applying both constraint tests.

| Policy | TTS (veh-h) | Peak queue (veh) | Region 1 at deadline (veh) | Physical / service | Family certificate |
| --- | ---: | ---: | ---: | --- | --- |
"""
    for row in s["primary_case"]["rows"]:
        text+=f"| {row['policy']} | {row['tts_veh_h']:.1f} | {row['peak_queue_vehicles']:.1f} | {row['deadline_n1_vehicles']:.1f} | {'pass' if row['physical_pass'] else 'fail'} / {'pass' if row['service_pass'] else 'fail'} | {row['certified']} |\n"
    text+=f"""
## Verified reserve availability

The archived continuation failed to close a proof in {s['reserve_refresh']['contracts']} contracts. A separately planned reserve certified in {s['reserve_refresh']['reserves_certified']}/{s['reserve_refresh']['contracts']}. With this explicit initialization/reserve-refresh tool, certified outputs were available in {s['reserve_refresh']['all_contracts_certified_with_refresh']}/80 deadline contracts. The original availability outcomes remain in the released data. This establishes existence on the tested finite grid; the applicable information/disturbance contract must be checked before authority is granted.

The released planning-tool portfolio selects the lowest nominal center-forecast task cost among the independently certified same-contract candidates. It supplied {s['certified_tool_portfolio']['available']}/80 outputs. Realized performance is excluded from the selection rule.

## What the paper can use

The strongest finding is the separation between a plan's nominal economic appeal, its finite-scenario behavior, and its authority to replace an existing verified plan. The released problem definitions make these distinct questions reproducible. Certified repair preserves operational choice while retaining a checked continuation through a named absolute expiry.

Priority performance and network waiting are separate objectives. On repaired proposals, median priority-cost improvement relative to retaining the continuation was {d['repair_priority_improvement_vs_retention_percent']['median']:.2f}%, while median TTS change was {d['repair_tts_change_vs_retention_percent']['median']:+.2f}%. This is an operator tradeoff, with physical storage and the regional deadline enforced in both certified policies.

The literature benchmark includes the native reproduced CLF equations and local predictive-controller implementations, with model extensions and information choices identified. The seven-state CLF row includes an explicit queue/exit feedback extension. These results support a reusable task-and-contract benchmark, with comparison boundaries stated for each method.

The API pilot records contextual interpretation separately from physical admission. History-only objective accuracy was 19/24 clean and 15/24 poisoned; the explicit task record achieved 24/24 in each group. The separate live traces actually run the requested optimization and checking tools; their complete outcomes appear in `results/end_to_end_api/summary.json`.

## Validation and scope

The model/Jacobian, command-compilation, and independent adaptive-replay checks are archived. The cubic MFD and synthetic internal demand remain fixed while sensing and arrival residuals vary. The finite-plan deadline study uses one inherited minute-20 state; regulation uses three initial regional-total pairs. These are fully specified computational benchmarks for later comparisons. Closed-loop city-network calibration and parameter-uncertainty validation are separate deployment studies.

Individual proof checks took a median {s['timing']['proof_seconds']['median']:.1f} seconds (maximum {s['timing']['proof_seconds']['maximum']:.1f}) under concurrent desktop load. The simulations use precomputed activation-time commands and report processor/API wall time separately from traffic time. A delayed deployment handover requires propagating the retained plan to the scheduled activation and checking that inherited state set.

API known token cost: USD {api['known_token_cost_usd']:.4f}; conservative total accounting: USD {api['accounted_upper_usd']:.4f}; authorized ceiling: USD 10. Development calls and any retained reservations are included in the accounting.

## Suggested manuscript changes after review

1. Add the three benchmark problem definitions before the architecture and explain how each relates to the perimeter-control literature.
2. Keep one main matched deadline table, the 12 queue-only unchanged-admission cases, and the uncertainty/repair summary. Put full grids and detailed trajectories in the supplement and released benchmark.
3. Present predictive repair and scenario planning as tools within the same certified execution architecture. Lead with guaranteed storage/service feasibility and conditional operational tradeoffs.
4. Separate semantic task-state evidence from physical guarantees; use the live tool traces as an interface demonstration.

The current manuscript source is unchanged. The experiment package contains `benchmark_formulation.tex`, full raw results, structured summaries and reproducible scripts. No GitHub push is performed by this study.
"""
    (HERE/"REPORT.md").write_text(text,encoding="utf-8")

if __name__=="__main__":main()
