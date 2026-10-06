"""Paired replay audit of the existing prospective certified-tool portfolio."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from common import HERE, save
from run_nominal_vs_full_certified import load

OUT = HERE / "results/portfolio_reanalysis_2026-10-01"
TOL = 1e-6


def paired(first, second):
    assert first["certificate"]["accepted"] and second["certificate"]["accepted"]
    ar = {(v["corner"],v["profile"]):v for v in first["realizations"]}
    br = {(v["corner"],v["profile"]):v for v in second["realizations"]}
    assert ar.keys()==br.keys() and len(ar)==640
    assert all(first["summary"][k]==second["summary"][k]==640
               for k in ("runs","physical_passes","service_passes"))
    result={}
    for name in ("priority_cost","tts_veh_h","queue_veh_h"):
        a=np.array([ar[k][name] for k in sorted(ar)])
        b=np.array([br[k][name] for k in sorted(br)])
        difference=a-b
        med=float(np.median(difference))
        result[name]=dict(status="portfolio_better" if med < -TOL else
                          "fixed_tool_better" if med > TOL else "tie",
                          median_relative_percent=float(np.median(100*difference/b)),
                          median_difference_veh_h=med)
    return result


def count(rows):
    output={"contracts":len(rows)}
    for metric in ("priority_cost","tts_veh_h","queue_veh_h"):
        statuses=Counter(r["metrics"][metric]["status"] for r in rows)
        values=[r["metrics"][metric]["median_relative_percent"] for r in rows]
        output[metric]=dict(portfolio_better=statuses["portfolio_better"],
                            fixed_tool_better=statuses["fixed_tool_better"],
                            ties=statuses["tie"],
                            median_relative_percent=float(np.median(values)),
                            range_relative_percent=[float(min(values)),float(max(values))])
    return output


def main():
    original=load("results/contracts")
    stronger=load("results/full_scenario")
    refresh=load("results/reserve_refresh")
    portfolio=json.loads((HERE/"results/certified_tool_selection.json").read_text())
    assert len(original)==len(stronger)==len(portfolio["rows"])==80
    source_map={"original":original,"full_scenario":stronger,"reserve_refresh":refresh}
    rows={"ordinary":[],"stronger":[]}
    for item in portfolio["rows"]:
        cid=item["contract"]["id"]
        selected=source_map[item["source"]][cid]["policies"][item["policy"]]
        assert selected["certificate"]["accepted"]
        assert selected["commands"]==item["commands"]
        for comparator in ("ordinary","stronger"):
            archive=original[cid] if comparator=="ordinary" else stronger[cid]
            key=archive.get("certified_repair_source") if comparator=="ordinary" else archive.get("certified_source")
            if key:
                fixed=archive["policies"][key]
            else:
                assert refresh[cid]["certified_source"]=="reserve_repair"
                fixed=refresh[cid]["policies"]["reserve_repair"]
                key="reserve_repair"
            rows[comparator].append(dict(id=cid,split=item["contract"]["split"],
                portfolio_source=item["source"],portfolio_policy=item["policy"],
                fixed_source=key,identical_commands=selected["commands"]==fixed["commands"],
                metrics=paired(selected,fixed)))
    summary=dict(selection_rule=portfolio["selection_rule"],
                 portfolio_sources=dict(Counter(v["source"] for v in portfolio["rows"])),
                 against_ordinary=count(rows["ordinary"]),
                 against_stronger=count(rows["stronger"]),
                 held_out_against_stronger=count([r for r in rows["stronger"] if r["split"]=="held_out"]),
                 selection_uses_replay_outcomes=False)
    save(OUT/"summary.json",summary)
    save(OUT/"paired_contracts.json",rows)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
