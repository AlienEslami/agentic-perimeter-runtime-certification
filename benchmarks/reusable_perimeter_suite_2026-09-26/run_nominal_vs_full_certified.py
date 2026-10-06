"""Matched traffic-performance audit of already certified ordinary/full MPC routes."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from common import HERE, save

OUT = HERE / "results/nominal_vs_full_certified_2026-10-01"
TOL = 1e-6


def load(folder):
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in (HERE / folder).glob("D*.json")}


def match(cid, original, full, original_source):
    a = original["policies"][original_source]
    b = full["policies"][full["certified_source"]]
    assert a["certificate"]["accepted"] and b["certificate"]["accepted"]
    assert a["summary"]["runs"] == b["summary"]["runs"] == 640
    assert a["summary"]["physical_passes"] == b["summary"]["physical_passes"] == 640
    assert a["summary"]["service_passes"] == b["summary"]["service_passes"] == 640
    ar = {(v["corner"], v["profile"]): v for v in a["realizations"]}
    br = {(v["corner"], v["profile"]): v for v in b["realizations"]}
    assert len(ar) == len(br) == 640 and ar.keys() == br.keys()
    outcomes = {}
    for name in ("priority_cost", "tts_veh_h", "queue_veh_h"):
        av = np.array([ar[k][name] for k in sorted(ar)])
        bv = np.array([br[k][name] for k in sorted(br)])
        differences = av - bv
        relative = 100*differences/bv
        med = float(np.median(differences))
        status = "ordinary_better" if med < -TOL else "stronger_better" if med > TOL else "tie"
        outcomes[name] = dict(status=status,
                              median_difference_ordinary_minus_stronger=float(med),
                              median_relative_difference_percent=float(np.median(relative)),
                              minimum_relative_difference_percent=float(np.min(relative)),
                              maximum_relative_difference_percent=float(np.max(relative)),
                              ordinary_median=float(np.median(av)),
                              stronger_median=float(np.median(bv)),
                              paired_realization_ordinary_better=int(np.sum(differences < -TOL)),
                              paired_realization_stronger_better=int(np.sum(differences > TOL)))
    c = original["contract"]
    return dict(id=cid, split=c["split"], sensing_half_width_vehicles=c["width_vehicles"],
                arrival_residual_veh_h=c["residual_veh_h"], deadline_vehicles=c["deadline_vehicles"],
                weight=c["weight"], ordinary_source=original_source,
                stronger_source=full["certified_source"],
                exact_same_commands=a["commands"] == b["commands"],
                metrics=outcomes)


def summarize(rows):
    result = dict(contracts=len(rows))
    for name in ("priority_cost", "tts_veh_h", "queue_veh_h"):
        counts = Counter(r["metrics"][name]["status"] for r in rows)
        values = [r["metrics"][name]["median_relative_difference_percent"] for r in rows]
        result[name] = dict(ordinary_better=counts["ordinary_better"],
                            stronger_better=counts["stronger_better"],ties=counts["tie"],
                            ordinary_minus_stronger_median_percent=float(np.median(values)) if values else None,
                            ordinary_minus_stronger_range_percent=[float(min(values)),float(max(values))] if values else None)
    result["same_compiled_commands"] = sum(r["exact_same_commands"] for r in rows)
    result["by_sensing_half_width"] = {
        str(width): dict(contracts=sum(r["sensing_half_width_vehicles"] == width for r in rows),
                         ordinary_priority_wins=sum(r["sensing_half_width_vehicles"] == width and
                                                   r["metrics"]["priority_cost"]["status"] == "ordinary_better"
                                                   for r in rows))
        for width in sorted(set(r["sensing_half_width_vehicles"] for r in rows))}
    result["by_arrival_residual"] = {
        str(res): dict(contracts=sum(r["arrival_residual_veh_h"] == res for r in rows),
                       ordinary_priority_wins=sum(r["arrival_residual_veh_h"] == res and
                                                 r["metrics"]["priority_cost"]["status"] == "ordinary_better"
                                                 for r in rows))
        for res in sorted(set(r["arrival_residual_veh_h"] for r in rows))}
    return result


def main():
    originals = load("results/contracts")
    full = load("results/full_scenario")
    reserve = load("results/reserve_refresh")
    assert len(originals) == len(full) == 80
    repair_rows = []
    retention_rows = []
    missing = []
    for cid in sorted(originals):
        o, f = originals[cid], full[cid]
        if o.get("certified_repair_source") and f.get("certified_source"):
            repair_rows.append(match(cid, o, f, o["certified_repair_source"]))
        else:
            missing.append(cid)
        if o.get("certified_retention_source") and f.get("certified_source"):
            retention_rows.append(match(cid, o, f, o["certified_retention_source"]))
    assert len(repair_rows) == len(retention_rows) == 74
    assert missing == ["D073","D075","D077","D079","D081","D083"]
    assert all(reserve[cid]["certified_source"] == "reserve_repair" and
               reserve[cid]["policies"]["reserve_repair"]["certificate"]["accepted"]
               for cid in missing)
    summary = dict(comparison="matched accepted actual plans under unchanged contracts",
                   original_nominal_directly_certified=sum(o["policies"]["nominal_mpc"]["certificate"]["accepted"] for o in originals.values()),
                   full_scenario_directly_certified=sum(f["policies"]["full_scenario_mpc"]["certificate"]["accepted"] for f in full.values()),
                   ordinary_with_certified_repair=len(repair_rows),
                   stronger_with_certified_repair=sum(f.get("certified_source") is not None for f in full.values()),
                   common_certified_reserve_contracts=missing,
                   primary_certified_repair_comparison=summarize(repair_rows),
                   separate_certified_retention_comparison=summarize(retention_rows),
                   held_out_repair_comparison=summarize([r for r in repair_rows if r["split"] == "held_out"]),
                   all_80_available_after_same_reserve_refresh=True,
                   unit_of_comparison="contract, each with 640 matched realization replays",
                   lower_is_better=True)
    save(OUT/"summary.json",summary)
    save(OUT/"certified_repair_pairs.json",repair_rows)
    save(OUT/"certified_retention_pairs.json",retention_rows)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
