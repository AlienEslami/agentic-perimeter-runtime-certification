"""Matched sampled counterfactuals for the sequential authority pilot."""
from __future__ import annotations

import argparse
import json
import numpy as np

from common import HERE, save, summarize
from run_sequential_authority import OUT, PAIRS, read, replay


def compact(rows):
    aggregate = summarize(rows)
    aggregate["minimum_physical_margin"] = min(r["minimum_state_margin"] for r in rows)
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", help="Additional predeclared high-weight Task D ID")
    args = parser.parse_args()
    pairs = ((args.pair, f"D{int(args.pair[1:])-1:03d}"),) if args.pair else PAIRS
    results = []
    for high_id, low_id in pairs:
        previous = json.loads((OUT/f"{high_id}_{low_id}.json").read_text())
        c = previous["contract"]
        high = read(high_id)
        low = read(low_id, "full_scenario")
        original = np.asarray(high["policies"]["grid_repair"]["commands"])
        robust = np.asarray(low["policies"]["full_scenario_mpc"]["commands"])
        nominal = np.asarray(high["policies"]["nominal_mpc"]["commands"])
        static = np.asarray([original[0], original[1], original[1]])
        unchecked = np.asarray([original[0], robust[1], nominal[1]])
        certified = np.asarray(previous["executed_commands"])
        assert np.array_equal(certified[0], original[0])
        cases = dict(static_original_plan=static, unchecked_tool_switching=unchecked,
                     certified_tool_switching=certified)
        summaries = {}
        for name, commands in cases.items():
            rows = replay(c, commands)
            summaries[name] = compact(rows)
        record = dict(pair=[high_id, low_id], commands={k:v.tolist() for k,v in cases.items()},
                      summaries=summaries,
                      status="sampled counterfactual only; certificates in original pilot record")
        save(OUT/f"{high_id}_{low_id}_counterfactuals.json", record)
        results.append(record)
        print(json.dumps(dict(pair=record["pair"], summaries={k:dict(
            physical_passes=v["physical_passes"], service_passes=v["service_passes"],
            median_tts_veh_h=v["tts_veh_h"]["median"],
            median_priority_cost=v["priority_cost"]["median"],
            worst_deadline_n1_vehicles=v["deadline_n1_vehicles"]["maximum"])
            for k,v in summaries.items()}), indent=2))
    if not args.pair:
        save(OUT/"counterfactual_summary.json", results)


if __name__ == "__main__":
    main()
