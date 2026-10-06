"""Full fixed-grid discovery scan specified in SEQUENTIAL_AUTHORITY_PROTOCOL.md."""
from __future__ import annotations

import json
import numpy as np

from common import HERE, save
from run_sequential_authority import read, replay

OUT = HERE / "results/sequential_authority_2026-10-01"


def main():
    records = []
    for high_number in range(13, 92, 2):
        high_id, low_id = f"D{high_number:03d}", f"D{high_number-1:03d}"
        high = read(high_id)
        low = read(low_id, "full_scenario")
        incumbent = high["policies"].get("grid_repair")
        strong = low["policies"].get("full_scenario_mpc")
        hard_keys = ("amplitude", "duration_h", "width_vehicles", "residual_veh_h",
                     "deadline_vehicles")
        matched = (all(high["contract"][k] == low["contract"][k] for k in hard_keys)
                   and high["contract"]["weight"] == 4
                   and low["contract"]["weight"] == 2)
        eligible = bool(matched and incumbent and strong
                        and incumbent["certificate"]["accepted"]
                        and strong["certificate"]["accepted"])
        record = dict(pair=[high_id, low_id], matched_hard_contract=matched,
                      eligible=eligible)
        if eligible:
            first = np.asarray(incumbent["commands"])
            second = np.asarray(strong["commands"])
            third = np.asarray(high["policies"]["nominal_mpc"]["commands"])
            commands = np.asarray([first[0], second[1], third[1]])
            rows = replay(high["contract"], commands)
            record.update(commands=commands.tolist(),
                          physical_violations=sum(not r["physical_pass"] for r in rows),
                          service_violations=sum(not r["service_pass"] for r in rows),
                          minimum_physical_margin=min(r["minimum_state_margin"] for r in rows),
                          maximum_deadline_n1_vehicles=max(r["deadline_n1_vehicles"] for r in rows),
                          deadline_limit_vehicles=high["contract"]["deadline_vehicles"],
                          median_tts_veh_h=float(np.median([r["tts_veh_h"] for r in rows])),
                          median_priority_cost=float(np.median([r["priority_cost"] for r in rows])))
        records.append(record)
        print(json.dumps(record))
    save(OUT/"full_grid_scan.json", records)
    print("eligible", sum(r["eligible"] for r in records),
          "physical_cases", sum(r.get("physical_violations", 0)>0 for r in records),
          "service_cases", sum(r.get("service_violations", 0)>0 for r in records))


if __name__ == "__main__":
    main()
