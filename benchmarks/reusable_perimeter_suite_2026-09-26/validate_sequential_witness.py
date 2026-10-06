"""Independent adaptive integration of one discovered queue-overflow witness."""
from __future__ import annotations

import json
import numpy as np
from scipy.integrate import solve_ivp

from common import HERE, START, END, arrival, fixture, mpc, plant, save
from run_sequential_authority import OUT, replay


def validate(high_id):
    low_id = f"D{int(high_id[1:])-1:03d}"
    name = f"{high_id}_{low_id}"
    previous = json.loads((OUT/f"{name}_counterfactuals.json").read_text())
    c = json.loads((OUT/f"{name}.json").read_text())["contract"]
    commands = np.asarray(previous["commands"]["unchecked_tool_switching"])
    rows = replay(c, commands)
    sample = max((r for r in rows if not r["physical_pass"]),
                 key=lambda r:r["peak_queue_vehicles"])
    assert sample["profile"] == "constant_high"
    signs = np.asarray([1. if v == "+" else -1. for v in sample["corner"]])
    x = np.asarray(fixture()["center"]) + signs*c["width_vehicles"]/100
    pieces = ((START, 5/12), (5/12, 1/2), (1/2, END))
    peak_queue = x[6]
    min_margin = np.inf
    peak_time = float(START)
    for (left, right), command in zip(pieces, commands):
        def rhs(t, y):
            return plant.field(y, command, arrival(t, c)+c["residual_veh_h"]/100)
        sol = solve_ivp(rhs, (float(left), float(right)), x, method="DOP853",
                        rtol=1e-12, atol=1e-13, max_step=1/720, dense_output=True)
        assert sol.success
        grid = np.linspace(float(left), float(right), round(float(right-left)*3600)+1)
        states = sol.sol(grid).T
        margins = states @ mpc.STATE_A.T + mpc.STATE_B
        min_margin = min(min_margin, float(np.min(margins)))
        at = int(np.argmax(states[:, 6]))
        if states[at, 6] > peak_queue:
            peak_queue = float(states[at, 6])
            peak_time = float(grid[at])
        x = sol.y[:, -1]
    result = dict(pair=[high_id, low_id], corner=sample["corner"], profile="constant_high",
                  integration="DOP853, rtol=1e-12, atol=1e-13, maximum step 5 s; 1 s dense grid",
                  peak_queue_vehicles=100*peak_queue,
                  queue_limit_vehicles=100*plant.QUEUE_LIMIT,
                  peak_time_minutes=60*peak_time,
                  minimum_physical_margin_units=min_margin,
                  sampled_RK4_peak_queue_vehicles=sample["peak_queue_vehicles"],
                  absolute_replay_difference_vehicles=abs(100*peak_queue-sample["peak_queue_vehicles"]),
                  deadline_n1_vehicles=100*x[0],
                  deadline_limit_vehicles=c["deadline_vehicles"])
    assert result["peak_queue_vehicles"] > result["queue_limit_vehicles"]+5
    assert result["absolute_replay_difference_vehicles"] < .01
    save(OUT/f"{name}_independent_witness.json", result)
    return result


def main():
    for high_id in ("D025", "D043", "D061"):
        print(json.dumps(validate(high_id), indent=2))


if __name__ == "__main__":
    main()
