"""Whole-family queue-branch inequalities for the 36 fixed deadline plans.

The inequalities cover every initial queue value and every measurable bounded
arrival residual on the unchanged 40-minute horizon. They establish when the
arrival signal can influence regional traffic, but not the service deadline.
"""
from __future__ import annotations

import json
import numpy as np
from mpmath import iv

from common import HERE, fixture, save
from run_falsification import records
from mfd_incident_affine_reachability import I

OUT = HERE / "results/directional_service_2026-10-01"


def interval_exp_negative_five():
    value = iv.exp(iv.mpf(-5))
    return I(np.nextafter(float(value.a), -np.inf),
             np.nextafter(float(value.b), np.inf))


def audit():
    zcenter = I(fixture()["center"][6])
    e = interval_exp_negative_five()
    reciprocal_sixty = iv.mpf(1)/iv.mpf(60)
    inv60 = I(np.nextafter(float(reciprocal_sixty.a), -np.inf),
              np.nextafter(float(reciprocal_sixty.b), np.inf))
    entries = []
    for path, record in records():
        c = record["contract"]
        assert c["amplitude"] == .3 and c["duration_h"] == 1.
        commands = np.asarray(record["policies"]["full_scenario_mpc"]["commands"])
        assert np.all(commands[:,3] == 1.)
        width = I(c["width_vehicles"]/100)
        residual = I(c["residual_veh_h"]/100)
        arrival = I(60.)*(I(1.)+I(c["amplitude"]))
        lo, hi = arrival-residual, arrival+residual
        zlo, zhi = zcenter-width, zcenter+width
        first_threshold = I(2.)*I(commands[0,2])
        second_threshold = I(2.)*I(commands[1,2])
        first_queue_upper = max(float(zhi.hi), float((hi*inv60).hi))
        first_supply_margin = float((first_threshold-I(first_queue_upper)).lo)
        lower_equilibrium = lo*inv60
        lower_queue_at_switch = lower_equilibrium + (zlo-lower_equilibrium)*e
        second_gate_margin = float((lower_queue_at_switch-second_threshold).lo)
        second_queue_growth = float((lo-I(120.)*I(commands[1,2])).lo)
        entry = dict(id=c["id"], first_supply_margin_queue_units=first_supply_margin,
                     second_gate_margin_queue_units=second_gate_margin,
                     second_queue_growth_lower_units_per_h=second_queue_growth,
                     first_supply_all_inputs=first_supply_margin>0,
                     second_gate_all_inputs=second_gate_margin>0 and second_queue_growth>=0)
        entries.append(entry)
    summary = dict(case_count=len(entries),
                   first_supply_all_cases=all(v["first_supply_all_inputs"] for v in entries),
                   second_gate_all_cases=all(v["second_gate_all_inputs"] for v in entries),
                   minimum_first_supply_margin_queue_units=min(v["first_supply_margin_queue_units"] for v in entries),
                   minimum_second_gate_margin_queue_units=min(v["second_gate_margin_queue_units"] for v in entries),
                   minimum_second_queue_growth_units_per_h=min(v["second_queue_growth_lower_units_per_h"] for v in entries),
                   consequence="Regional trajectories from minute 5 to deadline are independent of arrival residual after minute 5; the queue trajectory is not.",
                   service_certified_by_this_audit=False)
    save(OUT/"queue_branch_audit.json",dict(summary=summary,cases=entries))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    audit()
