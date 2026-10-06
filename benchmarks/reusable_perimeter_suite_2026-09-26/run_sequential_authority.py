"""Predeclared multi-decision authority pilot; see SEQUENTIAL_AUTHORITY_PROTOCOL.md."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from fractions import Fraction

import numpy as np

from common import (HERE, START, END, PROFILES, arrival, batch_field, fixture,
                    mpc, plant, residual, save, summarize)
from run_delayed_handover import activation_box, proof

OUT = HERE / "results/sequential_authority_2026-10-01"
T25 = Fraction(5, 12)
T30 = Fraction(1, 2)
PAIRS = (("D015", "D014"), ("D027", "D026"), ("D071", "D070"))


def read(case, kind="contracts"):
    return json.loads((HERE / "results" / kind / f"{case}.json").read_text())


def short_certificate(cert):
    return dict(accepted=cert["accepted"], proof_id=cert.get("proof_id"),
                physical=cert.get("verified_continuous_physical"),
                service=cert.get("verified_scheduled_service"),
                service_checks=cert.get("service_check_results"),
                failures=cert.get("failures"), wall_seconds=cert.get("wall_seconds"))


def replay(contract, commands):
    """Matched 128-corner x 5-profile numerical check of the executed schedule."""
    corners = np.asarray(list(itertools.product((-1., 1.), repeat=7)))
    x = np.repeat(np.asarray(fixture()["center"])[None, :]
                  + corners * contract["width_vehicles"] / 100, 5, axis=0)
    initial = x.copy()
    profile_index = np.tile(np.arange(5), len(corners))
    totals = np.zeros((len(x), 4))
    offered_integral = np.zeros(len(x))
    minimum = np.full(len(x), np.inf)
    peak_queue = x[:, 6].copy()
    h = 1 / 360
    for k in range(240):
        t = float(START) + k * h
        u = commands[0 if k < 30 else 1 if k < 60 else 2]
        weight = 4 if k < 30 or k >= 60 else 2

        def rhs(y, clock):
            forcing_clock = min(clock, np.nextafter(t+h, t))
            r = np.asarray([residual(t+h/2 if p == "alternating_5min" else forcing_clock,
                                     contract["residual_veh_h"]/100, p)
                            for p in PROFILES])[profile_index]
            offer = arrival(clock, contract) + r
            field, departures = batch_field(y, u, offer)
            costs = 100 * np.stack((y[:, 0]+y[:, 1]+y[:, 6], y[:, 6], departures,
                                    y[:, 0]+weight*y[:, 1]+y[:, 6]), axis=1)
            return field, costs, 100*(sum(plant.INTERNAL_DEMAND)+offer)

        f1, c1, a1 = rhs(x, t)
        x2 = x+h*f1/2
        f2, c2, a2 = rhs(x2, t+h/2)
        x3 = x+h*f2/2
        f3, c3, a3 = rhs(x3, t+h/2)
        x4 = x+h*f3
        f4, c4, a4 = rhs(x4, t+h)
        new = x+h*(f1+2*f2+2*f3+f4)/6
        totals += h*(c1+2*c2+2*c3+c4)/6
        offered_integral += h*(a1+2*a2+2*a3+a4)/6
        for y in (x, x2, x3, x4, new):
            minimum = np.minimum(minimum, np.min(y @ mpc.STATE_A.T+mpc.STATE_B, axis=1))
            peak_queue = np.maximum(peak_queue, y[:, 6])
        x = new

    balance = 100*((x[:, 0]+x[:, 1]+x[:, 6])
                   -(initial[:, 0]+initial[:, 1]+initial[:, 6])) \
              -offered_integral+totals[:, 2]
    rows = []
    for i in range(len(x)):
        rows.append(dict(corner="".join("+" if v > 0 else "-" for v in corners[i//5]),
                         profile=PROFILES[i % 5], tts_veh_h=float(totals[i, 0]),
                         queue_veh_h=float(totals[i, 1]), completed_trips=float(totals[i, 2]),
                         priority_cost=float(totals[i, 3]),
                         peak_queue_vehicles=float(100*peak_queue[i]),
                         deadline_n1_vehicles=float(100*x[i, 0]),
                         physical_pass=bool(minimum[i] >= -1e-7),
                         minimum_state_margin=float(minimum[i]),
                         service_pass=bool(100*x[i, 0] <= contract["deadline_vehicles"]+1e-6),
                         conservation_residual_vehicles=float(balance[i])))
    return rows


def run_pair(high_id, low_id):
    high = read(high_id)
    low = read(low_id)
    strong = read(low_id, "full_scenario")
    c = high["contract"]
    for key in ("amplitude", "duration_h", "width_vehicles", "residual_veh_h",
                "deadline_vehicles"):
        assert c[key] == low["contract"][key] == strong["contract"][key], key
    assert c["weight"] == 4 and low["contract"]["weight"] == 2
    incumbent = high["policies"]["grid_repair"]
    middle_source = strong["policies"]["full_scenario_mpc"]
    last_source = high["policies"]["nominal_mpc"]
    assert incumbent["certificate"]["accepted"]
    assert middle_source["certificate"]["accepted"]
    incumbent_commands = np.asarray(incumbent["commands"])
    middle_command = np.asarray(middle_source["commands"][1])
    last_command = np.asarray(last_source["commands"][1])
    start_center = np.asarray(fixture()["center"])
    start_width = c["width_vehicles"]/100
    result = dict(pair=[high_id, low_id], contract=c,
                  original_plan=incumbent_commands.tolist(),
                  original_proof=short_certificate(incumbent["certificate"]),
                  middle_source=middle_source["commands"],
                  middle_source_proof=short_certificate(middle_source["certificate"]),
                  last_source=last_source["commands"],
                  last_source_proof=short_certificate(last_source["certificate"]),
                  decisions=[])

    prefix25 = proof(start_center, start_width, [incumbent_commands[0]],
                     [T25-START], START, c, service=False)
    assert prefix25["accepted"], (high_id, prefix25.get("failures"))
    center25, width25, audit25 = activation_box(prefix25)
    result["prefix25"] = short_certificate(prefix25)
    result["box25"] = dict(center_units=center25.tolist(), width_units=width25.tolist(), audit=audit25)

    begun = time.perf_counter()
    try:
        candidate25 = proof(center25, width25, [middle_command], [END-T25], T25, c)
        accepted25 = bool(candidate25["accepted"])
        verdict25 = short_certificate(candidate25)
    except (ValueError, ArithmeticError) as exc:
        accepted25 = False
        verdict25 = dict(accepted=False, information_rejected=str(exc))
    result["decisions"].append(dict(minute=25, proposed_tool="full_scenario_mpc",
                                     proposed_command=middle_command.tolist(),
                                     accepted=accepted25, certificate=verdict25,
                                     seconds=time.perf_counter()-begun,
                                     outcome="replace" if accepted25 else "retain_verified_suffix"))
    active25 = middle_command if accepted25 else incumbent_commands[1]

    if accepted25:
        prefix30 = proof(center25, width25, [middle_command], [T30-T25], T25, c, service=False)
    else:
        prefix30 = proof(start_center, start_width, list(incumbent_commands),
                         [T25-START, T30-T25], START, c, service=False)
    assert prefix30["accepted"], (high_id, prefix30.get("failures"))
    center30, width30, audit30 = activation_box(prefix30)
    result["prefix30"] = short_certificate(prefix30)
    result["box30"] = dict(center_units=center30.tolist(), width_units=width30.tolist(), audit=audit30)

    begun = time.perf_counter()
    try:
        candidate30 = proof(center30, width30, [last_command], [END-T30], T30, c)
        accepted30 = bool(candidate30["accepted"])
        verdict30 = short_certificate(candidate30)
    except (ValueError, ArithmeticError) as exc:
        accepted30 = False
        verdict30 = dict(accepted=False, information_rejected=str(exc))
    result["decisions"].append(dict(minute=30, proposed_tool="nominal_mpc",
                                     proposed_command=last_command.tolist(),
                                     accepted=accepted30, certificate=verdict30,
                                     seconds=time.perf_counter()-begun,
                                     outcome="replace" if accepted30 else "retain_verified_suffix"))
    active30 = last_command if accepted30 else active25
    commands = np.asarray([incumbent_commands[0], active25, active30])
    result["executed_commands"] = commands.tolist()
    result["execution_authorities"] = [result["original_proof"]["proof_id"],
                                       verdict25.get("proof_id") if accepted25 else result["original_proof"]["proof_id"],
                                       verdict30.get("proof_id") if accepted30 else
                                       (verdict25.get("proof_id") if accepted25 else result["original_proof"]["proof_id"])]
    rows = replay(c, commands)
    result["summary"] = summarize(rows)
    result["minimum_physical_margin"] = min(r["minimum_state_margin"] for r in rows)
    result["realizations"] = rows
    assert result["summary"]["physical_passes"] == 640
    assert result["summary"]["service_passes"] == 640
    assert result["summary"]["maximum_conservation_error_vehicles"] < .1
    save(OUT/f"{high_id}_{low_id}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[p[0] for p in PAIRS])
    parser.add_argument("--pair", help="An additional predeclared high-weight Task D ID")
    args = parser.parse_args()
    summary = []
    pairs = ((args.pair, f"D{int(args.pair[1:])-1:03d}"),) if args.pair else PAIRS
    for high, low in pairs:
        if args.only and args.only != high:
            continue
        result = run_pair(high, low)
        row = dict(pair=result["pair"], outcomes=[d["outcome"] for d in result["decisions"]],
                   reasons=[d["certificate"].get("failures", d["certificate"].get("information_rejected"))
                            for d in result["decisions"]],
                   physical_passes=result["summary"]["physical_passes"],
                   service_passes=result["summary"]["service_passes"],
                   median_tts_veh_h=result["summary"]["tts_veh_h"]["median"],
                   median_priority_cost=result["summary"]["priority_cost"]["median"])
        summary.append(row)
        print(json.dumps(row, indent=2))
    if not args.only and not args.pair:
        save(OUT/"summary.json", summary)


if __name__ == "__main__":
    main()
