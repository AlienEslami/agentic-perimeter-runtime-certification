"""Future-state certification with timed search and retained execution.

The original scripts and numerical verifier are unchanged. Retention uses the
incumbent's original proof; replacement receives a new proof at its future slot.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import itertools
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
import common
from common import *

OUT = HERE / "results/delayed_handover"
OLD_SOURCES = common.portable_source_hashes


def sources():
    result = OLD_SOURCES()
    for p in (HERE / "DELAYED_PROTOCOL.md", Path(__file__)):
        result[p.relative_to(HERE).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


affine_module.source_hashes = sources


def inverse_exact(matrix):
    n = len(matrix)
    a = [[Fraction.from_float(float(v)) for v in row] +
         [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j] != 0)
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [v / scale for v in a[j]]
        for i in range(n):
            if i != j:
                scale = a[i][j]
                a[i] = [v - scale * w for v, w in zip(a[i], a[j])]
    return [row[n:] for row in a]


def activation_box(prefix):
    terminal = prefix["terminal_shape"]
    inverse = inverse_exact(terminal["metric_L"])
    radii = [Fraction.from_float(float(r)) for r in terminal["vector_radius"]]
    exact = [sum(abs(v) * r for v, r in zip(row, radii)) for row in inverse]
    widths = []
    for support in exact:
        value = float(support)
        if Fraction.from_float(value) < support:
            value = float(np.nextafter(value, np.inf))
        widths.append(value)
    return np.asarray(terminal["center"]), np.asarray(widths), dict(
        rule="exact binary-rational inverse metric and outward physical coordinate supports",
        exact_widths_units=[str(v) for v in exact], width_vehicles=(100*np.asarray(widths)).tolist(),
        source_prefix_proof=prefix["proof_id"])


def proof(center, widths, controls, durations, now, c, service=True):
    checks = ([dict(time_h=END, coordinate=0, upper=c["deadline_vehicles"]/100)] if service else [])
    return verify_command_plan(center, controls, durations, initial_time_h=now,
        amplitude=c["amplitude"], duration=c["duration_h"], initial_box=widths,
        residual_bound=c["residual_veh_h"]/100, step_seconds=15,
        verifier=AffineGeneratorVerifier(), service_checks=checks, retain_steps=False)


def command_at(controls, edges, clock):
    for i, end in enumerate(edges[1:]):
        if clock < float(end)-1e-11 or i == len(controls)-1:
            return controls[i]
    raise AssertionError("No active block")


def align(now, candidate, incumbent):
    new_change = now + Fraction(1, 12)
    old_change = START + Fraction(1, 12)
    edges = sorted({now, END, *[t for t in (new_change, old_change) if now < t < END]})
    controls, backup = [], []
    for left in edges[:-1]:
        controls.append(candidate[0 if left < new_change else 1])
        backup.append(incumbent[0 if left < old_change else 1])
    return np.asarray(controls), np.asarray(backup), edges


def compile_rows(rows):
    # Apply the identical existing row compiler without assuming two blocks.
    return np.asarray([compile_commands(np.vstack((u, u)))[0][0] for u in rows])


def simulation(c, incumbent, replacement, edges, center, widths, adaptive=False):
    # Always start at the original uncertainty corners, not at an arbitrary
    # point of the wider activation box.
    signs = np.ones((1, 7)) if adaptive else np.asarray(list(itertools.product((-1., 1.), repeat=7)))
    profiles = ["constant_high"] if adaptive else PROFILES
    rows = []
    if adaptive:
        initial = np.asarray(fixture()["center"])+signs[0]*c["width_vehicles"]/100
        y = np.r_[initial, np.zeros(5)]
        all_edges = sorted({START, END, START+Fraction(1, 12), *edges})
        peak = initial[6]; minimum = np.inf; membership = None
        for left, right in zip(all_edges[:-1], all_edges[1:]):
            u = (command_at(incumbent, [START, START+Fraction(1, 12), END], float(left))
                 if left < edges[0] else command_at(replacement, edges, float(left)))
            def rhs(t, yy):
                x = yy[:7]; offer = arrival(t, c)+c["residual_veh_h"]/100
                return np.r_[plant.field(x, u, offer), 100*(x[0]+x[1]+x[6]), 100*x[6],
                    100*plant.departure_rate(x,u), 100*(x[0]+c["weight"]*x[1]+x[6]),
                    100*(sum(plant.INTERNAL_DEMAND)+offer)]
            sol = solve_ivp(rhs, (float(left), float(right)), y, method="DOP853",
                rtol=1e-11, atol=1e-12, dense_output=True, max_step=1/360)
            assert sol.success
            grid = np.linspace(float(left),float(right),max(2,round(float(right-left)*1800)+1))
            xs = sol.sol(grid)[:7].T
            minimum = min(minimum,float(np.min(xs@mpc.STATE_A.T+mpc.STATE_B)))
            peak = max(peak,float(np.max(xs[:,6])))
            y = sol.y[:,-1]
            if right == edges[0]:
                membership = float(np.min(widths-abs(y[:7]-center)))
        return dict(tts_veh_h=float(y[7]),queue_veh_h=float(y[8]),completed_trips=float(y[9]),
            priority_cost=float(y[10]),peak_queue_vehicles=100*peak,deadline_n1_vehicles=100*y[0],
            physical_pass=bool(minimum>=-1e-7),service_pass=bool(100*y[0]<=c["deadline_vehicles"]+1e-6),
            activation_box_membership_margin_units=membership,
            conservation_residual_vehicles=float(100*(y[0]+y[1]+y[6]-sum(initial[[0,1,6]]))-y[11]+y[9]))
    x = np.repeat(np.asarray(fixture()["center"])[None,:]+signs*c["width_vehicles"]/100,5,axis=0)
    initial = x.copy(); totals=np.zeros((len(x),4)); offer_total=np.zeros(len(x))
    minimum=np.full(len(x),np.inf); peak=x[:,6].copy(); membership=None
    pindex=np.tile(np.arange(5),len(signs)); h=1/360
    for k in range(240):
        t=float(START)+k*h
        u=(command_at(incumbent,[START,START+Fraction(1,12),END],t) if t<float(edges[0])-1e-11
           else command_at(replacement,edges,t))
        def rhs(y,clock):
            forcing_clock=min(clock,np.nextafter(t+h,t))
            r=np.asarray([residual(t+h/2 if p=="alternating_5min" else forcing_clock,
                c["residual_veh_h"]/100,p) for p in profiles])[pindex]
            offer=arrival(clock,c)+r; f,dep=batch_field(y,u,offer)
            costs=100*np.stack((y[:,0]+y[:,1]+y[:,6],y[:,6],dep,
                y[:,0]+c["weight"]*y[:,1]+y[:,6]),axis=1)
            return f,costs,100*(sum(plant.INTERNAL_DEMAND)+offer)
        f1,c1,a1=rhs(x,t);x2=x+h*f1/2
        f2,c2,a2=rhs(x2,t+h/2);x3=x+h*f2/2
        f3,c3,a3=rhs(x3,t+h/2);x4=x+h*f3
        f4,c4,a4=rhs(x4,t+h);new=x+h/6*(f1+2*f2+2*f3+f4)
        totals+=h/6*(c1+2*c2+2*c3+c4);offer_total+=h/6*(a1+2*a2+2*a3+a4)
        for y in (x,x2,x3,x4,new):
            minimum=np.minimum(minimum,np.min(y@mpc.STATE_A.T+mpc.STATE_B,axis=1));peak=np.maximum(peak,y[:,6])
        x=new
        if abs(t+h-float(edges[0]))<1e-9:
            membership=np.min(widths-abs(x-center),axis=1)
    balance=100*(x[:,0]+x[:,1]+x[:,6]-initial[:,0]-initial[:,1]-initial[:,6])-offer_total+totals[:,2]
    for i in range(len(x)):
        rows.append(dict(corner="".join("+" if v>0 else "-" for v in signs[i//5]),profile=profiles[i%5],
            tts_veh_h=float(totals[i,0]),queue_veh_h=float(totals[i,1]),completed_trips=float(totals[i,2]),
            priority_cost=float(totals[i,3]),peak_queue_vehicles=float(100*peak[i]),
            deadline_n1_vehicles=float(100*x[i,0]),physical_pass=bool(minimum[i]>=-1e-7),
            service_pass=bool(100*x[i,0]<=c["deadline_vehicles"]+1e-6),
            conservation_residual_vehicles=float(balance[i]),activation_box_membership_margin_units=float(membership[i])))
    return rows


def run_case(c, delay):
    path=OUT/f"{c['id']}_{delay:03d}s.json"
    if path.exists():return path.stem,"cached"
    original=json.loads((HERE/"results/contracts"/(c["id"]+".json")).read_text())
    if original["backup_available"]:
        active=original["policies"]["continuation"]; source="original_continuation"
    else:
        reserve=json.loads((HERE/"results/reserve_refresh"/(c["id"]+".json")).read_text())
        active=reserve["policies"]["fresh_reserve"]; source="fresh_reserve"
    assert active["certificate"]["accepted"]
    incumbent=np.asarray(active["commands"]); now=START+Fraction(delay,3600)
    started=time.perf_counter()
    prefix=proof(np.asarray(fixture()["center"]),c["width_vehicles"]/100,
        [incumbent[0]],[now-START],START,c,service=False)
    assert prefix["accepted"]
    center,widths,box_audit=activation_box(prefix)
    result=dict(contract=c,delay_seconds=delay,activation_time_fraction=str(now),active_source=source,
        active_proof_id=active["certificate"]["proof_id"],prefix_certificate=prefix,
        activation_center_units=center.tolist(),activation_width_units=widths.tolist(),box_audit=box_audit,
        repair_attempts=[],replacement_installed=False)
    proposal,report=solve_service(center,float(now),arrival_fun=lambda t:arrival(t,c),
        service_target_vehicles=c["deadline_vehicles"],horizon_h=float(END-now),
        warm=command_at(incumbent,[START,START+Fraction(1,12),END],float(now)),
        cost_row=np.array([1.,c["weight"],0,0,0,0,1.]))
    result["solver"]=report
    chosen=None; chosen_cert=None; selected_theta=None
    # Even on timeout, use a partition of the unchanged absolute incumbent.
    placeholder=np.tile(command_at(incumbent,[START,START+Fraction(1,12),END],float(now)),(2,1))
    candidate,backup,edges=align(now,placeholder if proposal is None else compile_commands(proposal[1])[0],incumbent)
    durations=[r-l for l,r in zip(edges[:-1],edges[1:])]
    result["edges_fraction"]=[str(v) for v in edges]
    if proposal is not None and time.perf_counter()-started < delay:
        try:
            candidate_cert=proof(center,widths,candidate,durations,now,c)
            result["candidate_certificate"]=candidate_cert;result["candidate_commands"]=candidate.tolist()
            if candidate_cert["accepted"] and time.perf_counter()-started<=delay:
                chosen,chosen_cert,selected_theta=candidate,candidate_cert,0.
            elif time.perf_counter()-started<delay:
                endpoint=proof(center,widths,backup,durations,now,c)
                result["fresh_endpoint_certificate"]=endpoint
                if endpoint["accepted"]:
                    for theta in np.arange(1,21)/20:
                        if time.perf_counter()-started>=delay:break
                        controls=compile_rows((1-theta)*candidate+theta*backup)
                        cert=proof(center,widths,controls,durations,now,c)
                        ready=time.perf_counter()-started
                        result["repair_attempts"].append(dict(theta=float(theta),commands=controls.tolist(),
                            certificate=cert,ready_seconds=ready,within_budget=ready<=delay))
                        if cert["accepted"]:
                            if ready<=delay:chosen,chosen_cert,selected_theta=controls,cert,float(theta)
                            break
        except ValueError as error:
            # The outer box can include physically impossible states. A new
            # certificate cannot start from it; incumbent authority is intact.
            result["replacement_information_rejected"]=str(error)
    ready=time.perf_counter()-started
    if chosen is not None:
        assert chosen_cert["accepted"] and ready<=delay
        result.update(replacement_installed=True,selected_theta=selected_theta,
            installed_commands=chosen.tolist(),installed_certificate=chosen_cert)
    else:
        chosen=backup
        result.update(selected_theta=None,installed_commands=chosen.tolist(),
            retention_authority="unchanged original incumbent certificate and absolute schedule")
    result["preparation_seconds"]=ready
    result["late_result"]=ready>delay
    rows=simulation(c,incumbent,chosen,edges,center,widths)
    result["realizations"]=rows;result["summary"]=summarize(rows)
    adverse=simulation(c,incumbent,chosen,edges,center,widths,adaptive=True)
    result["adaptive_adverse_replay"]=adverse
    counterpart=next(r for r in rows if r["corner"]=="+++++++" and r["profile"]=="constant_high")
    errors={k:abs(adverse[k]-counterpart[k]) for k in ("tts_veh_h","queue_veh_h","completed_trips",
        "priority_cost","peak_queue_vehicles","deadline_n1_vehicles")}
    assert max(errors.values())<.08,errors
    assert result["summary"]["physical_passes"]==640 and result["summary"]["service_passes"]==640
    assert min(r["activation_box_membership_margin_units"] for r in rows)>=-1e-7
    assert adverse["physical_pass"] and adverse["service_pass"]
    result["independent_audit"]=dict(passed=True,errors=errors,
        minimum_activation_box_membership_margin_units=min(r["activation_box_membership_margin_units"] for r in rows))
    save(path,result)
    return path.stem,dict(installed=result["replacement_installed"],theta=selected_theta,
        seconds=round(ready,2),maximum_width_vehicles=round(max(box_audit["width_vehicles"]),2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--only");args=parser.parse_args()
    grid=json.loads((HERE/"contracts.json").read_text())
    selected=[c for c in grid if c["task"]=="D" and (c["split"]=="held_out" or c["id"]=="D027" or
        (c["width_vehicles"] in (1,50) and c["residual_veh_h"] in (60,300) and
         c["deadline_vehicles"] in (2750,2950) and c["weight"]==4))]
    assert len(selected)==17
    jobs=[(c,delay) for c in selected for delay in (60,120,300)]
    if args.only:jobs=[(c,d) for c,d in jobs if f"{c['id']}_{d:03d}s" in args.only.split(",")]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for f in as_completed([executor.submit(run_case,c,d) for c,d in jobs]):print(f.result(),flush=True)


if __name__=="__main__":main()
