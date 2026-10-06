"""Independent ledger/Jacobian review and saved-record multi-region audit.

No reachability grid, optimizer or ODE experiment is rerun. Matrix diagnostics
use90digits; the proof remains based on interval bounds and analytic identities.
The all-step record audit shares the established interval backend, not an
independent floating-point implementation. All outputs have new filenames.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from mpmath import mp

import mfd_multiregion_transfer as tested
from mfd_deadline_certificates import I, J, imat
from mfd_incident_queue_bounds import queue_step_bounds

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
OUTPUT = RESULTS / "mfd_multiregion_transfer_independent_audit_results.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b, label):
    assert np.array_equal(np.asarray(a), np.asarray(b)), label


def reference(net, state, command, arrival, branch=None):
    """High-precision incoming-flow equations and analytic block derivatives."""
    r, nc, n = net.r, net.nc, net.n
    x = [mp.mpf(float(v)) for v in state]
    u = [mp.mpf(float(v)) for v in command]
    lam = [mp.mpf(float(v)) for v in arrival]
    a, b, c = map(lambda v: mp.mpf(float(v)), (tested.MFD_A, tested.MFD_B, tested.MFD_C))
    totals = [sum(x[i*r+j] for j in range(r)) for i in range(r)]
    speed = [a*v*v+b*v+c for v in totals]
    slope = [2*a*v+b for v in totals]
    f = mp.matrix(n, 1); jac = mp.matrix(n, n)
    admit, derivative = [], []
    for i in range(r):
        supply = (60*x[nc+i] < 120*u[r+i]) if branch is None else branch[i] == "supply"
        admit.append(60*x[nc+i] if supply else 120*u[r+i])
        derivative.append(mp.mpf(60) if supply else mp.mpf(0))
    for i in range(r):
        previous = (i-1) % r
        for j in range(r):
            row = i*r+j
            out_coefficient = mp.mpf(1) if i == j else u[i]
            incoming = u[previous]*x[previous*r+j]*speed[previous] if j != previous else mp.mpf(0)
            allocation = mp.mpf(float(net.B[row, i]))
            f[row] = mp.mpf(float(net.demand[row]))+allocation*admit[i]+incoming-out_coefficient*x[row]*speed[i]
            for k in range(r):
                jac[row, i*r+k] -= out_coefficient*((speed[i] if j == k else 0)+x[row]*slope[i])
                if j != previous:
                    jac[row, previous*r+k] += u[previous]*((speed[previous] if j == k else 0)+x[previous*r+j]*slope[previous])
            jac[row, nc+i] = allocation*derivative[i]
        f[nc+i] = lam[i]-admit[i]
        jac[nc+i, nc+i] = -derivative[i]
    completion = sum(x[i*r+i]*speed[i] for i in range(r))
    return f, jac, completion


def within(matrix, interval):
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            assert mp.mpf(float(interval.lo[i, j])) <= matrix[i,j] <= mp.mpf(float(interval.hi[i, j])), (i,j,"matrix interval violation")


def independent_network_checks():
    records, entries = [], 0
    with mp.workdps(90):
        for r in (2, 3):
            net = tested.Ring(r)
            for col in range(r):
                assert all(Fraction.from_float(float(v)) >= 0 for v in net.B[:, col])
                assert sum(Fraction.from_float(float(v)) for v in net.B[:, col]) == 1
            for i in range(net.n):
                for j in range(net.n):
                    inverse_product = sum(Fraction.from_float(float(net.transform[i,k]))*Fraction.from_float(float(net.inverse[k,j])) for k in range(net.n))
                    assert inverse_product == int(i == j)
            incidence = np.vstack((net.B, -np.eye(r)))
            for i in range(net.n):
                for j in range(r):
                    product = sum(Fraction.from_float(float(net.transform[i,k]))*Fraction.from_float(float(incidence[k,j])) for k in range(net.n))
                    assert product == (-1 if i == net.nc+j else 0)
            T, Ti = mp.matrix(net.transform.tolist()), mp.matrix(net.inverse.tolist())
            for seed in range(6):
                rng = np.random.default_rng(99130+10*r+seed)
                x = net.initial*rng.uniform(.65, 1.35, net.n)
                command = net.command(Fraction(seed % 2, 2))
                arrival = net.arrival(Fraction(1, 3))
                x[net.nc:] = (.2 if seed % 2 else 2.)
                if seed == 4:
                    x[0] = 0.
                f, jac, completion = reference(net, x, command, arrival)
                measured = np.asarray(tested.independent_field(net, x, command, arrival))
                production = np.asarray(net.field(x, command, arrival))
                residual = np.max(np.abs(production-np.array(list(f), dtype=float).ravel()))
                assert residual < 2e-10 and np.max(abs(production-measured)) < 2e-10
                assert abs(sum(f)-(sum(mp.mpf(float(d)) for d in net.demand)+sum(mp.mpf(float(v)) for v in arrival)-completion)) < mp.mpf("1e-75")
                for method in ("class", "inventory"):
                    expected = jac if method == "class" else T*jac*Ti
                    point_box = net.jacobian(I(x), command, method)
                    within(expected, point_box)
                    entries += net.n*net.n
                    widths = np.full(net.n, .04)
                    outer = I(x-widths, x+widths)
                    enclosure = net.jacobian(outer, command, method)
                    for signed_corner in (-1., 1.):
                        point = x+signed_corner*widths
                        _, corner_jac, _ = reference(net, point, command, arrival)
                        within(corner_jac if method == "class" else T*corner_jac*Ti, enclosure)
                        entries += net.n*net.n
                # Exact switch faces: both one-sided derivatives must be included.
                for queue in range(r):
                    kink = x.copy()
                    kink[net.nc+queue] = 2.*command[r+queue]
                    half = np.zeros(net.n); half[net.nc+queue] = 1e-8
                    for side in ("supply", "gate"):
                        branches = ["supply" if 60*x[net.nc+i] < 120*command[r+i] else "gate" for i in range(r)]
                        branches[queue] = side
                        _, one_sided, _ = reference(net, kink, command, arrival, branches)
                        within(one_sided, net.jacobian(I(kink-half, kink+half), command, "class"))
                        within(T*one_sided*Ti, net.jacobian(I(kink-half, kink+half), command, "inventory"))
                        entries += 2*net.n*net.n
                records.append(dict(regions=r, seed=seed, maximum_field_difference=float(residual), mass_balance90digit=True,
                    interval_point_box_corner_and_kink_checks=True))
    return dict(passed=True, reference_digits=90, deterministic_cases=len(records), interval_jacobian_entry_checks=entries,
        exact_allocation_inverse_incidence=True, independent_receiver_equations=True, cases=records,
        scope="finite-point numerical implementation diagnostics complement analytic block derivation")


def selected_matrix_reference(step):
    n = len(step["center"])
    duration = Fraction(step["right"])-Fraction(step["left"])
    with mp.workdps(90):
        h = mp.mpf(duration.numerator)/duration.denominator
        for key, el, eh in (("frozen_A", "E_lower", "E_upper"), ("comparison_M", "EM_lower", "EM_upper")):
            matrix = mp.matrix(step[key]); exponential = mp.expm(matrix*h)
            within(exponential, I(step[el], step[eh]))
            if key == "comparison_M":
                term = mp.eye(n)*h; integral = term.copy()
                for k in range(1, 161):
                    term = matrix*term*h/(k+1); integral += term
                within(integral, I(step["Q_lower"], step["Q_upper"]))
    return 3*n*n


def audit_proof(summary, case):
    path = ROOT/case["proof_file"]
    assert sha(path) == case["proof_sha256"]
    proof = json.loads(path.read_text())
    desc = proof["descriptor"]
    assert tested.digest(desc) == proof["descriptor_sha256"]
    assert desc["source_sha256"] == summary["source_sha256"]
    assert desc["protocol_sha256"] == summary["protocol_sha256"]
    net = tested.Ring(case["regions"])
    assert desc["network"] == net.descriptor() and desc["method"] == case["method"]
    L = net.transform if case["method"] == "inventory" else np.eye(net.n)
    T = net.inverse if case["method"] == "inventory" else np.eye(net.n)
    same(L, desc["transform"], "coordinate matrix"); same(T, desc["inverse"], "inverse matrix")
    for block in desc["network"]["installed_plan"]:
        assert all(0 <= Fraction.from_float(float(u)) <= 1 for u in block["command"])
    widths = np.asarray(desc["physical_half_widths"])
    initial_interval_G = imat(L, np.diag(widths))
    G = initial_interval_G.mid(); rho = tested.rowsum(initial_interval_G-I(G))
    same(G, proof["initial"]["G"], "initial G"); same(rho, proof["initial"]["rho"], "initial rho")
    center = net.initial.copy()
    physical = I(center)+I(-widths,widths)
    same((I(net.limits)-imat(net.rows,physical)).lo, proof["initial_physical_margins"], "initial physical support")
    assert min(proof["initial_physical_margins"]) >= 0
    assert proof["structural_positivity"] == tested.positivity(net,widths,desc["per_queue_residual_vehicles_h"]/100.)
    queues = proof["initial"]["queues"]
    C = imat(net.rows, T)
    joined = I(np.vstack((np.eye(net.n), C.lo)), np.vstack((np.eye(net.n), C.hi)))
    clock, safe_clock = Fraction(0), Fraction(0)
    selected = {0,len(proof["steps"])//2,len(proof["steps"])-1}
    diagnostics = []
    for index, step in enumerate(proof["steps"]):
        left,right = Fraction(step["left"]),Fraction(step["right"])
        assert left == clock and 0 < right-left <= tested.NOMINAL_STEP
        assert not any(left < knot < right for knot in map(Fraction,("1/4","1/2","3/4","1")))
        same(center,step["center"],"center chain"); same(G,step["initial_G"],"G chain"); same(rho,step["initial_rho"],"rho chain")
        assert queues == step["queue_initial"] and step["generator_or_remainder_contracted"] is False
        command,arrival = net.command(left),net.arrival(left)
        same(command,step["command"],"commands");same(arrival,step["arrival"],"arrivals")
        residual = desc["per_queue_residual_vehicles_h"]/100.
        actual_queues = [queue_step_bounds(*queues[i],right-left,I(arrival[i]),I(0.),residual,command[net.r+i]) for i in range(net.r)]
        assert actual_queues == step["queues"]
        h = tested.rat(right-left);tau=I(0.,h.hi)
        velocity,acceleration = np.asarray(step["velocity"]),np.asarray(step["acceleration"])
        curve = I(center)+I(velocity)*tau+I(acceleration)*tau*tau
        hulls = [[min(float(curve.lo[net.nc+i]),actual_queues[i]["continuous"][0]),
                  max(float(curve.hi[net.nc+i]),actual_queues[i]["continuous"][1])] for i in range(net.r)]
        same(hulls,step["mean_value_queue_hulls"],"mean-value queue hull")
        trial=np.asarray(step["attempts"][-1]["trial"])
        physical_support=imat(I(abs(T)),I(trial)).hi
        corridor=curve+I(-physical_support,physical_support)
        for i in range(net.r):
            corridor.lo[net.nc+i]=max(corridor.lo[net.nc+i],hulls[i][0]);corridor.hi[net.nc+i]=min(corridor.hi[net.nc+i],hulls[i][1])
        same(corridor.lo,step["corridor_lower"],"corridor lower");same(corridor.hi,step["corridor_upper"],"corridor upper")
        jac=net.jacobian(corridor,command,case["method"])
        same(jac.lo,step["interval_J_lower"],"Jacobian lower");same(jac.hi,step["interval_J_upper"],"Jacobian upper")
        A=net.jacobian(I(center),command,case["method"]).mid();same(A,step["frozen_A"],"frozen matrix")
        M=jac.mag();np.fill_diagonal(M,np.diag(jac.hi));same(M,step["comparison_M"],"Metzler comparison")
        flow,ftail=tested.signed_range(joined,A,G,h);flows=tested.rowsum(flow)
        same(flows,step["generator_flow_support"],"signed flow support");assert ftail==step["tails"]["generator"]
        remflow,rtail=tested.signed_range(np.eye(net.n),M,np.diag(rho),h);assert rtail==step["tails"]["remainder"]
        mismatch=tested.rowsum(imat(jac-I(A),flow[:net.n]));same(mismatch,step["generator_mismatch"],"mismatch")
        base=(I(step["defect_upper"])+I(tested.rowsum(I(L[:,net.nc:])))*I(residual)).hi;same(base,step["base_force"],"disturbance channel")
        force=(I(mismatch)+I(base)).hi;same(force,step["force_upper"],"forcing")
        Q=I(step["Q_lower"],step["Q_upper"]);EM=I(step["EM_lower"],step["EM_upper"])
        assert np.min(Q.hi)>=0 and np.min(EM.hi)>=0
        convolution=imat(I(Q.hi),I(force)).hi;same(convolution,step["convolution"],"convolution")
        remainder=(I(tested.rowsum(remflow))+I(convolution)).hi;same(remainder,step["continuous_remainder"],"continuous remainder")
        radius=(I(flows[:net.n])+I(remainder)).hi;same(radius,step["attempts"][-1]["radius"],"radius")
        assert np.all(radius<trial)
        rowcurve=imat(net.rows,center)+imat(net.rows,velocity)*tau+imat(net.rows,acceleration)*tau*tau
        error=I(flows[net.n:])+imat(I(C.mag()),I(remainder))
        same(rowcurve.lo,step["row_predictor_lower"],"row lower");same(rowcurve.hi,step["row_predictor_upper"],"row upper")
        same(error.hi,step["row_error_upper"],"row error")
        margins=(I(net.limits)-rowcurve-error).lo.copy();same(margins,step["raw_physical_margins"],"raw margins")
        margins[:net.nc]=np.maximum(margins[:net.nc],0.)
        for i in range(net.r):
            margins[net.nc+net.r+i]=max(margins[net.nc+net.r+i],actual_queues[i]["continuous"][0])
            margins[net.nc+2*net.r+i]=max(margins[net.nc+2*net.r+i],float((I(20.)-I(actual_queues[i]["continuous"][1])).lo))
        same(margins,step["physical_margins"],"independent physical supports")
        if min(margins)>=0:safe_clock=right
        else:assert index==len(proof["steps"])-1 and not proof["continuous_physical_verified"]
        endpoint=I(center)+I(velocity)*h+I(acceleration)*h*h;center_next=endpoint.mid()
        rounding=imat(L,endpoint-I(center_next)).mag()
        E=I(step["E_lower"],step["E_upper"]);EG=imat(E,G);G_next=EG.mid()
        rho_next=(I(tested.rowsum(EG-I(G_next)))+imat(I(EM.hi),I(rho))+I(convolution)+I(rounding)).hi
        same(center_next,step["next_center"],"endpoint center");same(G_next,step["next_G"],"endpoint generator");same(rho_next,step["next_rho"],"endpoint remainder")
        if index in selected:
            for matrix,elo,ehi,tailname in ((A,"E_lower","E_upper","frozen"),(M,"EM_lower","EM_upper","comparison")):
                ex,qi,tail=tested.exp_integral(matrix,h)
                same(ex.lo,step[elo],"series lower");same(ex.hi,step[ehi],"series upper");assert tail==step["tails"][tailname]
                if tailname=="comparison":same(qi.lo,step["Q_lower"],"Q lower");same(qi.hi,step["Q_upper"],"Q upper")
            # Reassemble exact predictor defect polynomial, no ODE tolerance.
            polys=[J([center[k],velocity[k],acceleration[k]]) for k in range(net.n)]
            mv,ma=imat(L,velocity),imat(L,acceleration);defect=np.zeros(net.n)
            for branches in tested.branch_choices(net,curve,command):
                values=net.field(polys,command,[J([v]) for v in arrival],branches,case["method"])
                residuals=[values[k]-J([mv[k],I(2.)*ma[k]]) for k in range(net.n)]
                bound=I(np.zeros(net.n));power=I(1.)
                for degree in range(max(len(v.c) for v in residuals)):
                    coefficient=tested.interval_vector([v.c[degree] if degree<len(v.c) else I(0.) for v in residuals])
                    bound=bound+I(coefficient.mag())*power;power=power*I(h.hi)
                defect=np.maximum(defect,bound.hi)
            same(defect,step["defect_upper"],"exact residual coefficients")
            diagnostics.append(dict(index=index,left=str(left),right=str(right),reference_matrix_entries=selected_matrix_reference(step),polynomial_defect_reassembled=True))
        center,G,rho=center_next,G_next,rho_next;queues=[q["endpoint"] for q in actual_queues];clock=right
    assert str(clock)==proof["reached_time"]
    same(center,proof["terminal"]["center"],"terminalcenter");same(G,proof["terminal"]["G"],"terminalG");same(rho,proof["terminal"]["rho"],"terminalrho")
    assert queues==proof["terminal"]["queues"]
    if proof["continuous_physical_verified"]:
        assert clock==1 and safe_clock==1
        tr=imat(net.regional_rows,T)
        terminal=imat(net.regional_rows,center)+I(tested.rowsum(imat(tr,G)))+imat(I(tr.mag()),rho)
        for i,service in enumerate(proof["service_rows"]):
            assert service["passed"]==bool(terminal[i].hi<=27.)
            assert service["upper_vehicles"]==float((terminal[i]*I(100.)).hi)
        assert proof["scheduled_service_verified"]==all(s["passed"] for s in proof["service_rows"])
    assert proof["accepted"]==(proof["continuous_physical_verified"] and proof["scheduled_service_verified"])
    assert proof["accepted"]==case["accepted"] and proof["failure"]==case["failure"]
    regional_widths=[float(sum(Fraction.from_float(float(widths[i*net.r+j])) for j in range(net.r))*100) for i in range(net.r)]
    return dict(case_id=case["case_id"],passed=True,steps=len(proof["steps"]),source_file=case["proof_file"],source_sha256=sha(path),
        accepted=proof["accepted"],physical=proof["continuous_physical_verified"],reached_time=proof["reached_time"],
        verified_physical_prefix_exact_h=str(safe_clock),exact_binary_regional_half_widths_vehicles=regional_widths,
        selected_matrix_and_defect_diagnostics=diagnostics,actual_trial_attempts=sum(len(s["attempts"]) for s in proof["steps"]))


def run():
    began=time.perf_counter();source=RESULTS/"mfd_multiregion_transfer_results.json"
    summary=json.loads(source.read_text())
    assert summary.get("complete") and len(summary["cases"])==16
    result=dict(status="RUNNING",passed=False,audit_source_sha256=sha(Path(__file__)),summary_sha256=sha(source),
        scope="Independent90digit receiver-ledger/Jacobian diagnostics; all saved interval-step chains/supports; selected matrix/defect reconstructions; shared interval backend.",
        source_audits=[],network_checks=independent_network_checks(),proofs=[])
    def save():OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name,expected in summary["source_sha256"].items():
        assert sha(ROOT/name)==expected and sha(ROOT/summary["source_snapshots"][name])==expected
        result["source_audits"].append(dict(path=name,sha256=expected,passed=True))
    save()
    for case in summary["cases"]:
        record=audit_proof(summary,case);result["proofs"].append(record);save()
        print(json.dumps({k:record[k] for k in ("case_id","passed","steps","accepted")}),flush=True)
    # Confirm exact physical matching between coordinate ablations.
    for r in (2,3):
        for width,residual in ((1,60),(10,60),(25,60),(25,300)):
            paths=[ROOT/next(c["proof_file"] for c in summary["cases"] if c["regions"]==r and c["method"]==method and c["width_vehicles"]==width and c["residual_veh_h"]==residual) for method in ("class","inventory")]
            descriptors=[json.loads(p.read_text())["descriptor"] for p in paths]
            for key in ("network","physical_half_widths","per_queue_residual_vehicles_h","nominal_step","minimum_step","power_order"):
                assert descriptors[0][key]==descriptors[1][key]
    assert sha(source)==result["summary_sha256"]
    result.update(status="PASS",passed=True,proof_count=len(result["proofs"]),interval_step_count=sum(p["steps"] for p in result["proofs"]),
        selected_matrix_reference_entry_checks=sum(d["reference_matrix_entries"] for p in result["proofs"] for d in p["selected_matrix_and_defect_diagnostics"]),
        matched_coordinate_ablation_contracts=8,wall_seconds=time.perf_counter()-began)
    save();print(json.dumps({k:result[k] for k in ("status","proof_count","interval_step_count","wall_seconds")}),flush=True)
    return result


if __name__=="__main__":run()
