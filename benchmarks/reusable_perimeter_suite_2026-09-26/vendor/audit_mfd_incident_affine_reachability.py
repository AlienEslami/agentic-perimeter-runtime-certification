"""Audit saved signed-generator proofs without new optimization/ODE/proof runs.

Every stored interval step is checked for exact clocks, generator/remainder
chain, strict self-inclusion, signed physical supports, endpoint rounding and
queue composition. First/middle/last steps independently reconstruct the
Jacobian corridor, Metzler matrix, polynomial defect and interval matrix
functions. Shared outward arithmetic is used; this is a deterministic record
audit, not an independently implemented interval backend. Separate 90-digit
diagnostics test the matrix primitives and selected saved matrices.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from mpmath import mp

import mfd_incident_model as model
from mfd_incident_mpc_certification import constant_command_mode, exact_command_constraints, rational_lower
from mfd_incident_verifier import I, J, Pulse, branch_for_box, closed_field, evaluate_transformed_jacobian, imat, interval_fraction, jacobian, norm_upper
from mfd_deadline_certificates import isum
from mfd_incident_positive_reachability import initial_positivity
from mfd_incident_queue_bounds import queue_step_bounds
from mfd_incident_uncertainty_frontier import initial_supports
from audit_affine_matrix_functions_review import run as matrix_primitive_review

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
OUTPUT = RESULTS / "mfd_incident_affine_reachability_audit_results.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(actual, expected, label):
    assert np.array_equal(np.asarray(actual), np.asarray(expected)), label


def target(version):
    path = (RESULTS / "mfd_incident_affine_reachability_v2_executed_source.py" if version == 2
            else ROOT / "reproducibility/mfd_incident_affine_reachability.py")
    spec = importlib.util.spec_from_file_location(f"audited_affine_v{version}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def exact_grid(descriptor, pulse):
    cursor = Fraction(descriptor["initial_time"])
    cuts_extra = list(map(Fraction, descriptor["timecuts"]))
    grid = []
    for segment, duration in enumerate(map(Fraction, descriptor["durations"])):
        endpoint = cursor + duration
        cuts = [cursor] + sorted(set(c for c in pulse.knots + cuts_extra if cursor < c < endpoint)) + [endpoint]
        for left, right in zip(cuts[:-1], cuts[1:]):
            count = max(1, math.ceil(float((right-left)*3600)/descriptor["step_seconds"]))
            times = [left+(right-left)*Fraction(k, count) for k in range(count+1)]
            grid.extend((segment, a, b) for a, b in zip(times[:-1], times[1:]))
        cursor = endpoint
    return grid


def predictor(center, mode, left, right, pulse, L):
    h = interval_fraction(right-left)
    tau = I(0., h.hi)
    arrival0, slope = pulse.affine(left, right)
    branch, _ = branch_for_box(I(center), mode)
    branch = "gate" if branch == "both" else branch
    values = closed_field([I(v) for v in center], mode, arrival0, branch)
    fbox = I(np.array([v.lo for v in values]), np.array([v.hi for v in values]))
    velocity = fbox.mid()
    erow = np.zeros(7); erow[6] = 1.
    acceleration = ((imat(jacobian(I(center), mode, branch), fbox)+I(erow)*slope)*I(.5)).mid()
    endpoint = I(center)+I(velocity)*h+I(acceleration)*h*h
    next_center = endpoint.mid()
    rounding = imat(L, endpoint-I(next_center)).mag()
    curve = I(center)+I(velocity)*tau+I(acceleration)*tau*tau
    return dict(h=h, tau=tau, arrival0=arrival0, slope=slope, chosen=branch,
                velocity=velocity, acceleration=acceleration, curve_box=curve,
                next_center=next_center, rounding=rounding)


def selected_corridor_check(module, version, center, mode, geo, pred, step, residual):
    """Independent assembly of the recorded final successful trial."""
    L, inverse = geo["L"], geo["T"]
    h, tau = pred["h"], pred["tau"]
    trial = np.asarray(step["apriori_radius"])
    physical_support = imat(I(inverse.mag()), I(trial)).hi
    physical_box = pred["curve_box"]+I(-physical_support, physical_support)
    metric_curve = imat(L, I(center)-I(mode.center))+imat(L, pred["velocity"])*tau+imat(L, pred["acceleration"])*tau*tau
    metric_box = metric_curve+I(-trial, trial)
    if version == 3:
        queue_hull = I(*step["independent_physics"]["mean_value_queue_hull"])
        physical_box.lo[6] = max(physical_box.lo[6], float(queue_hull.lo))
        physical_box.hi[6] = min(physical_box.hi[6], float(queue_hull.hi))
        metric_queue = queue_hull-I(mode.center[6])
        metric_box.lo[6] = max(metric_box.lo[6], float(metric_queue.lo))
        metric_box.hi[6] = min(metric_box.hi[6], float(metric_queue.hi))
    branch, gap = branch_for_box(physical_box, mode)
    assert branch == step["active_branch"]
    same([float(gap.lo), float(gap.hi)], step["branch_gap_interval"], "branch gap")
    matrices = [evaluate_transformed_jacobian(geo["jacobian_record"], metric_box, smooth, centered="exact")
                for smooth in (("gate", "supply") if branch == "both" else (branch,))]
    M = np.maximum.reduce([matrix.mag() for matrix in matrices])
    np.fill_diagonal(M, np.maximum.reduce([np.diag(matrix.hi) for matrix in matrices]))
    same(M, step["affine"]["metzler_M"], "reconstructed robust Metzler matrix")
    A = evaluate_transformed_jacobian(geo["jacobian_record"], imat(L, I(center)-I(mode.center)), pred["chosen"], centered="exact").mid()
    same(A, step["affine"]["frozen_A"], "frozen linearization")
    curve = [J([center[i], pred["velocity"][i], pred["acceleration"][i]]) for i in range(7)]
    curve_branch, _ = branch_for_box(pred["curve_box"], mode)
    defects = []
    for smooth in (("gate", "supply") if curve_branch == "both" else (curve_branch,)):
        field = closed_field(curve, mode, J([pred["arrival0"], pred["slope"]]), smooth)
        residuals = [field[i]-J([pred["velocity"][i], I(2.)*I(pred["acceleration"][i])]) for i in range(7)]
        result, power = I(np.zeros(7)), I(1.)
        for degree in range(max(len(r.c) for r in residuals)):
            coefficient = I(np.array([r.c[degree].lo if degree < len(r.c) else 0. for r in residuals]),
                            np.array([r.c[degree].hi if degree < len(r.c) else 0. for r in residuals]))
            transformed = imat(L, coefficient)
            result = result+I(-transformed.mag(), transformed.mag())*power
            power = power*I(h.hi)
        defects.append(result.mag())
    defect = np.maximum.reduce(defects)
    assert float(max(defect)) == step["defect_upper"]
    base = (I(defect)+I(abs(L[:, 6]))*I(residual)).hi
    flow, _ = module.signed_flow_range(np.eye(7), A, step["affine"]["initial_generators"], h)
    mismatch = np.maximum.reduce([module.rowsum_upper(imat(matrix-I(A), flow)) for matrix in matrices])
    same(mismatch, step["attempts"][-1]["generator_mismatch_upper"], "generator mismatch")
    same((I(base)+I(mismatch)).hi, step["attempts"][-1]["forcing_upper"], "total robust forcing")
    for matrix, fields, tail_key in ((A, ("exponential_lower", "exponential_upper"), "exponential_tail"),
                                     (M, ("metzler_exponential_lower", "metzler_exponential_upper"), "metzler_tail")):
        E, Q, tail = module.exponential_and_integral(matrix, h)
        same(E.lo, step["affine"][fields[0]], fields[0])
        same(E.hi, step["affine"][fields[1]], fields[1])
        assert tail == step["affine"][tail_key]
        if tail_key == "metzler_tail":
            same(Q.lo, step["affine"]["integral_lower"], "integral lower")
            same(Q.hi, step["affine"]["integral_upper"], "integral upper")


def high_precision_saved_step(step, exact_h):
    with mp.workdps(90):
        h = mp.mpf(exact_h.numerator)/exact_h.denominator
        a = step["affine"]
        for key, lo_key, hi_key in (("frozen_A", "exponential_lower", "exponential_upper"),
                                    ("metzler_M", "metzler_exponential_lower", "metzler_exponential_upper")):
            matrix = mp.matrix(a[key])
            exp = mp.expm(matrix*h)
            for i in range(7):
                for j in range(7):
                    assert mp.mpf(a[lo_key][i][j]) <= exp[i,j] <= mp.mpf(a[hi_key][i][j])
            if key == "metzler_M":
                term = mp.eye(7)*h
                integral = term.copy()
                for k in range(1, 121):
                    term = matrix*term*h/(k+1)
                    integral += term
                for i in range(7):
                    for j in range(7):
                        assert mp.mpf(a["integral_lower"][i][j]) <= integral[i,j] <= mp.mpf(a["integral_upper"][i][j])
    return 147


def audit_proof(module, version, summary, row):
    proof_path = ROOT/row["proof_file"]
    assert sha(proof_path) == row["proof_sha256"]
    payload = json.loads(proof_path.read_text())
    assert payload["source_sha256"] == summary["source_sha256"]
    cert = payload["certificate"]
    desc = cert["proof_descriptor"]
    assert hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest() == cert["proof_id"]
    assert cert["adapter_audit"]["proof_hash_recomputed"]
    for key in ("accepted", "certified_until_h", "wall_seconds"):
        assert cert[key] == row[key]
    assert row["maximum_box_hull_radius"] == cert["maximum_metric_radius"]
    assert row["service_upper_vehicles"] == (100.*cert["service_check_results"][0]["upper"] if cert["service_check_results"] else None)
    commands = cert["adapter_audit"]["commands"]
    modes = [constant_command_mode(command, f"constant_block_{k}") for k, command in enumerate(commands)]
    verifier = module.AffineGeneratorVerifier()
    geometries = [verifier.geometry(mode) for mode in modes]
    assert desc["mode_fingerprints"] == [g["fingerprint"] for g in geometries]
    for contract, geo in zip(desc["affine_constraint_contracts"], geometries):
        same(contract["rows_lower"], geo["rows"].lo, "constraint row lower")
        same(contract["rows_upper"], geo["rows"].hi, "constraint row upper")
        same(contract["bounds_lower"], geo["bounds"].lo, "constraint bound lower")
        same(contract["bounds_upper"], geo["bounds"].hi, "constraint bound upper")
    pulse = Pulse(*map(Fraction, desc["pulse"]))
    grid = exact_grid(desc, pulse)
    assert len(grid) == 241
    center = np.asarray(desc["initial_state"])
    widths = np.asarray(cert["initial_box_units"])
    assert initial_supports(center, modes[0].feedforward, widths)["valid"]
    L = geometries[0]["L"]
    exact_G = imat(L, I(np.diag(widths)))
    G = exact_G.mid()
    rho = module.rowsum_upper(exact_G-I(G))
    signed = desc["signed_generator_method"]
    same(signed["initial"]["generators"], G, "initial G")
    same(signed["initial"]["remainder"], rho, "initial rho")
    if version == 3:
        assert signed["source_sha256"] == summary.get("method_source_sha256", summary["source_sha256"])
        assert all(summary["source_sha256"][name] == digest for name, digest in signed["source_sha256"].items())
        assert signed["protocol_sha256"] == summary["protocol_sha256"]
        structural = initial_positivity(center, widths, modes, desc["residual_bound"])
        assert structural == signed["structural_positivity_premises"] and structural["passed"]
        assert signed["exact_initial_physical_admissibility"] == initial_supports(center, modes[0].feedforward, widths)
        assert signed["generator_or_remainder_contraction"] is False
        queue = I(float(center[6]))+I(-widths[6], widths[6])
        queue_box = [max(0., float(queue.lo)), float(queue.hi)]
    steps = cert["interval_steps"]
    assert len(steps) == cert["completed_interval_steps"]
    selected = {0, len(steps)//2, len(steps)-1}
    selected_records, high_precision_entries = [], 0
    minimum_margins = np.full(len(geometries[0]["names"]), np.inf)
    last_safe = Fraction(desc["initial_time"])
    checkpoint_by_time = {item["exact_time_fraction"]: item for item in cert["reachable_checkpoints"]}
    for index, step in enumerate(steps):
        segment, left, right = grid[index]
        assert step["segment"] == segment and step["left_h"] == float(left) and step["right_h"] == float(right)
        geo, mode = geometries[segment], modes[segment]
        a, attempt = step["affine"], step["attempts"][-1]
        same(a["initial_generators"], G, "G chain")
        same(a["initial_remainder"], rho, "rho chain")
        assert np.all(rho >= 0.)
        same((I(module.rowsum_upper(G))+I(rho)).hi, step["initial_radius"], "initial hull")
        same(a["metzler_M"], attempt["robust_comparison_M"], "comparison M consistency")
        M = np.asarray(a["metzler_M"])
        assert np.all((M-np.diag(np.diag(M))) >= 0.)
        pred = predictor(center, mode, left, right, pulse, L)
        assert step["predictor_endpoint_rounding"] == norm_upper(I(pred["rounding"]))
        C = imat(geo["rows"], geo["T"])
        combined_rows = I(np.vstack((np.eye(7), C.lo)), np.vstack((np.eye(7), C.hi)))
        flow, flow_tail = module.signed_flow_range(combined_rows, a["frozen_A"], G, pred["h"])
        flow_support = module.rowsum_upper(flow)
        assert flow_tail == a["flow_tail"]
        remainder_flow, rem_tail = module.signed_flow_range(np.eye(7), M, np.diag(rho), pred["h"])
        rem_support = module.rowsum_upper(remainder_flow)
        assert rem_tail == a["remainder_flow_tail"]
        same(flow_support[:7], attempt["signed_generator_support"], "continuous signed support")
        same(rem_support, attempt["old_remainder_flow_support"], "continuous old remainder")
        Q = I(a["integral_lower"], a["integral_upper"])
        EM = I(a["metzler_exponential_lower"], a["metzler_exponential_upper"])
        assert np.all(Q.hi >= 0.) and np.all(EM.hi >= 0.)
        convolution = imat(I(Q.hi), I(attempt["forcing_upper"])).hi
        same(convolution, attempt["convolution_upper"], "convolution")
        continuous_remainder = (I(rem_support)+I(convolution)).hi
        continuous_error = (I(flow_support[:7])+I(continuous_remainder)).hi
        same(continuous_error, attempt["continuous_error_upper"], "continuous error")
        same(attempt["trial_radius"], step["apriori_radius"], "trial radius")
        assert np.all(continuous_error < np.asarray(step["apriori_radius"]))
        error_support = I(flow_support[7:])+imat(I(C.mag()), I(continuous_remainder))
        same(error_support.hi, a["continuous_affine_error_support_upper"], "affine physical support")
        row_curve = imat(geo["rows"], center)+imat(geo["rows"], pred["velocity"])*pred["tau"]+imat(geo["rows"], pred["acceleration"])*pred["tau"]*pred["tau"]
        margins = (geo["bounds"]-row_curve-error_support).lo.copy()
        exact = exact_command_constraints(commands[segment])
        assert exact == a["exact_constant_command"] and exact["passed"]
        for name, value in exact["exact_margins"].items():
            margins[geo["names"].index(name)] = rational_lower(value)
        if version == 3:
            physics = step["independent_physics"]
            same(queue_box, physics["initial_queue"], "scalar queue chain")
            queue_record = queue_step_bounds(*queue_box, right-left, pred["arrival0"], pred["slope"], desc["residual_bound"], mode.feedforward[2])
            assert queue_record == physics["autonomous_queue"]
            queue_box = queue_record["endpoint"]
            same([min(float(pred["curve_box"].lo[6]), queue_record["continuous"][0]),
                  max(float(pred["curve_box"].hi[6]), queue_record["continuous"][1])], physics["mean_value_queue_hull"], "mean-value queue hull")
            same(margins, physics["raw_affine_margins_lower"], "raw physical margins")
            assert physics["positive_rows"] == structural["lower_rows_certified"]
            assert physics["generator_or_remainder_contracted"] is False
            for name in structural["lower_rows_certified"]:
                i = geo["names"].index(name)
                margins[i] = max(0., margins[i])
            i, j = geo["names"].index("exterior_queue_lower"), geo["names"].index("exterior_queue_upper")
            margins[i] = max(margins[i], queue_record["continuous"][0])
            margins[j] = max(margins[j], float((I(model.QUEUE_LIMIT)-I(queue_record["continuous"][1])).lo))
            assert step["log_norm_upper"] == float(max(isum(I(M[:, j]) for j in range(7)).hi))
        same(margins, step["continuous_constraint_margins_lower"], "installed physical margins")
        E = I(a["exponential_lower"], a["exponential_upper"])
        endpoint_G = imat(E, G)
        G_next = endpoint_G.mid()
        G_round = module.rowsum_upper(endpoint_G-I(G_next))
        rho_next = (I(G_round)+imat(I(EM.hi), I(rho))+I(convolution)+I(pred["rounding"])).hi
        same(G_next, a["endpoint_generators"], "endpoint G")
        same(rho_next, a["endpoint_remainder"], "endpoint rho")
        radius = (I(module.rowsum_upper(G_next))+I(rho_next)).hi
        same(radius, step["radius"], "endpoint outer hull")
        if index in selected:
            selected_corridor_check(module, version, center, mode, geo, pred, step, desc["residual_bound"])
            high_precision_entries += high_precision_saved_step(step, right-left)
            selected_records.append(dict(index=index, exact_left=str(left), exact_right=str(right), full_corridor_forcing_matrix_check=True))
        center, G, rho = pred["next_center"], G_next, rho_next
        if str(right) in checkpoint_by_time:
            checkpoint = checkpoint_by_time[str(right)]
            same(center, checkpoint["center"], "checkpoint center")
            same(radius, checkpoint["radius"], "checkpoint hull")
        minimum_margins = np.minimum(minimum_margins, margins)
        if np.min(margins) >= 0:
            last_safe = right
        else:
            assert index == len(steps)-1 and not cert["verified_continuous_physical"]
            failure = cert["failures"][0]
            assert failure["constraint"] == geo["names"][int(np.argmin(margins))]
            assert failure["margin_lower"] == float(min(margins))
    same(G, cert["affine_terminal_generators"], "terminal G")
    same(rho, cert["affine_terminal_remainder"], "terminal rho")
    same(center, cert["terminal_shape"]["center"], "terminal center")
    assert cert["certified_until_h"] == float(last_safe)
    same(minimum_margins, list(cert["continuous_constraint_margins_lower"].values()), "aggregate physical margins")
    for checkpoint in cert["reachable_checkpoints"]:
        assert checkpoint["proof_id"] == cert["proof_id"]
        if version == 3:
            assert checkpoint["outer_hull_only"] and checkpoint["restart_authorized"] is False
    if version == 3:
        for check in signed["final_signed_service_support"]:
            row_interval = I(geometries[-1]["T"].lo[check["coordinate"]:check["coordinate"]+1], geometries[-1]["T"].hi[check["coordinate"]:check["coordinate"]+1])
            gs = module.rowsum_upper(imat(row_interval, G))[0]
            rs = imat(I(row_interval.mag()), I(rho)).hi[0]
            upper = float((I(center[check["coordinate"]])+I(gs)+I(rs)).hi)
            assert gs == check["generator_support"] and rs == check["remainder_support"] and upper == check["signed_generator_upper"]
            assert check["installed_upper"] == min(check["old_box_hull_upper"], upper)
            matching = [s for s in desc["service_checks"] if Fraction(s["exact_time"]) == Fraction(cert["terminal_shape"]["exact_time_fraction"])
                        and s["coordinate"] == check["coordinate"]]
            assert len(matching) == 1
            assert check["passed"] == (check["installed_upper"] <= matching[0]["upper"])
    assert cert["accepted"] == (cert["verified_continuous_physical"] and cert["verified_terminal_service"] and cert["verified_scheduled_service"])
    if cert["accepted"]:
        assert len(steps) == len(grid) and cert["certified_until_h"] == 1.
    return dict(version=version, row_id=row["row_id"], passed=True, proof_file=row["proof_file"],
                proof_sha256=row["proof_sha256"], proof_id=cert["proof_id"], interval_steps=len(steps),
                accepted=cert["accepted"], exact_full_grid_steps=len(grid), selected_full_arithmetic=selected_records,
                reference_90digit_entry_checks=high_precision_entries, queue_steps=len(steps) if version == 3 else 0)


def run():
    started = time.perf_counter()
    result = dict(status="RUNNING", passed=False, audit_source_sha256=sha(Path(__file__)),
        scope="Saved proof arithmetic/provenance audit; no optimizer, ODE trajectories, live calls or new scientific proofs.",
        independent_backend=False, all_steps="Exact clocks; G/rho chain; signed continuous support; strict self-inclusion; physical margins; endpoint/hull; v3 scalar queue sequence.",
        selected_steps="First/middle/last per proof reconstruct corridor/Jacobian/defect/forcing and endpoint matrix enclosures; 90-digit matrix point diagnostics.",
        preserved_metadata_notes=["v2 log_norm_upper is the frozen-A infinity norm, not an actual-J logarithmic norm; it was not used for admission. v3 corrects this field.",
                                  "v2 method-source hash is in the payload, not descriptor; v3 binds source/protocol hashes in the descriptor.",
                                  "v2 checkpoint boxes are outer hulls only; arbitrary restart/suffix membership authorization is invalid. v3 explicitly labels this restriction."],
        source_audits=[], primitive_reviews=[], proofs=[])
    def save():
        OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    save()
    for version in (2, 3):
        module, module_path = target(version)
        summary_path = RESULTS/f"mfd_incident_affine_reachability_v{version}_results.json"
        summary = json.loads(summary_path.read_text())
        for name, expected in summary["source_sha256"].items():
            path = ROOT/name
            if name == "reproducibility/mfd_incident_affine_reachability.py":
                path = module_path
            assert sha(path) == expected, (version, name, "source hash drift")
            snapshot = summary.get("source_snapshots", {}).get(name)
            if snapshot:
                assert sha(ROOT/snapshot) == expected
            result["source_audits"].append(dict(version=version, source=name, resolved_path=str(path), sha256=expected, snapshot=snapshot, passed=True))
        if version == 3:
            assert sha(ROOT/summary["protocol_path"]) == summary["protocol_sha256"]
            assert json.loads((ROOT/summary["protocol_path"]).read_text()) == summary["protocol"]
            assert len(summary["rows"]) == len(summary["execution_order"]) == 17
        else:
            assert len(summary["rows"]) == 15
        result["primitive_reviews"].append(matrix_primitive_review(module_path))
        for row in summary["rows"]:
            record = audit_proof(module, version, summary, row)
            result["proofs"].append(record)
            save()
            print(json.dumps({k: record[k] for k in ("version", "row_id", "passed", "interval_steps")}), flush=True)
    result.update(status="PASS", passed=True, wall_seconds=time.perf_counter()-started,
                  proof_count=len(result["proofs"]), interval_step_count=sum(r["interval_steps"] for r in result["proofs"]),
                  queue_step_count=sum(r["queue_steps"] for r in result["proofs"]),
                  selected_corridor_step_count=sum(len(r["selected_full_arithmetic"]) for r in result["proofs"]),
                  saved_matrix_90digit_entry_checks=sum(r["reference_90digit_entry_checks"] for r in result["proofs"]))
    save()
    print(json.dumps({k:result[k] for k in ("status", "proof_count", "interval_step_count", "queue_step_count", "wall_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    run()
