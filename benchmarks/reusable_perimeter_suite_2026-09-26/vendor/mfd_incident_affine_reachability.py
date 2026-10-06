"""Signed-generator finite-plan reachability prototype (separate from baseline).

Derivation v2. In fixed coordinates y=L(x-p), use y=exp(At)G*xi+eta,
|xi_i|<=1, |eta(0)|<=rho. On the trial tube |y_i|<R_i every generalized
Jacobian belongs to interval J. For ANY chosen binary matrix A,
 eta'=J(t)eta+(J(t)-A)exp(At)G*xi+predictor_defect+Le7*w.
Let M_ii=sup J_ii and M_ij=sup |J_ij| off-diagonal, over both min branches.
Bound b=max_branch row_sum_mag((J_branch-A)*range(exp(At)G))+defect+|Le7|wbar.
The componentwise comparison gives |eta(t)|<=exp(Mt)rho+Q(t)b, where
Q(t)=integral_0^t exp(Ms)ds is nonnegative and increasing componentwise.
This retains negative diagonal uncertainty in the remainder dynamics.
The original generator propagates as exp(At)G. Signed matrix power series
evaluate C exp(At) G BEFORE absolute values, including a norm-verified tail.
Strict inclusion of these componentwise supports in R gives the first-exit
proof. Every physical affine support uses the same signed row propagation.
At the endpoint, round exp(Ah)G to a binary generator and put its rounding
error, propagated old rho, nonlinear/process forcing and predictor rounding
into the new rho. No generator reduction or unverified exponential is used.

V3 composition adds independently proved compartment positivity and scalar queue
bounds. They discharge only their physical supports. For branch/Jacobian selection
the old corridor is intersected with the convex hull of the predictor queue and
the actual scalar queue enclosure, so every mean-value segment remains enclosed.
The signed generator and remainder are NEVER contracted by these auxiliary facts.
Final-time service support may use the tighter signed-generator expression.
Checkpoint boxes are outer hulls only, not restart/continuation authorizations.

Scope: unchanged first archived MPC plan; explicit admissible physical initial
box; fixed conservation coordinates; arbitrary measurable bounded arrival
residual; no numerical ODE tolerance as certificate. A separate immutable JSON
protocol is written before any new proof, and its hash and all method-module
hashes are bound in every v3 proof descriptor. Oldv1/v2 outcomes are preserved.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil

from mpmath import iv
import numpy as np

import mfd_incident_model as model
from mfd_incident_mpc_certification import ConstantCommandVerifier, exact_command_constraints, rational_lower, verify_command_plan
from mfd_incident_verifier import (DIM, EnclosureFailure, I, J, Pulse, branch_for_box,
    closed_field, evaluate_transformed_jacobian, imat, interval_fraction, jacobian,
    norm_upper)
from mfd_deadline_certificates import isum
from mfd_incident_positive_reachability import initial_positivity
from mfd_incident_queue_bounds import queue_step_bounds
from mfd_incident_uncertainty_frontier import initial_supports

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PREFIX = "mfd_incident_affine_reachability_v3"
ORDER = 18
PROTOCOL_PATH = RESULTS / "mfd_affine_v3_protocol.json"


def upward(value):
    return float(np.nextafter(float(value), np.inf))


def downward(value):
    return float(np.nextafter(float(value), -np.inf))


def rowsum_upper(matrix):
    matrix = I(matrix)
    return isum(I(matrix.mag()[:, j]) for j in range(matrix.lo.shape[1])).hi


def norm_inf_upper(matrix):
    return float(max(rowsum_upper(matrix)))


def reciprocal(number):
    value = iv.mpf(1) / number
    return I(downward(value.a), upward(value.b))


def exponential_and_integral(matrix, h, order=ORDER):
    """Verified endpoint exp(Ah), integral exp(At)dt; arbitrary real matrix."""
    matrix = I(matrix)
    term = I(np.eye(DIM))
    exponential = term
    integral = term * h
    for k in range(1, order+1):
        term = imat(matrix, term) * h * reciprocal(k)
        exponential = exponential + term
        integral = integral + term * h * reciprocal(k+1)
    z = iv.mpf(norm_inf_upper(matrix)) * iv.mpf([float(h.lo), float(h.hi)])
    exp_tail = upward((iv.exp(z)*z**(order+1)/math.factorial(order+1)).b)
    int_tail = upward((iv.mpf(float(h.hi))*iv.exp(z)*z**(order+1)/math.factorial(order+2)).b)
    return exponential + I(-exp_tail, exp_tail), integral + I(-int_tail, int_tail), dict(
        order=order, induced_infinity_norm_upper=norm_inf_upper(matrix),
        exponential_tail_infinity_bound=exp_tail, integral_tail_infinity_bound=int_tail)


def signed_flow_range(rows, matrix, generators, h, order=ORDER):
    """Interval enclosure of C exp(At) G for all0<=t<=h.

    Terms C A^k G/k! retain signed cancellations before the one-dimensional
    time interval is applied. Tail entry ij is bounded by
    ||C_i||1 * ||exponential remainder||inf * ||G_:j||inf.
    """
    rows, matrix, generators = I(rows), I(matrix), I(generators)
    term = generators
    result = imat(rows, term)
    for k in range(1, order+1):
        term = imat(matrix, term) * reciprocal(k)
        hp = upward((iv.mpf(float(h.hi))**k).b)
        result = result + imat(rows, term) * I(0., hp)
    norm = norm_inf_upper(matrix)
    z = iv.mpf(norm)*iv.mpf(float(h.hi))
    tail = upward((iv.exp(z)*z**(order+1)/math.factorial(order+1)).b)
    coefficient = I(rowsum_upper(rows)[:, None]) * I(np.max(generators.mag(), axis=0)[None, :]) * I(tail)
    return result + I(-coefficient.hi, coefficient.hi), dict(
        order=order, matrix_tail_infinity_bound=tail,
        maximum_transformed_tail_entry=float(np.max(coefficient.hi)))


class AffineGeneratorVerifier(ConstantCommandVerifier):
    """Preserves the initial generator across every unchanged-coordinate step."""

    def verify_plan(self, initial_state, modes, durations_h=None, **kwargs):
        if kwargs.get("initial_shape") is not None:
            raise ValueError("Prototype accepts a physical initial box only")
        selected = [self.library[int(mode)] if isinstance(mode, (int, np.integer)) else mode for mode in modes]
        geo = self.geometry(selected[0])
        widths = np.broadcast_to(np.asarray(kwargs.get("initial_box", .01)), (DIM,))
        if kwargs.get("amplitude", .3) < 0:
            raise ValueError("Nonnegative incident amplitude required for positivity lower-arrival premise")
        self.initial_physical = initial_supports(initial_state, selected[0].feedforward, widths)
        if not self.initial_physical["valid"]:
            raise ValueError("Entire explicit initial physical information box must be admissible")
        self.structural = initial_positivity(np.asarray(initial_state), widths, selected, kwargs.get("residual_bound", .6))
        if not self.structural["passed"]:
            raise ValueError(("Independent positivity premises failed", self.structural))
        queue_initial = I(float(initial_state[6]))+I(-widths[6], widths[6])
        self.queue_box = [max(0., float(queue_initial.lo)), float(queue_initial.hi)]
        self.start_source_hashes = source_hashes()
        exact_generators = imat(geo["L"], I(np.diag(widths)))
        self.affine_generators = exact_generators.mid()
        self.affine_remainder = rowsum_upper(exact_generators-I(self.affine_generators))
        self.affine_L = geo["L"].copy()
        self.affine_initial = dict(generators=self.affine_generators.tolist(), remainder=self.affine_remainder.tolist())
        result = super().verify_plan(initial_state, selected, durations_h, **kwargs)
        if source_hashes() != self.start_source_hashes:
            raise RuntimeError("Method/protocol source drift during proof")
        result["pre_affine_proof_id"] = result["proof_id"]
        final_service = []
        if result["verified_continuous_physical"]:
            terminal = result["terminal_shape"]
            final_geo = self.geometry(selected[-1])
            for check in result["service_check_results"]:
                if check["time_h"] != terminal["time_h"]:
                    continue
                index = check["coordinate"]
                row = I(final_geo["T"].lo[index:index+1], final_geo["T"].hi[index:index+1])
                generator_support = rowsum_upper(imat(row, self.affine_generators))[0]
                remainder_support = imat(I(row.mag()), I(self.affine_remainder)).hi[0]
                signed_upper = float((I(terminal["center"][index])+I(generator_support)+I(remainder_support)).hi)
                old_upper = check["upper"]
                check["upper"] = min(old_upper, signed_upper)
                check["passed"] = bool(check["upper"] <= check["limit"])
                final_service.append(dict(time_h=check["time_h"], coordinate=index,
                    old_box_hull_upper=old_upper, signed_generator_upper=signed_upper,
                    generator_support=generator_support, remainder_support=float(remainder_support),
                    installed_upper=check["upper"], passed=check["passed"]))
            result["verified_scheduled_service"] = all(check["passed"] for check in result["service_check_results"])
            result["failures"] = [f for f in result["failures"] if f["kind"] != "scheduled_service_not_certified"]
            if not result["verified_scheduled_service"]:
                result["failures"].append(dict(kind="scheduled_service_not_certified", checks=result["service_check_results"]))
            result["accepted"] = result["verified"] = bool(result["verified_terminal_service"] and result["verified_scheduled_service"])
        result["proof_descriptor"]["signed_generator_method"] = dict(
            rule="signed_initial_generator_plus_robust_variational_remainder_positive_queue_v3", series_order=ORDER,
            source_sha256=self.start_source_hashes, protocol_sha256=digest(PROTOCOL_PATH),
            initial=self.affine_initial, generator_reduction=False,
            exact_initial_physical_admissibility=self.initial_physical,
            structural_positivity_premises=self.structural,
            self_inclusion="strict componentwise signed-propagator support in trial tube",
            endpoint="verified matrix exponential and Metzler integral with norm-tail bounds",
            independent_queue_use="physical queue supports and intersection of old Jacobian corridor with hull(predictor queue, scalar actual queue)",
            generator_or_remainder_contraction=False, final_signed_service_support=final_service,
            checkpoint_semantics="outer_hull_only; membership does not authorize arbitrary restart or suffix recertification")
        result["proof_id"] = hashlib.sha256(json.dumps(result["proof_descriptor"], sort_keys=True).encode()).hexdigest()
        for checkpoint in result["reachable_checkpoints"]:
            checkpoint["proof_id"] = result["proof_id"]
            checkpoint["outer_hull_only"] = True
            checkpoint["restart_authorized"] = False
        result["affine_terminal_generators"] = self.affine_generators.tolist()
        result["affine_terminal_remainder"] = self.affine_remainder.tolist()
        result["scope_note"] = "Checkpoint boxes are outer hulls of the signed-generator sets; they may be used only as outer enclosures. No initial-shape continuation API is supplied by this prototype."
        return result

    def step(self, center, radius, mode, left, right, pulse, residual_bound):
        if not np.array_equal(mode.gain, np.zeros((4, DIM))):
            raise ValueError("Prototype restricted to constant command plans")
        geo = self.geometry(mode)
        L, inverse = geo["L"], geo["T"]
        if not np.array_equal(L, self.affine_L):
            raise ValueError("Prototype uses one fixed coordinate transform")
        G, rho = self.affine_generators, self.affine_remainder
        h = interval_fraction(right-left)
        tau = I(0., h.hi)
        arrival0, arrival_slope = pulse.affine(left, right)
        queue_initial = self.queue_box.copy()
        queue_record = queue_step_bounds(*queue_initial, right-left, arrival0, arrival_slope,
                                         residual_bound, mode.feedforward[2])
        center_branch, _ = branch_for_box(I(center), mode)
        chosen = "gate" if center_branch == "both" else center_branch
        values = closed_field([I(value) for value in center], mode, arrival0, chosen)
        fbox = I(np.array([v.lo for v in values]), np.array([v.hi for v in values]))
        velocity = fbox.mid()
        acceleration_box = imat(jacobian(I(center), mode, chosen), fbox)
        slope_row = np.zeros(DIM); slope_row[6] = 1.
        acceleration = ((acceleration_box + I(slope_row)*arrival_slope)*I(.5)).mid()
        curve = [J([center[i], velocity[i], acceleration[i]]) for i in range(DIM)]
        curve_box = I(center)+I(velocity)*tau+I(acceleration)*tau*tau
        queue_hull = I(min(float(curve_box.lo[6]), queue_record["continuous"][0]),
                       max(float(curve_box.hi[6]), queue_record["continuous"][1]))
        curve_branch, _ = branch_for_box(curve_box, mode)
        defect_vectors = []
        for branch in (("gate", "supply") if curve_branch == "both" else (curve_branch,)):
            predicted = closed_field(curve, mode, J([arrival0, arrival_slope]), branch)
            residuals = [predicted[i]-J([velocity[i], I(2.)*I(acceleration[i])]) for i in range(DIM)]
            degree = max(len(item.c) for item in residuals)
            assert degree <= 9
            transformed_residual, power = I(np.zeros(DIM)), I(1.)
            for index in range(degree):
                coefficient = I(np.array([item.c[index].lo if index < len(item.c) else 0. for item in residuals]),
                                np.array([item.c[index].hi if index < len(item.c) else 0. for item in residuals]))
                transformed = imat(L, coefficient)
                transformed_residual = transformed_residual + I(-transformed.mag(), transformed.mag())*power
                power = power*I(h.hi)
            defect_vectors.append(transformed_residual.mag())
        defect = np.maximum.reduce(defect_vectors)
        base_forcing = (I(defect)+I(abs(L[:, 6]))*I(residual_bound)).hi
        endpoint = I(center)+I(velocity)*h+I(acceleration)*h*h
        next_center = endpoint.mid()
        rounding = imat(L, endpoint-I(next_center)).mag()
        metric_point = imat(L, I(center)-I(mode.center))
        A = evaluate_transformed_jacobian(geo["jacobian_record"], metric_point, chosen, centered="exact").mid()
        E, _, exp_record = exponential_and_integral(A, h)
        C = imat(geo["rows"], inverse)
        combined_rows = I(np.vstack((np.eye(DIM), C.lo)), np.vstack((np.eye(DIM), C.hi)))
        flow, flow_record = signed_flow_range(combined_rows, A, G, h)
        flow_support = rowsum_upper(flow)
        r0 = (I(rowsum_upper(G))+I(rho)).hi
        trial = (I(1.1)*(I(np.maximum(r0, flow_support[:DIM]))+I(h.hi)*I(base_forcing)+I(1e-12))).hi
        metric_curve = imat(L, I(center)-I(mode.center))+imat(L, velocity)*tau+imat(L, acceleration)*tau*tau
        attempts = []
        for iteration in range(24):
            physical_support = imat(I(inverse.mag()), I(trial)).hi
            physical_box = curve_box+I(-physical_support, physical_support)
            physical_box.lo[6] = max(float(physical_box.lo[6]), float(queue_hull.lo))
            physical_box.hi[6] = min(float(physical_box.hi[6]), float(queue_hull.hi))
            assert physical_box.lo[6] <= physical_box.hi[6]
            branch, branch_gap = branch_for_box(physical_box, mode)
            metric_box = metric_curve+I(-trial, trial)
            assert np.array_equal(L[6], np.eye(DIM)[6])
            metric_queue = queue_hull-I(mode.center[6])
            metric_box.lo[6] = max(float(metric_box.lo[6]), float(metric_queue.lo))
            metric_box.hi[6] = min(float(metric_box.hi[6]), float(metric_queue.hi))
            assert metric_box.lo[6] <= metric_box.hi[6]
            matrices = [evaluate_transformed_jacobian(geo["jacobian_record"], metric_box, smooth, centered="exact")
                        for smooth in (("gate", "supply") if branch == "both" else (branch,))]
            M = np.maximum.reduce([matrix.mag() for matrix in matrices])
            diagonal = np.maximum.reduce([np.diag(matrix.hi) for matrix in matrices])
            np.fill_diagonal(M, diagonal)
            EM, Q, metzler_record = exponential_and_integral(M, h)
            assert np.all(Q.hi >= 0.) and np.all(EM.hi >= 0.)
            remainder_flow, remainder_flow_record = signed_flow_range(np.eye(DIM), M, np.diag(rho), h)
            remainder_flow_support = rowsum_upper(remainder_flow)
            generator_mismatch = np.maximum.reduce([rowsum_upper(imat(matrix-I(A), flow[:DIM])) for matrix in matrices])
            forcing = (I(generator_mismatch)+I(base_forcing)).hi
            convolution = imat(I(Q.hi), I(forcing)).hi
            continuous_remainder = (I(remainder_flow_support)+I(convolution)).hi
            continuous_radius = (I(flow_support[:DIM])+I(continuous_remainder)).hi
            attempts.append(dict(trial_radius=trial.tolist(), signed_generator_support=flow_support[:DIM].tolist(),
                                 robust_comparison_M=M.tolist(), generator_mismatch_upper=generator_mismatch.tolist(),
                                 old_remainder_flow_support=remainder_flow_support.tolist(), forcing_upper=forcing.tolist(),
                                 convolution_upper=convolution.tolist(), continuous_error_upper=continuous_radius.tolist(),
                                 branch=branch, branch_gap=[float(branch_gap.lo), float(branch_gap.hi)]))
            if np.all(continuous_radius < trial):
                # Preserve predictor row cancellations, and propagate signed generator supports.
                row_curve = imat(geo["rows"], center)+imat(geo["rows"], velocity)*tau+imat(geo["rows"], acceleration)*tau*tau
                error_support = I(flow_support[DIM:])+imat(I(C.mag()), I(continuous_remainder))
                margins = geo["bounds"]-row_curve-error_support
                margins_lower = margins.lo.copy()
                exact = exact_command_constraints(mode.feedforward)
                for name, value in exact["exact_margins"].items():
                    margins_lower[geo["names"].index(name)] = rational_lower(value)
                raw_margins = margins_lower.copy()
                for name in self.structural["lower_rows_certified"]:
                    index = geo["names"].index(name)
                    margins_lower[index] = max(0., margins_lower[index])
                queue_lo = geo["names"].index("exterior_queue_lower")
                queue_hi = geo["names"].index("exterior_queue_upper")
                margins_lower[queue_lo] = max(margins_lower[queue_lo], queue_record["continuous"][0])
                margins_lower[queue_hi] = max(margins_lower[queue_hi], float((I(model.QUEUE_LIMIT)-I(queue_record["continuous"][1])).lo))
                endpoint_generators = imat(E, G)
                new_G = endpoint_generators.mid()
                G_rounding = rowsum_upper(endpoint_generators-I(new_G))
                new_rho = (I(G_rounding)+imat(I(EM.hi), I(rho))+I(convolution)+I(rounding)).hi
                next_radius = (I(rowsum_upper(new_G))+I(new_rho)).hi
                self.affine_generators, self.affine_remainder = new_G, new_rho
                self.queue_box = queue_record["endpoint"]
                actual_log_norm = float(max(isum(I(M[:, j]) for j in range(DIM)).hi))
                return next_center, next_radius, dict(
                    left_h=float(left), right_h=float(right), defect_upper=float(max(defect)),
                    initial_radius=r0.tolist(), radius=next_radius.tolist(), apriori_radius=trial.tolist(),
                    predictor_endpoint_rounding=norm_upper(I(rounding)), log_norm_upper=actual_log_norm,
                    frozen_A_induced_infinity_norm_upper=norm_inf_upper(A),
                    comparison_method="signed-generator propagation plus robust variational Metzler remainder",
                    active_branch=branch, branch_gap_interval=[float(branch_gap.lo), float(branch_gap.hi)],
                    continuous_constraint_margins_lower=margins_lower.tolist(), attempts=attempts,
                    independent_physics=dict(initial_queue=queue_initial, autonomous_queue=queue_record,
                        mean_value_queue_hull=[float(queue_hull.lo), float(queue_hull.hi)],
                        raw_affine_margins_lower=raw_margins.tolist(),
                        positive_rows=self.structural["lower_rows_certified"],
                        generator_or_remainder_contracted=False),
                    affine=dict(initial_generators=G.tolist(), initial_remainder=rho.tolist(), frozen_A=A.tolist(),
                                metzler_M=M.tolist(), exponential_lower=E.lo.tolist(), exponential_upper=E.hi.tolist(),
                                metzler_exponential_lower=EM.lo.tolist(), metzler_exponential_upper=EM.hi.tolist(),
                                integral_lower=Q.lo.tolist(), integral_upper=Q.hi.tolist(),
                                exponential_tail=exp_record, metzler_tail=metzler_record, flow_tail=flow_record,
                                remainder_flow_tail=remainder_flow_record,
                                endpoint_generators=new_G.tolist(), endpoint_remainder=new_rho.tolist(),
                                continuous_affine_error_support_upper=error_support.hi.tolist(),
                                exact_constant_command=exact))
            trial = np.maximum((I(trial)*I(1.1)).hi,
                               (I(continuous_radius)*I(1.1)+I(1e-12)).hi)
            if not np.all(np.isfinite(trial)) or max(trial) > 100.:
                break
        raise EnclosureFailure("signed_generator_self_inclusion_failed", dict(
            left_h=float(left), right_h=float(right), attempts=attempts, frozen_A=A.tolist(),
            exponential_tail=exp_record, metzler_tail=metzler_record, flow_tail=flow_record,
            independent_queue=queue_record, queue_initial=queue_initial,
            mean_value_queue_hull=[float(queue_hull.lo), float(queue_hull.hi)]))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_protocol():
    record = dict(method="signed-generator robust-variational enclosure with independent positive/queue supports v3",
        algorithm="G'=A G; eta'=J eta+(J-A)G xi+defect+Le7w; robust Metzler comparison for eta",
        independent_physics="verified initial positivity; scalar autonomous queue bounds; no G/rho contraction",
        mean_value_domain="old corridor intersect convex hull of nominal predictor queue and all actual scalar queues",
        final_service="minimum of inherited box-hull and signed-generator endpoint affine supports",
        checkpoints="outer_hull_only; no restart or arbitrary suffix authorization",
        step_seconds=15., series_order=ORDER, queue_width_rule="min(state width,10) vehicles",
        first_tests=[[100., 60.], [50., 300.], [10., 600.], [1., 1200.]],
        followup_grid=[[w, r] for r in (60., 120., 300.) for w in (1., 10., 25., 50., 100.)],
        horizon_h=1., service_limit_vehicles=2850., commands="archived first primary MPC plan unchanged",
        source_record="mfd_incident_mpc_grid_results.json", all_outcomes_retained=True,
        prior_versions="v1 additive-diagonal failure and v2 complete15grid retained with executed source snapshots")
    if PROTOCOL_PATH.exists():
        assert json.loads(PROTOCOL_PATH.read_text()) == record, "Frozen v3 protocol mismatch"
    else:
        PROTOCOL_PATH.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return record


def source_hashes():
    names = ["reproducibility/mfd_incident_affine_reachability.py", "reproducibility/mfd_incident_model.py",
             "reproducibility/mfd_cubic_pilot.py", "reproducibility/mfd_incident_verifier.py",
             "reproducibility/mfd_incident_mpc_certification.py", "reproducibility/mfd_deadline_certificates.py",
             "reproducibility/mfd_incident_positive_reachability.py", "POSITIVE_REACHABILITY_PROTOCOL.md",
             "reproducibility/mfd_incident_queue_bounds.py", "reproducibility/mfd_incident_uncertainty_frontier.py",
             PROTOCOL_PATH.relative_to(ROOT).as_posix(), "reproducibility/results/mfd_incident_mpc_grid_results.json"]
    return {name: digest(ROOT/name) for name in names}


def run(limit=None):
    protocol = frozen_protocol()
    archive_path = RESULTS/"mfd_incident_mpc_grid_results.json"
    archive = json.loads(archive_path.read_text())
    first = next(r for r in archive["runs"] if r["amplitude"] == .3 and r["duration_h"] == 1.)["decisions"][0]
    hashes = source_hashes()
    snapshots = {}
    for name, sha in hashes.items():
        destination = RESULTS/"ars"/sha[:16]/Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(ROOT/name, destination)
        assert digest(destination) == sha
        snapshots[name] = destination.relative_to(ROOT).as_posix()
    output = RESULTS/f"{PREFIX}_results.json"
    if output.exists():
        payload = json.loads(output.read_text())
        assert payload["source_sha256"] == hashes, "Source changed; preserve old outputs and choose a new revision prefix"
    else:
        full_grid = protocol["first_tests"]+[pair for pair in protocol["followup_grid"] if pair not in protocol["first_tests"]]
        payload = dict(source_sha256=hashes, source_snapshots=snapshots,
                       protocol_path=PROTOCOL_PATH.relative_to(ROOT).as_posix(), protocol_sha256=digest(PROTOCOL_PATH),
                       protocol=protocol, execution_order=full_grid, rows=[])
        output.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    attempted = 0
    for width, residual in payload["execution_order"]:
        key = f"v{width:g}_r{residual:g}"
        if any(r["row_id"] == key for r in payload["rows"]):
            continue
        if limit is not None and attempted >= limit:
            break
        widths = np.array([width]*6+[min(width, 10.)])/100.
        verifier = AffineGeneratorVerifier()
        cert = verify_command_plan(first["initial_state"], first["controls"],
                                   [Fraction(1, 12), Fraction(11, 12)], initial_box=widths,
                                   residual_bound=residual/100., verifier=verifier, retain_steps=True)
        proof = RESULTS/f"{PREFIX}_{key}_proof.json"
        proof.write_text(json.dumps(dict(source_sha256=hashes, certificate=cert), indent=2, allow_nan=False)+"\n", encoding="utf-8")
        failure = cert["failures"][0] if cert["failures"] else None
        row = dict(row_id=key, state_half_width_vehicles=width, residual_vehicles_per_hour=residual,
                   accepted=cert["accepted"], certified_until_h=cert["certified_until_h"],
                   first_failure=None if failure is None else {k: v for k, v in failure.items() if k not in ("failed_step", "attempts")},
                   service_upper_vehicles=100.*cert["service_check_results"][0]["upper"] if cert["service_check_results"] else None,
                   wall_seconds=cert["wall_seconds"], maximum_box_hull_radius=cert["maximum_metric_radius"],
                   proof_file=proof.relative_to(ROOT).as_posix(), proof_sha256=digest(proof))
        payload["rows"].append(row)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
        attempted += 1
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.limit)
