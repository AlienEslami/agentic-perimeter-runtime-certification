"""Frozen connected-ring transfer test; dynamic dimension, validated intervals.

No imported DIM, controller, or model global is changed. Read the separately
frozen MULTIREGION_TRANSFER_PROTOCOL.md for the model, grid and proof scope.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from mpmath import iv, mp
from scipy.integrate import solve_ivp

from mfd_deadline_certificates import I, J, imat, isum
from mfd_incident_queue_bounds import queue_step_bounds

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PROTOCOL = ROOT / "MULTIREGION_TRANSFER_PROTOCOL.md"
PREFIX = "mfd_multiregion_transfer"
OUTPUT = RESULTS / f"{PREFIX}_results.json"
ORDER = 18
NOMINAL_STEP = Fraction(1, 120)
MIN_STEP = Fraction(1, 1920)
MFD_A, MFD_B, MFD_C = .0014877, -.29815, 15.0912


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, record):
    """Scientific outputs are generated data, not hand-authored source files."""
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


def sources():
    files = [Path(__file__), PROTOCOL, ROOT / "reproducibility/mfd_deadline_certificates.py",
             ROOT / "reproducibility/mfd_incident_queue_bounds.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def upper(value):
    return float(np.nextafter(float(value), np.inf))


def lower(value):
    return float(np.nextafter(float(value), -np.inf))


def rat(value):
    value = value if isinstance(value, Fraction) else Fraction.from_float(float(value))
    near = float(value)
    binary = Fraction.from_float(near)
    return I(np.nextafter(near, -np.inf) if binary > value else near,
             np.nextafter(near, np.inf) if binary < value else near)


def rowsum(matrix):
    matrix = I(matrix)
    return isum(I(matrix.mag()[:, j]) for j in range(matrix.lo.shape[1])).hi


def exp_integral(matrix, h):
    matrix, h = I(matrix), I(h)
    n = matrix.lo.shape[0]
    term = I(np.eye(n)); exponential = term; integral = term*h
    for k in range(1, ORDER+1):
        term = imat(matrix, term)*h*rat(Fraction(1, k))
        exponential = exponential+term
        integral = integral+term*h*rat(Fraction(1, k+1))
    norm = float(max(rowsum(matrix)))
    z = iv.mpf(norm)*iv.mpf([float(h.lo), float(h.hi)])
    et = upper((iv.exp(z)*z**(ORDER+1)/math.factorial(ORDER+1)).b)
    qt = upper((iv.mpf(float(h.hi))*iv.exp(z)*z**(ORDER+1)/math.factorial(ORDER+2)).b)
    return exponential+I(-et, et), integral+I(-qt, qt), dict(order=ORDER, norm_inf_upper=norm,
        exponential_tail=et, integral_tail=qt)


def signed_range(rows, matrix, generators, h):
    rows, matrix, generators = I(rows), I(matrix), I(generators)
    term = generators; result = imat(rows, term)
    for k in range(1, ORDER+1):
        term = imat(matrix, term)*rat(Fraction(1, k))
        power = upper((iv.mpf(float(h.hi))**k).b)
        result = result+imat(rows, term)*I(0., power)
    norm = float(max(rowsum(matrix)))
    z = iv.mpf(norm)*iv.mpf(float(h.hi))
    tail = upper((iv.exp(z)*z**(ORDER+1)/math.factorial(ORDER+1)).b)
    error = I(rowsum(rows)[:, None])*I(np.max(generators.mag(), axis=0)[None, :])*I(tail)
    return result+I(-error.hi, error.hi), dict(order=ORDER, norm_inf_upper=norm,
        exponential_tail=tail, maximum_row_generator_tail=float(np.max(error.hi)))


class Ring:
    def __init__(self, regions):
        if regions not in (2, 3):
            raise ValueError("This frozen numerical protocol has two network sizes")
        self.r = regions; self.nc = regions**2; self.n = self.nc+regions
        self.B = np.zeros((self.nc, regions))
        self.demand = np.zeros(self.nc)
        self.initial = np.zeros(self.n)
        for i in range(regions):
            for j in range(regions):
                index = i*regions+j
                self.B[index, i] = .5 if i == j else .5/(regions-1)
                self.demand[index] = 60. if i == j else 5./(regions-1)
                self.initial[index] = 15. if i == j else 15./(regions-1)
        self.initial[self.nc:] = 2.
        self.transform = np.eye(self.n)
        self.transform[:self.nc, self.nc:] = self.B
        self.inverse = np.eye(self.n)
        self.inverse[:self.nc, self.nc:] = -self.B
        self.regional_rows = np.zeros((regions, self.n))
        for i in range(regions):
            self.regional_rows[i, i*regions:(i+1)*regions] = 1.
        self.rows = np.vstack((-np.eye(self.n)[:self.nc], self.regional_rows,
                               -np.eye(self.n)[self.nc:], np.eye(self.n)[self.nc:]))
        self.limits = np.r_[np.zeros(self.nc), np.full(regions, 100.),
                            np.zeros(regions), np.full(regions, 20.)]
        self.names = ([f"class_{i}_{j}_lower" for i in range(regions) for j in range(regions)]
                      +[f"region_{i}_upper" for i in range(regions)]
                      +[f"queue_{i}_lower" for i in range(regions)]
                      +[f"queue_{i}_upper" for i in range(regions)])

    def command(self, t):
        route = np.full(self.r, .65); entry = np.full(self.r, .5)
        if t >= Fraction(1, 2):
            route[0] = .8; route[1] = .55; entry[0] = .6
        return np.r_[route, entry]

    def arrival(self, t):
        value = np.full(self.r, 40.)
        if Fraction(1, 4) <= t < Fraction(3, 4):
            value[0] = 52.
        return value

    def traffic(self, c, command):
        """Generic float, interval or polynomial directed-edge implementation."""
        zero = c[0]*0.
        out = [zero+self.demand[k] for k in range(self.nc)]
        for i in range(self.r):
            total = sum(c[i*self.r+j] for j in range(self.r))
            speed = (total*MFD_A+MFD_B)*total+MFD_C
            for j in range(self.r):
                index = i*self.r+j
                flow = c[index]*speed*(1. if i == j else command[i])
                out[index] = out[index]-flow
                if i != j:
                    following = ((i+1) % self.r)*self.r+j
                    out[following] = out[following]+flow
        return out

    def field(self, state, command, arrival, branches=None, method="class"):
        out = self.traffic(state[:self.nc], command)
        admission = []
        for i in range(self.r):
            if branches is None:
                value = min(60.*state[self.nc+i], 120.*command[self.r+i])
            else:
                value = (state[self.nc+i]*60. if branches[i] == "supply"
                         else (state[self.nc+i]*0.+command[self.r+i])*120.)
            admission.append(value)
        for k in range(self.nc):
            for i in range(self.r):
                if self.B[k, i]:
                    out[k] = out[k]+self.B[k, i]*(admission[i] if method == "class" else arrival[i])
        out.extend(arrival[i]-admission[i] for i in range(self.r))
        return out

    def traffic_jacobian(self, box, command):
        lo = np.zeros((self.nc, self.nc)); hi = lo.copy()
        H = I(lo, hi)
        for i in range(self.r):
            total = isum(box[i*self.r+j] for j in range(self.r))
            speed = (total*I(MFD_A)+I(MFD_B))*total+I(MFD_C)
            slope = total*I(2.)*I(MFD_A)+I(MFD_B)
            for j in range(self.r):
                row = i*self.r+j
                for k in range(self.r):
                    col = i*self.r+k
                    deriv = (box[row]*slope+(speed if j == k else I(0.)))*I(1. if i == j else command[i])
                    value = H[row, col]-deriv
                    H.lo[row, col], H.hi[row, col] = value.lo, value.hi
                    if i != j:
                        target = ((i+1) % self.r)*self.r+j
                        value = H[target, col]+deriv
                        H.lo[target, col], H.hi[target, col] = value.lo, value.hi
        return H

    def jacobian(self, physical_box, command, method):
        H = self.traffic_jacobian(physical_box, command)
        lo = np.zeros((self.n, self.n)); hi = lo.copy()
        lo[:self.nc, :self.nc], hi[:self.nc, :self.nc] = H.lo, H.hi
        for i in range(self.r):
            gap = physical_box[self.nc+i]*I(60.)-I(command[self.r+i])*I(120.)
            derivative = I(60.) if gap.hi < 0 else I(0.) if gap.lo > 0 else I(0., 60.)
            lo[self.nc+i, self.nc+i], hi[self.nc+i, self.nc+i] = -derivative.hi, -derivative.lo
            if method == "class":
                column = I(self.B[:, i])*derivative
                lo[:self.nc, self.nc+i], hi[:self.nc, self.nc+i] = column.lo, column.hi
        if method == "inventory":
            block = -imat(H, self.B)
            lo[:self.nc, self.nc:], hi[:self.nc, self.nc:] = block.lo, block.hi
        return I(lo, hi)

    def physical_jacobian_mid(self, point, command):
        return self.jacobian(I(point), command, "class").mid()

    def descriptor(self):
        return dict(regions=self.r, dimension=self.n, class_order=[f"c_{i}_{j}" for i in range(self.r) for j in range(self.r)],
            queue_order=[f"z_{i}" for i in range(self.r)], B=self.B.tolist(),
            routing="clockwise_ring_until_final_destination", class_demand=self.demand.tolist(),
            physical_initial_center=self.initial.tolist(), mfd_binary_coefficients=[MFD_A, MFD_B, MFD_C],
            installed_plan=[dict(left="0", right="1/2", command=self.command(Fraction(0)).tolist()),
                            dict(left="1/2", right="1", command=self.command(Fraction(1, 2)).tolist())],
            service=dict(absolute_time="1", all_regional_upper_vehicles=2700.),
            physical_row_names=self.names, physical_rows=self.rows.tolist(), physical_limits=self.limits.tolist())


def branch_choices(net, curve, command):
    choices = []
    for i in range(net.r):
        gap = curve[net.nc+i]*I(60.)-I(command[net.r+i])*I(120.)
        choices.append(("supply",) if gap.hi < 0 else ("gate",) if gap.lo > 0 else ("supply", "gate"))
    return list(itertools.product(*choices))


def interval_vector(values):
    return I(np.array([float(v.lo) for v in values]), np.array([float(v.hi) for v in values]))


def positivity(net, widths, residual):
    discriminant_margin = I(4.)*I(MFD_A)*I(MFD_C)-I(MFD_B)*I(MFD_B)
    initial = I(net.initial)+I(-widths, widths)
    mass_upper = isum(initial[k] for k in range(net.n))+I(1.)*(
        isum(I(v) for v in net.demand)+(I(52.)+I(residual))*I(net.r))
    # This mass bound is intentionally independent of tested physical capacities.
    return dict(passed=bool(discriminant_margin.lo > 0 and MFD_A > 0 and np.min(initial.lo) >= 0
                           and np.min(net.demand) >= 0 and 40.-residual >= 0),
        four_ac_minus_b_squared_lower=float(discriminant_margin.lo),
        lower_arrival_per_queue=40.-residual, initial_lower=initial.lo.tolist(),
        one_hour_mass_upper=float(mass_upper.hi), upper_capacity_used_in_argument=False)


class Failure(Exception):
    def __init__(self, reason, evidence):
        self.reason, self.evidence = reason, evidence
        super().__init__(reason)


def validated_step(net, method, center, G, rho, queue_boxes, left, right, residual):
    """A strict first-exit enclosure in unchanged coordinates; no hull restart."""
    n, nc = net.n, net.nc
    L = net.transform if method == "inventory" else np.eye(n)
    T = net.inverse if method == "inventory" else np.eye(n)
    command, arrival = net.command(left), net.arrival(left)
    h = rat(right-left); tau = I(0., h.hi)
    queues = [queue_step_bounds(*queue_boxes[i], right-left, I(arrival[i]), I(0.), residual, command[net.r+i])
              for i in range(net.r)]
    branch0 = tuple("supply" if 60.*center[nc+i] < 120.*command[net.r+i] else "gate" for i in range(net.r))
    fpoint = interval_vector(net.field([I(v) for v in center], command, [I(a) for a in arrival], branch0))
    velocity = fpoint.mid()
    acceleration = (imat(net.jacobian(I(center), command, "class"), fpoint)*I(.5)).mid()
    curve = I(center)+I(velocity)*tau+I(acceleration)*tau*tau
    polynomial = [J([center[k], velocity[k], acceleration[k]]) for k in range(n)]
    metric_velocity, metric_acceleration = imat(L, velocity), imat(L, acceleration)
    defect = np.zeros(n)
    for branches in branch_choices(net, curve, command):
        values = net.field(polynomial, command, [J([v]) for v in arrival], branches, method)
        residual_polys = [values[k]-J([metric_velocity[k], I(2.)*metric_acceleration[k]]) for k in range(n)]
        degree = max(len(v.c) for v in residual_polys)
        assert degree <= 7
        support = I(np.zeros(n)); power = I(1.)
        for degree_index in range(degree):
            coefficient = interval_vector([v.c[degree_index] if degree_index < len(v.c) else I(0.) for v in residual_polys])
            support = support+I(coefficient.mag())*power
            power = power*I(h.hi)
        defect = np.maximum(defect, support.hi)
    arrival_columns = I(L[:, nc:])
    base_force = (I(defect)+I(rowsum(arrival_columns))*I(residual)).hi
    endpoint = I(center)+I(velocity)*h+I(acceleration)*h*h
    next_center = endpoint.mid()
    rounding = imat(L, endpoint-I(next_center)).mag()
    A = net.jacobian(I(center), command, method).mid()
    E, _, etail = exp_integral(A, h)
    C = imat(net.rows, T)
    joined = I(np.vstack((np.eye(n), C.lo)), np.vstack((np.eye(n), C.hi)))
    flow, flow_tail = signed_range(joined, A, G, h)
    flow_support = rowsum(flow)
    initial_radius = (I(rowsum(G))+I(rho)).hi
    trial = (I(1.1)*(I(np.maximum(initial_radius, flow_support[:n]))+I(h.hi)*I(base_force)+I(1e-12))).hi
    queue_hulls = [[min(float(curve.lo[nc+i]), queues[i]["continuous"][0]),
                    max(float(curve.hi[nc+i]), queues[i]["continuous"][1])] for i in range(net.r)]
    attempts = []
    for iteration in range(18):
        support = imat(I(abs(T)), I(trial)).hi
        corridor = curve+I(-support, support)
        for i in range(net.r):
            corridor.lo[nc+i] = max(corridor.lo[nc+i], queue_hulls[i][0])
            corridor.hi[nc+i] = min(corridor.hi[nc+i], queue_hulls[i][1])
        if np.any(corridor.lo > corridor.hi):
            raise ArithmeticError("Empty mean-value corridor")
        Jbox = net.jacobian(corridor, command, method)
        M = Jbox.mag(); np.fill_diagonal(M, np.diag(Jbox.hi))
        EM, Q, mtail = exp_integral(M, h)
        if np.min(EM.hi) < 0 or np.min(Q.hi) < 0:
            raise ArithmeticError("Invalid Metzler propagator upper enclosure")
        remainder_flow, rtail = signed_range(np.eye(n), M, np.diag(rho), h)
        mismatch = rowsum(imat(Jbox-I(A), flow[:n]))
        force = (I(mismatch)+I(base_force)).hi
        convolution = imat(I(Q.hi), I(force)).hi
        continuous_remainder = (I(rowsum(remainder_flow))+I(convolution)).hi
        radius = (I(flow_support[:n])+I(continuous_remainder)).hi
        attempts.append(dict(trial=trial.tolist(), radius=radius.tolist()))
        if np.all(radius < trial):
            row_curve = imat(net.rows, center)+imat(net.rows, velocity)*tau+imat(net.rows, acceleration)*tau*tau
            error = I(flow_support[n:])+imat(I(C.mag()), I(continuous_remainder))
            margins = (I(net.limits)-row_curve-error).lo.copy()
            raw = margins.copy()
            margins[:nc] = np.maximum(margins[:nc], 0.)  # Independent positivity, never tube contraction.
            for i in range(net.r):
                margins[nc+net.r+i] = max(margins[nc+net.r+i], queues[i]["continuous"][0])
                margins[nc+2*net.r+i] = max(margins[nc+2*net.r+i], float((I(20.)-I(queues[i]["continuous"][1])).lo))
            EG = imat(E, G); next_G = EG.mid()
            next_rho = (I(rowsum(EG-I(next_G)))+imat(I(EM.hi), I(rho))+I(convolution)+I(rounding)).hi
            record = dict(left=str(left), right=str(right), command=command.tolist(), arrival=arrival.tolist(),
                center=center.tolist(), velocity=velocity.tolist(), acceleration=acceleration.tolist(),
                initial_G=G.tolist(), initial_rho=rho.tolist(), queue_initial=queue_boxes,
                next_center=next_center.tolist(), next_G=next_G.tolist(), next_rho=next_rho.tolist(),
                queues=queues, mean_value_queue_hulls=queue_hulls, corridor_lower=corridor.lo.tolist(), corridor_upper=corridor.hi.tolist(),
                interval_J_lower=Jbox.lo.tolist(), interval_J_upper=Jbox.hi.tolist(), frozen_A=A.tolist(), comparison_M=M.tolist(),
                E_lower=E.lo.tolist(), E_upper=E.hi.tolist(), EM_lower=EM.lo.tolist(), EM_upper=EM.hi.tolist(),
                Q_lower=Q.lo.tolist(), Q_upper=Q.hi.tolist(), defect_upper=defect.tolist(), base_force=base_force.tolist(),
                generator_mismatch=mismatch.tolist(), force_upper=force.tolist(), convolution=convolution.tolist(),
                continuous_remainder=continuous_remainder.tolist(), generator_flow_support=flow_support.tolist(),
                row_predictor_lower=row_curve.lo.tolist(), row_predictor_upper=row_curve.hi.tolist(),
                row_error_upper=error.hi.tolist(), raw_physical_margins=raw.tolist(), physical_margins=margins.tolist(),
                attempts=attempts, tails=dict(frozen=etail, comparison=mtail, generator=flow_tail, remainder=rtail),
                generator_or_remainder_contracted=False)
            return next_center, next_G, next_rho, [q["endpoint"] for q in queues], record
        trial = np.maximum((I(trial)*I(1.1)).hi, (I(radius)*I(1.1)+I(1e-12)).hi)
        if not np.all(np.isfinite(trial)) or max(trial) > 100.:
            break
    raise Failure("self_inclusion_failed", dict(left=str(left), right=str(right), attempts=attempts))


def verify(net, method, width_vehicles, residual_veh_h, source_hashes):
    started = time.perf_counter()
    widths = np.r_[np.full(net.nc, width_vehicles/(100.*net.r)), np.full(net.r, min(width_vehicles, 10.)/100.)]
    residual = residual_veh_h/100.
    L = net.transform if method == "inventory" else np.eye(net.n)
    T = net.inverse if method == "inventory" else np.eye(net.n)
    initial_G = imat(L, np.diag(widths)); G = initial_G.mid(); rho = rowsum(initial_G-I(G))
    center = net.initial.copy()
    qbox = I(center[net.nc:])+I(-widths[net.nc:], widths[net.nc:])
    queues = [[float(qbox.lo[i]), float(qbox.hi[i])] for i in range(net.r)]
    descriptor = dict(protocol_sha256=sha(PROTOCOL), source_sha256=source_hashes, network=net.descriptor(), method=method,
        regional_half_width_vehicles=width_vehicles, physical_class_half_widths_vehicles=(100*widths[:net.nc]).tolist(),
        physical_half_widths=widths.tolist(), per_queue_residual_vehicles_h=residual_veh_h,
        total_absolute_arrival_residual_bound_vehicles_h=net.r*residual_veh_h,
        transform=L.tolist(), inverse=T.tolist(), initial_G=G.tolist(), initial_rho=rho.tolist(),
        nominal_step=str(NOMINAL_STEP), minimum_step=str(MIN_STEP), power_order=ORDER)
    positive = positivity(net, widths, residual)
    if not positive["passed"]:
        raise ValueError("Independent positivity premises failed")
    initial_box = I(center)+I(-widths, widths)
    initial_margins = (I(net.limits)-imat(net.rows, initial_box)).lo
    if np.min(initial_margins) < 0:
        raise ValueError("Physical initial information box inadmissible")
    gate_checks = []
    for now in (Fraction(0), Fraction(1, 2)):
        coefficients = net.command(now)
        exact = [Fraction.from_float(float(u)) for u in coefficients]
        if not all(0 <= u <= 1 for u in exact):
            raise ValueError("Exact constant gate constraints failed")
        gate_checks.append(dict(left=str(now), exact_command=list(map(str, exact)), all_gates_in_unit_interval=True))
    initial_state = dict(center=center.tolist(), G=G.tolist(), rho=rho.tolist(), queues=queues)
    left = Fraction(0); steps = []; failed_trials = []; failure = None
    while left < 1:
        next_knot = min(t for t in [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)] if t > left)
        step = min(NOMINAL_STEP, next_knot-left)
        while True:
            try:
                candidate = validated_step(net, method, center, G, rho, queues, left, left+step, residual)
                break
            except Failure as exc:
                failed_trials.append(dict(reason=exc.reason, evidence=exc.evidence))
                if step <= MIN_STEP:
                    failure = dict(kind=exc.reason, left=str(left), evidence=exc.evidence)
                    break
                step /= 2
        if failure:
            break
        center, G, rho, queues, evidence = candidate
        steps.append(evidence); left += step
        if min(evidence["physical_margins"]) < 0:
            failure = dict(kind="physical_row_not_certified", right=str(left),
                rows=[net.names[k] for k, value in enumerate(evidence["physical_margins"]) if value < 0])
            break
    physical = failure is None and left == 1
    services = []
    if physical:
        transformed_rows = imat(net.regional_rows, T)
        terminal = imat(net.regional_rows, center)+I(rowsum(imat(transformed_rows, G)))+imat(I(transformed_rows.mag()), rho)
        for i in range(net.r):
            services.append(dict(region=i, upper_vehicles=float((terminal[i]*I(100.)).hi),
                limit_vehicles=2700., passed=bool(terminal[i].hi <= 27.)))
    service = physical and all(s["passed"] for s in services)
    if physical and not service:
        failure = dict(kind="scheduled_service_not_certified")
    return dict(descriptor=descriptor, descriptor_sha256=digest(descriptor), structural_positivity=positive, exact_gate_checks=gate_checks,
        initial_physical_margins=initial_margins.tolist(), initial=initial_state, steps=steps, failed_trials=failed_trials,
        continuous_physical_verified=physical, scheduled_service_verified=service, accepted=physical and service,
        service_rows=services, failure=failure, reached_time=str(left),
        terminal=dict(center=center.tolist(), G=G.tolist(), rho=rho.tolist(), queues=queues),
        wall_seconds=time.perf_counter()-started)


def independent_field(net, state, command, arrival):
    """Receiving-region formulation, independent of the edge-update RHS above."""
    c = np.asarray(state[:net.nc]).reshape(net.r, net.r)
    totals = c.sum(axis=1)
    speed = MFD_A*totals**2+MFD_B*totals+MFD_C
    admission = np.minimum(60.*state[net.nc:], 120.*command[net.r:])
    derivative = np.zeros_like(np.asarray(state))
    for i in range(net.r):
        previous = (i-1) % net.r
        for j in range(net.r):
            own = c[i, j]*speed[i]*(1. if i == j else command[i])
            incoming = (c[previous, j]*speed[previous]*command[previous] if j != previous else 0.)
            derivative[i*net.r+j] = net.demand[i*net.r+j]-own+incoming+net.B[i*net.r+j, i]*admission[i]
    derivative[net.nc:] = arrival-admission
    return derivative


def replay(net, *, width=0., residual_veh_h=0., adverse=False):
    """Tight-tolerance adaptive independent numerical diagnostic, not proof."""
    widths = np.r_[np.full(net.nc, width/(100.*net.r)), np.full(net.r, min(width, 10.)/100.)]
    initial = net.initial+(.5*widths*np.where(np.arange(net.n) % 2, -1., 1.) if adverse else 0.)
    state = np.r_[initial, 0., 0., 0.]
    times, states = [0.], [state.copy()]
    for quarter in range(4):
        left, right = Fraction(quarter, 4), Fraction(quarter+1, 4)
        command, arrival = net.command(left), net.arrival(left)
        if adverse:
            arrival = arrival+residual_veh_h/100.*np.where((np.arange(net.r)+quarter) % 2, -1., 1.)
        def rhs(t, augmented):
            actual = augmented[:net.n]
            dx = independent_field(net, actual, command, arrival)
            c = actual[:net.nc].reshape(net.r, net.r)
            totals = c.sum(axis=1)
            speed = MFD_A*totals**2+MFD_B*totals+MFD_C
            departures = np.sum(np.diag(c)*speed)
            return np.r_[dx, 100.*np.sum(actual), 100.*departures, 100.*(sum(net.demand)+sum(arrival))]
        sample = np.linspace(float(left), float(right), 241)
        solved = solve_ivp(rhs, [float(left), float(right)], state, method="DOP853", rtol=2e-12, atol=2e-13,
                           t_eval=sample, max_step=1/1200)
        if not solved.success:
            raise RuntimeError(solved.message)
        times.extend(solved.t[1:].tolist()); states.extend(solved.y[:, 1:].T.tolist()); state = solved.y[:, -1]
    array = np.asarray(states)
    return dict(adverse=adverse, initial=initial.tolist(), time_h=times, state=array[:, :net.n].tolist(),
        tts_veh_h=float(state[-3]), departures_vehicles=float(state[-2]), offered_vehicles=float(state[-1]),
        conservation_residual_vehicles=float(100.*sum(initial)+state[-1]-state[-2]-100.*sum(state[:net.n])),
        final_regional_vehicles=(100.*net.regional_rows@state[:net.n]).tolist(),
        maximum_queue_vehicles=(100*np.max(array[:, net.nc:net.n], axis=0)).tolist(),
        minimum_class_vehicles=float(100*np.min(array[:, :net.nc])))


def primitive_audit():
    cases = []
    for regions in (2, 3):
        net = Ring(regions)
        for i in range(net.n):
            for j in range(net.n):
                exact = sum(Fraction.from_float(float(net.transform[i, k]))*Fraction.from_float(float(net.inverse[k, j])) for k in range(net.n))
                assert exact == int(i == j)
        S = np.vstack((net.B, -np.eye(net.r)))
        assert np.array_equal(net.transform@S, np.vstack((np.zeros((net.nc, net.r)), -np.eye(net.r))))
        assert np.array_equal(net.B.sum(axis=0), np.ones(net.r))
        for seed in range(4):
            rng = np.random.default_rng(3700+10*regions+seed)
            x = net.initial*rng.uniform(.8, 1.2, net.n)
            x[net.nc:] = .2 if seed % 2 else 2.
            command = net.command(Fraction(seed % 2, 2)); arrival = net.arrival(Fraction(1, 3))
            values = np.asarray(net.field(x, command, arrival))
            independent = independent_field(net, x, command, arrival)
            field_error = float(max(abs(values-independent)))
            assert field_error < 1e-10
            totals = net.regional_rows@x
            mass_derivative = sum(net.demand)+sum(arrival)-sum(x[i*regions+i]*(MFD_A*totals[i]**2+MFD_B*totals[i]+MFD_C) for i in range(regions))
            assert abs(sum(values)-mass_derivative) < 1e-10
            branches = tuple("supply" if 60.*x[net.nc+i] < 120.*command[regions+i] else "gate" for i in range(regions))
            analytic = net.jacobian(I(x), command, "class")
            complex_j = np.zeros((net.n, net.n)); central_j = complex_j.copy()
            for k in range(net.n):
                perturbation = np.zeros(net.n); perturbation[k] = 1.
                complex_j[:, k] = np.imag(net.field(x.astype(complex)+1e-25j*perturbation, command, arrival, branches))/1e-25
                central_j[:, k] = (np.asarray(net.field(x+1e-5*perturbation, command, arrival, branches))-
                                   np.asarray(net.field(x-1e-5*perturbation, command, arrival, branches)))/(2e-5)
            cs_error = float(np.max(abs(complex_j-analytic.mid())))
            fd_error = float(np.max(abs(central_j-analytic.mid())))
            assert cs_error < 1e-10 and fd_error < 1e-7
            transformed = net.jacobian(I(x), command, "inventory")
            transform_error = float(np.max(abs(transformed.mid()-net.transform@complex_j@net.inverse)))
            assert transform_error < 1e-10
            assert any(abs(complex_j[j, k]) > 1e-6 for j in range(net.nc) for k in range(net.nc) if j//regions != k//regions)
            cases.append(dict(regions=regions, seed=seed, field_error=field_error,
                complex_step_error=cs_error, central_difference_error=fd_error, transformed_jacobian_error=transform_error))
    matrix_checks = []
    with mp.workdps(90):
        for n in (6, 12):
            A = np.diag(np.linspace(-60., -1., n))
            for k in range(n-1):
                A[k, k+1] = .75; A[k+1, k] = -.25
            h = rat(Fraction(1, 120)); E, Q, tails = exp_integral(A, h)
            exact_A = mp.matrix([[mp.mpf(float(A[i, j])) for j in range(n)] for i in range(n)])
            exact_h = mp.mpf(1)/120
            exact_E = mp.expm(exact_A*exact_h)
            exact_Q = mp.inverse(exact_A)*(exact_E-mp.eye(n))
            for i in range(n):
                for j in range(n):
                    assert mp.mpf(float(E.lo[i, j])) <= exact_E[i, j] <= mp.mpf(float(E.hi[i, j]))
                    assert mp.mpf(float(Q.lo[i, j])) <= exact_Q[i, j] <= mp.mpf(float(Q.hi[i, j]))
            matrix_checks.append(dict(dimension=n, entries_checked=2*n*n, tails=tails))
    return dict(passed=True, field_jacobian_cases=cases, matrix_checks=matrix_checks,
        exact_share_transform_and_incidence_checks=True, connected_interregion_coupling_checked=True)


def grid():
    return [dict(regions=r, method=m, width_vehicles=d, residual_veh_h=w,
                 case_id=f"r{r}_{m}_d{d}_w{w}")
            for r in (2, 3) for d, w in ((1, 60), (10, 60), (25, 60), (25, 300)) for m in ("class", "inventory")]


def audit_record(proof):
    assert proof["descriptor_sha256"] == digest(proof["descriptor"])
    net = Ring(proof["descriptor"]["network"]["regions"])
    assert proof["descriptor"]["network"] == net.descriptor()
    previous = proof["initial"]; clock = Fraction(0)
    for step in proof["steps"]:
        assert Fraction(step["left"]) == clock
        assert step["center"] == previous["center"]
        assert step["initial_G"] == previous["G"] and step["initial_rho"] == previous["rho"]
        assert step["queue_initial"] == previous["queues"]
        assert step["command"] == net.command(clock).tolist() and step["arrival"] == net.arrival(clock).tolist()
        assert np.all(np.asarray(step["attempts"][-1]["radius"]) < np.asarray(step["attempts"][-1]["trial"]))
        assert step["generator_or_remainder_contracted"] is False
        for a, b in zip(step["row_predictor_upper"], step["row_error_upper"]):
            assert math.isfinite(a+b)
        clock = Fraction(step["right"])
        previous = dict(center=step["next_center"], G=step["next_G"], rho=step["next_rho"], queues=[q["endpoint"] for q in step["queues"]])
    assert str(clock) == proof["reached_time"]
    if proof["continuous_physical_verified"]:
        assert clock == 1 and all(min(step["physical_margins"]) >= 0 for step in proof["steps"])
    assert proof["accepted"] == (proof["continuous_physical_verified"] and proof["scheduled_service_verified"])
    return dict(passed=True, steps=len(proof["steps"]), descriptor_bound=True, clock_and_correlated_chain=True)


def run(limit=None):
    RESULTS.mkdir(exist_ok=True)
    hashes = sources(); menu = grid()
    if OUTPUT.exists():
        record = json.loads(OUTPUT.read_text())
        if record["source_sha256"] != hashes or record["grid"] != menu:
            raise RuntimeError("Preserve frozen outcomes; source/protocol or grid changed")
    else:
        audit = primitive_audit()
        snapshots = {}
        target = RESULTS / "mrs" / digest(hashes)[:12]
        target.mkdir(parents=True, exist_ok=True)
        for name, expected in hashes.items():
            destination = target / Path(name).name
            shutil.copyfile(ROOT/name, destination)
            assert sha(destination) == expected
            snapshots[name] = str(destination.relative_to(ROOT)).replace("\\", "/")
        record = dict(protocol_sha256=sha(PROTOCOL), source_sha256=hashes, source_snapshots=snapshots,
            grid=menu, primitive_audit=audit, cases=[], replays={})
        save(OUTPUT, record)
    completed = {case["case_id"] for case in record["cases"]}
    count = 0
    for spec in menu:
        if spec["case_id"] in completed:
            continue
        if limit is not None and count >= limit:
            break
        net = Ring(spec["regions"])
        proof = verify(net, spec["method"], spec["width_vehicles"], spec["residual_veh_h"], hashes)
        proof_path = RESULTS/f"{PREFIX}_{spec['case_id']}_proof.json"
        if proof_path.exists():
            raise RuntimeError("Refusing to overwrite a previous proof")
        save(proof_path, proof)
        audited = audit_record(proof)
        summary = dict(**spec, accepted=proof["accepted"], physical=proof["continuous_physical_verified"],
            service=proof["scheduled_service_verified"], terminal_service_rows=proof["service_rows"],
            reached_time=proof["reached_time"], failure=proof["failure"], steps=len(proof["steps"]),
            wall_seconds=proof["wall_seconds"], proof_file=str(proof_path.relative_to(ROOT)).replace("\\", "/"),
            proof_sha256=sha(proof_path), audit=audited)
        record["cases"].append(summary); count += 1
        assert sources() == hashes
        save(OUTPUT, record)
        print(json.dumps(summary), flush=True)
    for r in (2, 3):
        key = str(r)
        if key not in record["replays"]:
            record["replays"][key] = dict(nominal=replay(Ring(r)), adverse=replay(Ring(r), width=25., residual_veh_h=300., adverse=True))
            save(OUTPUT, record)
    record["complete"] = len(record["cases"]) == len(menu)
    assert sources() == hashes
    save(OUTPUT, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(primitive_audit(), indent=2), flush=True)
    else:
        result = run(args.limit)
        print(json.dumps(dict(complete=result["complete"], cases=len(result["cases"])), indent=2), flush=True)
