#!/usr/bin/env python3
"""Continuous finite-plan interval verifier for the seven-state incident model.

The mathematical plant uses the exact binary floating-point installed physical
and controller coefficients. All acceptance inequalities use outward intervals;
ODE tolerances, sampled trajectories, and raw eigensolver output are not proof
bounds. A quadratic predictor has an interval-polynomial residual of degree at
most eight. A self-validating transformed error ball and an interval logarithmic
norm bound enclose every measurable exterior-arrival residual in [-eta,eta].

No global stability, recursive feasibility, or unbounded-time guarantee is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from mpmath import iv

import mfd_incident_model as model
# Dimension-independent outward arithmetic reused without changing old module.
from mfd_deadline_certificates import I, J, imat, inverse_interval, isum, norm_upper, operator_upper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "mfd_incident_verifier_pilot_results.json"
DIM = 7


def fraction(value: float | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def interval_fraction(value: Fraction) -> I:
    near = float(value)
    exact = Fraction.from_float(near)
    return I(np.nextafter(near, -np.inf) if exact > value else near,
             np.nextafter(near, np.inf) if exact < value else near)


def hull(a: I, b: I) -> I:
    return I(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def transpose(a: I) -> I:
    return I(a.lo.T, a.hi.T)


def symmetric_eigen_upper(matrix: I) -> float:
    """Gershgorin after an interval-verified numerical change of basis.

    The numerical eigenvectors only choose a basis. Its Gram matrix is enclosed
    and its positive lower eigenvalue bound verifies invertibility.
    """
    matrix = I(matrix)
    _, basis = np.linalg.eigh(matrix.mid())
    transformed = imat(basis.T, imat(matrix, basis))
    gram = imat(basis.T, basis)
    size = matrix.lo.shape[0]
    upper = max(float((I(transformed[i, i].hi) +
                       isum(I(transformed[i, j].mag()) for j in range(size) if j != i)).hi)
                for i in range(size))
    gram_lower = min(float((I(gram[i, i].lo) -
                            isum(I(gram[i, j].mag()) for j in range(size) if j != i)).lo)
                     for i in range(size))
    gram_upper = max(float((I(gram[i, i].hi) +
                            isum(I(gram[i, j].mag()) for j in range(size) if j != i)).hi)
                     for i in range(size))
    if gram_lower <= 0:
        raise ValueError("interval eigenbasis Gram lower bound is not positive")
    denominator = gram_lower if upper >= 0 else gram_upper
    return float((iv.mpf(upper) / iv.mpf(denominator)).b)


class Dual:
    """Seven-dimensional interval automatic differentiation."""
    def __init__(self, value: Any, derivative: Any = None):
        self.v = I(value)
        self.d = I(np.zeros(DIM) if derivative is None else derivative)

    @staticmethod
    def cast(other):
        return other if isinstance(other, Dual) else Dual(other)

    def __add__(self, other):
        other = self.cast(other)
        return Dual(self.v + other.v, self.d + other.d)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __sub__(self, other):
        return self + (-self.cast(other))

    def __rsub__(self, other):
        return self.cast(other) + (-self)

    def __mul__(self, other):
        other = self.cast(other)
        return Dual(self.v * other.v, self.d * other.v + self.v * other.d)

    __rmul__ = __mul__


class Polynomial:
    """Sparse seven-variable interval polynomial, used only in cached setup."""
    zero = (0,) * DIM

    def __init__(self, terms):
        self.terms = terms

    @staticmethod
    def cast(value):
        return value if isinstance(value, Polynomial) else Polynomial({Polynomial.zero: I(value)})

    def __add__(self, other):
        other = self.cast(other)
        result = self.terms.copy()
        for key, value in other.terms.items():
            result[key] = result.get(key, I(0.)) + value
        return Polynomial(result)

    __radd__ = __add__

    def __neg__(self):
        return Polynomial({key: -value for key, value in self.terms.items()})

    def __sub__(self, other):
        return self + (-self.cast(other))

    def __rsub__(self, other):
        return self.cast(other) + (-self)

    def __mul__(self, other):
        other = self.cast(other)
        result = {}
        for left, a in self.terms.items():
            for right, b in other.terms.items():
                key = tuple(left[i] + right[i] for i in range(DIM))
                result[key] = result.get(key, I(0.)) + a * b
        return Polynomial(result)

    __rmul__ = __mul__


def transformed_jacobian_record(mode, L, inverse):
    variables = []
    for index in range(DIM):
        exponent = [0] * DIM
        exponent[index] = 1
        variables.append(Polynomial({tuple(exponent): I(1.)}))
    states = [Polynomial.cast(mode.center[i]) + sum(variables[j] * inverse[i, j] for j in range(DIM))
              for i in range(DIM)]
    physical = closed_field(states, mode, 0., "gate")
    transformed = [sum(physical[j] * L[i, j] for j in range(DIM)) for i in range(DIM)]
    derivatives, keys = [], set()
    for i in range(DIM):
        for j in range(DIM):
            terms = {}
            for exponent, coefficient in transformed[i].terms.items():
                if exponent[j]:
                    lower = list(exponent)
                    lower[j] -= 1
                    lower = tuple(lower)
                    terms[lower] = coefficient * I(exponent[j])
                    keys.add(lower)
            derivatives.append(terms)
    keys = sorted(keys)
    coefficients = I(np.array([[terms.get(key, I(0.)).lo for key in keys] for terms in derivatives]),
                     np.array([[terms.get(key, I(0.)).hi for key in keys] for terms in derivatives]))
    # Switching to supply-limited admission changes only the constant Jacobian.
    row = I(model.RELEASE_RATE) * I(np.eye(DIM)[6]) - I(model.CAPACITY) * I(mode.gain[2])
    channel = I(np.array([0., 1., 0., .5, 0., 0., -1.]).reshape(DIM, 1))
    physical_difference = imat(channel, I(row.lo.reshape(1, DIM), row.hi.reshape(1, DIM)))
    difference = imat(L, imat(physical_difference, inverse))
    derivative_keys = sorted(set(tuple(exponent[k] - int(k == j) for k in range(DIM))
                                 for exponent in keys for j in range(DIM) if exponent[j]))
    derivative_index = {key: index for index, key in enumerate(derivative_keys)}
    gradient_lo = np.zeros((DIM * DIM, DIM, len(derivative_keys)))
    gradient_hi = np.zeros_like(gradient_lo)
    for column, exponent in enumerate(keys):
        for j in range(DIM):
            if exponent[j]:
                key = tuple(exponent[k] - int(k == j) for k in range(DIM))
                product = coefficients[:, column] * I(exponent[j])
                gradient_lo[:, j, derivative_index[key]] = product.lo
                gradient_hi[:, j, derivative_index[key]] = product.hi
    shift_groups = []
    key_index = {key: index for index, key in enumerate(keys)}
    for beta in keys:
        parent_indices, differences, multipliers = [], [], []
        for parent, alpha in enumerate(keys):
            if all(a >= b for a, b in zip(alpha, beta)):
                parent_indices.append(parent)
                differences.append(key_index[tuple(a - b for a, b in zip(alpha, beta))])
                multipliers.append(math.prod(math.comb(a, b) for a, b in zip(alpha, beta)))
        shift_groups.append((np.array(parent_indices), np.array(differences), np.array(multipliers)))
    return (np.array(keys, dtype=int), coefficients, difference,
            np.array(derivative_keys, dtype=int),
            I(gradient_lo.reshape(DIM**3, -1), gradient_hi.reshape(DIM**3, -1)), shift_groups)


def monomials(powers, box):
    terms = I(np.ones(len(powers)))
    for index in range(DIM):
        for power in range(1, 4):
            mask = powers[:, index] >= power
            terms *= I(np.where(mask, box.lo[index], 1.), np.where(mask, box.hi[index], 1.))
    return terms


def evaluate_transformed_jacobian(record, box, branch, centered=False, symmetric=False):
    powers, coefficients, difference, gradient_powers, gradient_coefficients, shift_groups = record
    if symmetric:
        coefficient_transpose = I(coefficients.lo.reshape(DIM, DIM, -1).transpose(1, 0, 2).reshape(DIM * DIM, -1),
                                  coefficients.hi.reshape(DIM, DIM, -1).transpose(1, 0, 2).reshape(DIM * DIM, -1))
        coefficients = (coefficients + coefficient_transpose) * I(.5)
        difference = (difference + transpose(difference)) * I(.5)
        gradient_transpose = I(gradient_coefficients.lo.reshape(DIM, DIM, DIM, -1).transpose(1, 0, 2, 3).reshape(DIM**3, -1),
                               gradient_coefficients.hi.reshape(DIM, DIM, DIM, -1).transpose(1, 0, 2, 3).reshape(DIM**3, -1))
        gradient_coefficients = (gradient_coefficients + gradient_transpose) * I(.5)
    if centered == "exact":
        point = box.mid()
        point_terms = monomials(powers, I(point))
        shifted_lo = np.zeros_like(coefficients.lo)
        shifted_hi = np.zeros_like(coefficients.hi)
        for index, (parents, remainders, multipliers) in enumerate(shift_groups):
            values = point_terms[remainders] * I(multipliers)
            shifted = imat(coefficients[:, parents], values)
            shifted_lo[:, index], shifted_hi[:, index] = shifted.lo, shifted.hi
        flattened = imat(I(shifted_lo, shifted_hi), monomials(powers, box - I(point)))
    elif centered:
        point = box.mid()
        flattened = imat(coefficients, monomials(powers, I(point)))
        gradients = imat(gradient_coefficients, monomials(gradient_powers, box))
        gradients = I(gradients.lo.reshape(DIM * DIM, DIM), gradients.hi.reshape(DIM * DIM, DIM))
        flattened += imat(gradients, box - I(point))
    else:
        flattened = imat(coefficients, monomials(powers, box))
    gate = I(flattened.lo.reshape(DIM, DIM), flattened.hi.reshape(DIM, DIM))
    return gate if branch == "gate" else gate + difference if branch == "supply" else hull(gate, gate + difference)


def closed_field(state, mode: model.Mode, arrival, branch: str):
    """Generic interval/dual/polynomial vector field for either smooth branch."""
    command = [mode.feedforward[i] + sum((state[j] - mode.center[j]) * mode.gain[i, j]
                                        for j in range(DIM)) for i in range(4)]
    n1, n2, n11, n22, n10, n20, queue = state
    u12, u21, u02, u20 = command
    a, b, c = model.original.MFD_A, model.original.MFD_B, model.original.MFD_C
    speed1, speed2 = a * n1 * n1 + b * n1 + c, a * n2 * n2 + b * n2 + c
    q11, q12, q10 = n11 * speed1, (n1 - n11 - n10) * speed1, n10 * speed1
    q22, q21, q20 = n22 * speed2, (n2 - n22 - n20) * speed2, n20 * speed2
    admission = model.CAPACITY * u02 if branch == "gate" else model.RELEASE_RATE * queue
    d11, d12, d10, d22, d21, d20 = map(float, model.INTERNAL_DEMAND)
    return [-q11 - (q12 + q10) * u12 + q21 * u21 + d11 + d12 + d10,
            -q22 + (q12 + q10) * u12 - q21 * u21 + admission - q20 * u20 + d22 + d21 + d20,
            -q11 + q21 * u21 + d11,
            -q22 + q12 * u12 + 0.5 * admission + d22,
            -q10 * u12 + d10,
            -q20 * u20 + q10 * u12 + d20,
            arrival - admission]


def branch_for_box(box: I, mode: model.Mode) -> tuple[str, I]:
    command = I(mode.feedforward[2]) + isum((box[j] - I(mode.center[j])) * I(mode.gain[2, j])
                                          for j in range(DIM))
    gap = I(model.RELEASE_RATE) * box[6] - I(model.CAPACITY) * command
    return ("gate" if gap.lo >= 0 else "supply" if gap.hi <= 0 else "both"), gap


def jacobian(box: I, mode: model.Mode, branch: str) -> I:
    def one(which):
        values = closed_field([Dual(box[i], np.eye(DIM)[i]) for i in range(DIM)], mode, 0., which)
        return I(np.array([value.d.lo for value in values]), np.array([value.d.hi for value in values]))
    return hull(one("gate"), one("supply")) if branch == "both" else one(branch)


def polynomial_interval(polynomial: J, tau: I) -> I:
    value = I(0.)
    for coefficient in reversed(polynomial.c):
        value = value * tau + coefficient
    return value


class Pulse:
    """Exact rational cutpoints for a trapezoid with installed float constants."""
    def __init__(self, amplitude=.3, duration=.5, start=.25, ramp=1 / 12):
        self.amplitude, self.duration, self.start, self.ramp = map(fraction, (amplitude, duration, start, ramp))
        if min(self.duration, self.ramp) <= 0 or self.duration < 2 * self.ramp or self.amplitude < 0:
            raise ValueError("nonnegative amplitude and a trapezoid with duration >= 2*ramp required")
        self.knots = [self.start, self.start + self.ramp,
                      self.start + self.duration - self.ramp, self.start + self.duration]

    def affine(self, left: Fraction, right: Fraction) -> tuple[I, I]:
        midpoint = (left + right) / 2
        base = fraction(model.BASE_ARRIVAL)
        if midpoint <= self.knots[0] or midpoint >= self.knots[3]:
            offset, slope = base, Fraction(0)
        elif midpoint < self.knots[1]:
            slope = base * self.amplitude / self.ramp
            offset = base + slope * (left - self.start)
        elif midpoint <= self.knots[2]:
            offset, slope = base * (1 + self.amplitude), Fraction(0)
        else:
            slope = -base * self.amplitude / self.ramp
            offset = base + base * self.amplitude * (self.start + self.duration - left) / self.ramp
        return interval_fraction(offset), interval_fraction(slope)


def affine_constraints(mode: model.Mode) -> tuple[list[str], I, I]:
    """Construct installed-feedback constraints without rounded affine offsets."""
    names, rows, bounds = [], [], []
    for index, name in enumerate(model.STATE_ORDER):
        names.append(name + "_lower")
        rows.append(I(-np.eye(DIM)[index]))
        bounds.append(I(0.))
    for name, indices in (("n12_lower", (0, 2, 4)), ("n21_lower", (1, 3, 5))):
        row = np.zeros(DIM)
        row[indices[0]], row[indices[1]], row[indices[2]] = -1., 1., 1.
        names.append(name)
        rows.append(I(row))
        bounds.append(I(0.))
    for index, limit, name in ((0, 100., "n1_upper"), (1, 100., "n2_upper"),
                                (6, model.QUEUE_LIMIT, "exterior_queue_upper")):
        names.append(name)
        rows.append(I(np.eye(DIM)[index]))
        bounds.append(I(limit))
    offset = I(mode.feedforward) - imat(mode.gain, mode.center)
    for index, name in enumerate(model.original.INPUT_ORDER):
        names.extend([name + "_upper", name + "_lower"])
        rows.extend([I(mode.gain[index]), -I(mode.gain[index])])
        bounds.extend([I(1.) - offset[index], offset[index]])
    for left, right in ((1, 0), (3, 2)):
        row = I(mode.gain[left]) - I(mode.gain[right])
        delta = offset[left] - offset[right]
        names.extend([f"direction_{left}_{right}_upper", f"direction_{left}_{right}_lower"])
        rows.extend([row, -row])
        bounds.extend([I(.75) - delta, I(.75) + delta])
    return names, I(np.array([row.lo for row in rows]), np.array([row.hi for row in rows])), \
        I(np.array([bound.lo for bound in bounds]), np.array([bound.hi for bound in bounds]))


class PlanVerifier:
    def __init__(self, library=None, jacobian_evaluation="transformed_exact_centered", metric_strategy="installed"):
        self.library = model.default_library() if library is None else list(library)
        self.cache: dict[int, dict[str, Any]] = {}
        if jacobian_evaluation not in ("transformed_exact_centered", "transformed_centered", "transformed_polynomial", "physical_interval"):
            raise ValueError("unknown Jacobian evaluation")
        self.jacobian_evaluation = jacobian_evaluation
        if metric_strategy not in ("installed", "common_branch", "conservation_diagonal", "trajectory_common", "conservation_box", "physical_box", "class_box"):
            raise ValueError("unknown metric strategy")
        self.metric_strategy = metric_strategy
        self.is_box = metric_strategy.endswith("_box")

    def choose_metric(self, mode):
        if self.metric_strategy == "installed" or self.is_box:
            return mode.P, dict(strategy="installed", scope="invertible coordinate choice only")
        import cvxpy as cp
        points = [alpha * model.BASE_STATE + (1 - alpha) * mode.center for alpha in np.linspace(0., 1., 5)]
        if self.metric_strategy == "trajectory_common":
            # These trajectories ONLY choose coordinates. Their accuracy is
            # immaterial to the subsequent independent interval proof.
            for amplitude, duration in ((0., 1.), (.3, 1.), (.4, .5)):
                x = model.BASE_STATE.copy()
                for index in range(540):
                    x = model.rk4(x, mode.command, index / 360., 1 / 360.,
                                  lambda t: model.arrival(t, amplitude=amplitude, duration=duration))
                    if index % 45 == 44 and np.all(np.isfinite(x)):
                        points.append(x.copy())
        matrices = [jacobian(I(point), mode, branch).mid() for point in points for branch in ("gate", "supply")]
        conservation = np.eye(DIM)
        conservation[1, 6], conservation[3, 6] = 1., .5
        weights = cp.Variable(DIM) if self.metric_strategy == "conservation_diagonal" else None
        metric = (conservation.T @ cp.diag(weights) @ conservation if weights is not None
                  else cp.Variable((DIM, DIM), symmetric=True))
        constraints = [metric >> np.eye(DIM)]
        constraints += [matrix.T @ metric + metric @ matrix + .5 * metric << -np.eye(DIM) * .01
                        for matrix in matrices]
        problem = cp.Problem(cp.Minimize(cp.trace(metric)), constraints)
        problem.solve(solver="CLARABEL", tol_gap_abs=1.e-9, tol_feas=1.e-9, tol_gap_rel=1.e-9)
        if metric.value is None or not np.all(np.isfinite(metric.value)):
            return mode.P, dict(strategy="common_branch_fallback_installed", optimizer_status=problem.status)
        candidate = (metric.value + metric.value.T) / 2
        if weights is not None:
            return candidate, dict(strategy="conservation_diagonal", optimizer_status=problem.status,
                                   diagonal_weights=weights.value.tolist(), design_points=5,
                                   scope="numerical geometry selection in y2=n2+queue,y4=n22+queue/2; interval-verified at runtime")
        return candidate, dict(strategy=self.metric_strategy, optimizer_status=problem.status,
                               design_points=len(points), requested_rate_per_hour=.5,
                               scope="numerical geometry selection only; all runtime bounds are independently interval-verified")

    def geometry(self, mode: model.Mode) -> dict[str, Any]:
        key = id(mode)
        fingerprint = hashlib.sha256(b"".join(np.asarray(value, dtype=float).tobytes()
                                             for value in (mode.center, mode.feedforward, mode.gain, mode.A, mode.P))).hexdigest()
        if key not in self.cache or self.cache[key]["fingerprint"] != fingerprint:
            metric, metric_selection = self.choose_metric(mode)
            chol = np.linalg.cholesky(metric).T
            transformed = chol @ mode.A @ np.linalg.inv(chol)
            _, rotation = np.linalg.eigh((transformed + transformed.T) / 2)
            left = rotation.T @ chol
            if self.is_box:
                left = np.array([[0, 0, 1, 0, 0, 0, 0],
                                 [1, 0, -1, 0, -1, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 1, 0, 0, .5],
                                 [0, 1, 0, -1, 0, -1, .5],
                                 [0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 1]], dtype=float)
                metric_selection = dict(strategy="conservation_box", admission_channel="transforms exactly to minus queue coordinate",
                                        coordinates=["n11", "n12", "n10", "n22+queue/2", "n21+queue/2", "n20", "queue"])
                if self.metric_strategy == "class_box":
                    left[3, 6], left[4, 6] = 0., 0.
                    metric_selection = dict(strategy="class_box", coordinates=["n11", "n12", "n10", "n22", "n21", "n20", "queue"])
                elif self.metric_strategy == "physical_box":
                    left = np.eye(DIM)
                    metric_selection = dict(strategy="physical_box", coordinates=list(model.STATE_ORDER))
            if metric_selection["strategy"] == "conservation_diagonal":
                conservation = np.eye(DIM)
                conservation[1, 6], conservation[3, 6] = 1., .5
                left = np.diag(np.sqrt(metric_selection["diagonal_weights"])) @ conservation
            inverse = inverse_interval(left)
            names, rows, bounds = affine_constraints(mode)
            support = np.array([norm_upper(imat(rows[i:i + 1], inverse)) for i in range(len(names))])
            coordinate_support = np.array([norm_upper(inverse[index]) for index in range(DIM)])
            self.cache[key] = dict(mode=mode, fingerprint=fingerprint,
                                   metric_selection=metric_selection,
                                   L=left, T=inverse, names=names, rows=rows, bounds=bounds,
                                   support=support, coordinate_support=coordinate_support,
                                   constraint_box_map=imat(rows, inverse).mag(),
                                   disturbance_factor=norm_upper(I(left[:, 6])),
                                   jacobian_record=transformed_jacobian_record(mode, left, inverse))
        return self.cache[key]

    def switch_radius(self, radius: float, source_L: np.ndarray, destination: model.Mode) -> float:
        geo = self.geometry(destination)
        if np.array_equal(source_L, geo["L"]):
            if self.is_box and not np.asarray(radius).ndim:
                return np.full(DIM, float(radius))
            if not self.is_box and np.asarray(radius).ndim:
                return norm_upper(I(radius))
            return radius
        transformation = imat(geo["L"], inverse_interval(source_L))
        if self.is_box:
            if np.asarray(radius).ndim:
                return imat(I(transformation.mag()), I(radius)).hi
            return np.array([float((I(norm_upper(transformation[i])) * I(radius)).hi) for i in range(DIM)])
        if np.asarray(radius).ndim:
            return norm_upper(imat(transformation, I(-np.asarray(radius), np.asarray(radius))))
        multiplier = operator_upper(transformation)
        return float((I(radius) * I(multiplier)).hi)

    def coordinate_error(self, geo, radius):
        return (imat(I(geo["T"].mag()), I(radius)) if np.asarray(radius).ndim else
                I(geo["coordinate_support"]) * I(radius))

    def step(self, center: np.ndarray, radius: float, mode: model.Mode,
             left: Fraction, right: Fraction, pulse: Pulse, residual_bound: float) -> tuple[np.ndarray, float, dict]:
        geo = self.geometry(mode)
        L, inverse = geo["L"], geo["T"]
        h = interval_fraction(right - left)
        tau = I(0., h.hi)
        arrival0, arrival_slope = pulse.affine(left, right)
        center_branch, _ = branch_for_box(I(center), mode)
        chosen = "gate" if center_branch == "both" else center_branch
        values = closed_field([I(value) for value in center], mode, arrival0, chosen)
        fbox = I(np.array([value.lo for value in values]), np.array([value.hi for value in values]))
        velocity = fbox.mid()
        acceleration_box = imat(jacobian(I(center), mode, chosen), fbox)
        forcing_slope = np.zeros(DIM)
        forcing_slope[6] = 1.
        acceleration = ((acceleration_box + I(forcing_slope) * arrival_slope) * I(.5)).mid()
        curve = [J([center[i], velocity[i], acceleration[i]]) for i in range(DIM)]
        curve_box = I(center) + I(velocity) * tau + I(acceleration) * tau * tau
        curve_branch, _ = branch_for_box(curve_box, mode)
        branches = ("gate", "supply") if curve_branch == "both" else (curve_branch,)
        defects, defect_vectors = [], []
        for branch in branches:
            predicted = closed_field(curve, mode, J([arrival0, arrival_slope]), branch)
            residuals = [predicted[i] - J([velocity[i], I(2.) * I(acceleration[i])]) for i in range(DIM)]
            degree = max(len(item.c) for item in residuals)
            if degree > 9:
                raise ValueError("unexpected predictor defect degree")
            transformed_residual = I(np.zeros(DIM))
            power = I(1.)
            # Transform coefficients BEFORE their absolute bounds; preserve cancellations.
            for index in range(degree):
                coefficient = I(np.array([item.c[index].lo if index < len(item.c) else 0. for item in residuals]),
                                np.array([item.c[index].hi if index < len(item.c) else 0. for item in residuals]))
                transformed_coefficient = imat(L, coefficient)
                transformed_residual += I(-transformed_coefficient.mag(), transformed_coefficient.mag()) * power
                power = power * I(h.hi)
            defects.append(norm_upper(transformed_residual))
            defect_vectors.append(transformed_residual.mag())
        defect = max(defects)
        forcing = float((I(geo["disturbance_factor"]) * I(residual_bound) + I(defect)).hi)
        endpoint = I(center) + I(velocity) * h + I(acceleration) * h * h
        next_center = endpoint.mid()
        rounding = norm_upper(imat(L, endpoint - I(next_center)))
        if self.is_box:
            forcing_vector = (I(abs(L[:, 6])) * I(residual_bound) + I(np.maximum.reduce(defect_vectors))).hi
            rounding_vector = imat(L, endpoint - I(next_center)).mag()
            trial = (I(1.1) * (I(radius) + I(h.hi) * I(forcing_vector) + I(1.e-12))).hi
            attempts = []
            for iteration in range(20):
                support = imat(I(inverse.mag()), I(trial)).hi
                physical_box = curve_box + I(-support, support)
                branch, branch_gap = branch_for_box(physical_box, mode)
                metric_curve = (imat(L, I(center) - I(mode.center)) + imat(L, velocity) * tau
                                + imat(L, acceleration) * tau * tau)
                metric_box = metric_curve + I(-trial, trial)
                matrices = [evaluate_transformed_jacobian(geo["jacobian_record"], metric_box, smooth,
                                                           centered="exact")
                            for smooth in (("gate", "supply") if branch == "both" else (branch,))]
                majorant = np.maximum.reduce([matrix.mag() for matrix in matrices])
                diagonal = np.maximum.reduce([np.diag(matrix.hi) for matrix in matrices])
                np.fill_diagonal(majorant, 0.)
                coupling = imat(I(majorant), I(trial)) + I(forcing_vector)
                ih = iv.mpf([float(h.lo), float(h.hi)])
                end_radius = np.empty(DIM)
                for index in range(DIM):
                    rate = float(diagonal[index])
                    if rate == 0.:
                        bound = iv.mpf(float(radius[index])) + ih * iv.mpf(float(coupling.hi[index]))
                    else:
                        factor = iv.exp(iv.mpf(rate) * ih)
                        # expm1 encloses exp(a*h)-1 directly, avoiding cancellation
                        # followed by division by a near-zero Jacobian bound.
                        bound = (factor * iv.mpf(float(radius[index])) + iv.expm1(iv.mpf(rate) * ih) / iv.mpf(rate)
                                 * iv.mpf(float(coupling.hi[index])))
                    end_radius[index] = float(bound.b)
                attempts.append(dict(trial_radius=trial.tolist(), endpoint_comparison_radius=end_radius.tolist(),
                                     active_branch=branch, comparison_diagonal_upper=diagonal.tolist(),
                                     comparison_offdiagonal_upper=majorant.tolist(),
                                     branch_gap_lower=float(branch_gap.lo), branch_gap_upper=float(branch_gap.hi)))
                if np.all(np.maximum(radius, end_radius) < trial):
                    margins = (geo["bounds"] - imat(geo["rows"], curve_box)
                               - imat(I(geo["constraint_box_map"]), I(trial)))
                    next_radius = (I(end_radius) + I(rounding_vector)).hi
                    return next_center, next_radius, dict(
                        left_h=float(left), right_h=float(right), defect_upper=defect,
                        initial_radius=np.asarray(radius).tolist(), radius=next_radius.tolist(),
                        apriori_radius=trial.tolist(), predictor_endpoint_rounding=rounding,
                        log_norm_upper=float(np.max((I(diagonal) + imat(I(majorant), I(np.ones(DIM)))).hi)),
                        comparison_method="componentwise diagonal-exponential self-inclusion",
                        active_branch=branch, branch_gap_interval=[float(branch_gap.lo), float(branch_gap.hi)],
                        continuous_constraint_margins_lower=margins.lo.tolist(), attempts=attempts)
                trial = np.maximum((I(trial) * I(1.1)).hi,
                                   (I(np.maximum(radius, end_radius)) * I(1.1) + I(1.e-12)).hi)
                if not np.all(np.isfinite(trial)) or max(trial) > 100.:
                    break
            raise EnclosureFailure("componentwise_apriori_inclusion_failed",
                                   dict(left_h=float(left), right_h=float(right), attempts=attempts))
        trial = max(1.e-12, float((I(1.1) * (I(radius) + I(h.hi) * I(forcing))).hi))
        attempts = []
        for iteration in range(14):
            physical_box = curve_box + I(-geo["coordinate_support"], geo["coordinate_support"]) * I(trial)
            branch, branch_gap = branch_for_box(physical_box, mode)
            metric_curve = (imat(L, I(center) - I(mode.center)) + imat(L, velocity) * tau
                            + imat(L, acceleration) * tau * tau)
            metric_box = metric_curve + I(-trial, trial)
            branch_log_norms = {}
            # Logarithmic norm is convex in the Jacobian. Max of the two
            # vertex bounds covers the generalized convex hull while avoiding
            # a much larger, artificial independent-entry interval hull.
            for smooth_branch in (("gate", "supply") if branch == "both" else (branch,)):
                transformed_jacobian = (evaluate_transformed_jacobian(
                    geo["jacobian_record"], metric_box, smooth_branch,
                    centered="exact" if self.jacobian_evaluation == "transformed_exact_centered" else self.jacobian_evaluation == "transformed_centered",
                    symmetric=True)
                    if self.jacobian_evaluation != "physical_interval" else
                    imat(L, imat(jacobian(physical_box, mode, smooth_branch), inverse)))
                symmetric = (transformed_jacobian if self.jacobian_evaluation != "physical_interval" else
                             (transformed_jacobian + transpose(transformed_jacobian)) * I(.5))
                branch_log_norms[smooth_branch] = symmetric_eigen_upper(symmetric)
            mu = max(branch_log_norms.values())
            ih = iv.mpf([float(h.lo), float(h.hi)])
            if mu == 0.:
                end_radius = float((iv.mpf(radius) + ih * iv.mpf(forcing)).b)
            else:
                factor = iv.exp(iv.mpf(mu) * ih)
                end_radius = float((factor * iv.mpf(radius) + iv.expm1(iv.mpf(mu) * ih) / iv.mpf(mu) * iv.mpf(forcing)).b)
            attempts.append(dict(trial_radius=trial, endpoint_comparison_radius=end_radius,
                                 log_norm_upper=mu, active_branch=branch,
                                 smooth_branch_log_norm_bounds=branch_log_norms,
                                 branch_gap_lower=float(branch_gap.lo), branch_gap_upper=float(branch_gap.hi)))
            if max(radius, end_radius) < trial:
                margins = geo["bounds"] - imat(geo["rows"], curve_box) - I(geo["support"]) * I(trial)
                next_radius = float((I(end_radius) + I(rounding)).hi)
                return next_center, next_radius, dict(
                    left_h=float(left), right_h=float(right), defect_upper=defect,
                    initial_radius=radius, radius=next_radius, apriori_radius=trial,
                    predictor_endpoint_rounding=rounding, log_norm_upper=mu,
                    active_branch=branch, branch_gap_interval=[float(branch_gap.lo), float(branch_gap.hi)],
                    continuous_constraint_margins_lower=margins.lo.tolist(), attempts=attempts)
            trial = max(float((I(trial) * I(1.6)).hi), float((I(end_radius) * I(1.15)).hi))
            if not math.isfinite(trial) or trial > 100.:
                break
        raise EnclosureFailure("apriori_inclusion_failed", dict(left_h=float(left), right_h=float(right),
                                                               predictor_defect_upper=defect, attempts=attempts))

    def verify_plan(self, initial_state, modes, durations_h=None, *, initial_time_h=0.,
                    amplitude=.3, duration=.5, pulse_start=.25, ramp=1 / 12,
                    residual_bound=.6, initial_box=.01, step_seconds=15.,
                    terminal_regional_upper=None, terminal_queue_upper=None,
                    initial_shape=None, retain_steps=False, service_checks=None,
                    timecuts_h=None) -> dict[str, Any]:
        started = time.perf_counter()
        if not modes or not math.isfinite(residual_bound) or residual_bound < 0 or not math.isfinite(step_seconds) or step_seconds <= 0:
            raise ValueError("nonempty plan, nonnegative residual, positive step required")
        selected = [self.library[int(mode)] if isinstance(mode, (int, np.integer)) else mode for mode in modes]
        durations = [Fraction(1, 60)] * len(selected) if durations_h is None else list(map(fraction, durations_h))
        if len(durations) != len(selected) or min(durations) <= 0:
            raise ValueError("positive duration required for every mode")
        center = np.asarray(initial_state, dtype=float)
        if center.shape != (DIM,) or not np.all(np.isfinite(center)):
            raise ValueError("finite seven-state initial center required")
        pulse = Pulse(amplitude, duration, pulse_start, ramp)
        first_geo = self.geometry(selected[0])
        if initial_shape is not None:
            if not np.array_equal(center, np.array(initial_shape["center"])):
                raise ValueError("initial_shape center must match initial_state exactly")
            source_radius = np.asarray(initial_shape["radius"], dtype=float)
            if np.any(source_radius < 0) or not np.all(np.isfinite(source_radius)):
                raise ValueError("finite nonnegative initial metric radius required")
            if source_radius.shape not in ((), (DIM,)):
                raise ValueError("initial radius must be scalar or seven componentwise radii")
            source_L = np.asarray(initial_shape["metric_L"], dtype=float)
            if source_L.shape != (DIM, DIM) or not np.all(np.isfinite(source_L)):
                raise ValueError("finite seven-dimensional initial metric matrix required")
            radius = self.switch_radius(source_radius, source_L, selected[0])
        else:
            widths = np.broadcast_to(np.asarray(initial_box, dtype=float), (DIM,))
            if np.any(widths < 0) or not np.all(np.isfinite(widths)):
                raise ValueError("finite nonnegative initial box widths required")
            initial_transformed = imat(first_geo["L"], I(-widths, widths))
            radius = initial_transformed.mag() if self.is_box else norm_upper(initial_transformed)
        initial_radius = radius
        current_L = first_geo["L"]
        cursor = fraction(initial_time_h)
        final_time = cursor + sum(durations)
        requested_checks = []
        for check in service_checks or []:
            check_time = fraction(check["time_h"])
            if not cursor <= check_time <= final_time:
                raise ValueError("requested service check lies outside the prescribed plan horizon")
            coordinate = check.get("coordinate", 0)
            if isinstance(coordinate, str):
                coordinate = model.STATE_ORDER.index(coordinate)
            if not 0 <= int(coordinate) < DIM or not math.isfinite(float(check["upper"])):
                raise ValueError("service coordinate and finite upper bound required")
            requested_checks.append((check_time, int(coordinate), float(check["upper"])))
        extra_cuts = [fraction(value) for value in (timecuts_h or [])]
        extra_cuts += [item[0] for item in requested_checks]
        service_results = []
        proof_descriptor = dict(initial_state=center.tolist(), initial_radius=np.asarray(radius).tolist(),
                                initial_L=current_L.tolist(), initial_time=str(cursor),
                                durations=[str(value) for value in durations],
                                mode_fingerprints=[self.geometry(mode)["fingerprint"] for mode in selected],
                                pulse=[str(pulse.amplitude), str(pulse.duration), str(pulse.start), str(pulse.ramp)],
                                residual_bound=residual_bound, metric_strategy=self.metric_strategy,
                                step_seconds=step_seconds, timecuts=[str(value) for value in extra_cuts],
                                service_checks=[dict(exact_time=str(at), coordinate=coordinate, upper=upper)
                                                for at, coordinate, upper in requested_checks],
                                terminal_regional_upper=None if terminal_regional_upper is None else
                                np.broadcast_to(terminal_regional_upper, (2,)).tolist(),
                                terminal_queue_upper=terminal_queue_upper,
                                model_contract=dict(equations="seven_state_class_conservation_with_min_admission_v1",
                                                    state_order=list(model.STATE_ORDER),
                                                    MFD_coefficients=[model.original.MFD_A, model.original.MFD_B, model.original.MFD_C],
                                                    internal_demand=model.INTERNAL_DEMAND.tolist(),
                                                    admission_capacity=model.CAPACITY, release_rate=model.RELEASE_RATE,
                                                    base_arrival=model.BASE_ARRIVAL, queue_limit=model.QUEUE_LIMIT,
                                                    regional_jam_limits=[100., 100.], command_box=[0., 1.],
                                                    directional_limit=.75, admission_split=.5),
                                affine_constraint_contracts=[dict(mode_fingerprint=self.geometry(mode)["fingerprint"],
                                                                 rows_lower=self.geometry(mode)["rows"].lo.tolist(),
                                                                 rows_upper=self.geometry(mode)["rows"].hi.tolist(),
                                                                 bounds_lower=self.geometry(mode)["bounds"].lo.tolist(),
                                                                 bounds_upper=self.geometry(mode)["bounds"].hi.tolist())
                                                            for mode in selected])
        proof_id = hashlib.sha256(json.dumps(proof_descriptor, sort_keys=True).encode("utf-8")).hexdigest()
        checkpoints = []

        def checkpoint(at, current_geo, segment):
            checkpoints.append(dict(exact_time_fraction=str(at), time_h=float(at), center=center.tolist(),
                                    metric_L=current_geo["L"].tolist(), radius=np.asarray(radius).tolist(),
                                    vector_radius=np.asarray(radius).tolist() if np.asarray(radius).ndim else None,
                                    shape_kind="componentwise_box" if np.asarray(radius).ndim else "euclidean_ball",
                                    segment=segment, mode_fingerprint=current_geo["fingerprint"], proof_id=proof_id))

        checkpoint(cursor, first_geo, -1)

        def record_service(at, current_geo):
            for check_time, coordinate, limit in requested_checks:
                if at == check_time:
                    upper = float((I(center[coordinate]) + self.coordinate_error(current_geo, radius)[coordinate]).hi)
                    service_results.append(dict(time_h=float(at), coordinate=coordinate,
                                                coordinate_name=model.STATE_ORDER[coordinate],
                                                upper=upper, limit=limit, passed=bool(upper <= limit)))

        record_service(cursor, first_geo)
        failures, segments, steps = [], [], []
        minimum_margins = np.full(len(first_geo["names"]), np.inf)
        maximum_mu, maximum_defect, maximum_radius = -np.inf, 0., float(np.max(radius))
        branch_counts = {"gate": 0, "supply": 0, "both": 0}
        switch_records = []
        inflation_audits = []
        completed_steps = 0
        last_safe_time = cursor
        for index, (mode, segment_duration) in enumerate(zip(selected, durations)):
            geo = self.geometry(mode)
            old_radius = radius
            radius = self.switch_radius(radius, current_L, mode)
            current_L = geo["L"]
            switch_records.append(dict(segment=index, time_h=float(cursor), from_radius=np.asarray(old_radius).tolist(),
                                       to_radius=np.asarray(radius).tolist(), mode=mode.name))
            endpoint = cursor + segment_duration
            cuts = [cursor] + sorted(set(cut for cut in pulse.knots + extra_cuts if cursor < cut < endpoint)) + [endpoint]
            for start, end in zip(cuts[:-1], cuts[1:]):
                pieces = max(1, math.ceil(float((end - start) * 3600) / step_seconds))
                grid = [start + (end - start) * Fraction(k, pieces) for k in range(pieces + 1)]
                for left, right in zip(grid[:-1], grid[1:]):
                    try:
                        center, radius, info = self.step(center, radius, mode, left, right, pulse, residual_bound)
                    except EnclosureFailure as exc:
                        failures.append(dict(kind=exc.kind, segment=index, mode=mode.name, **exc.record))
                        break
                    completed_steps += 1
                    cursor = right
                    minimum_margins = np.minimum(minimum_margins, info["continuous_constraint_margins_lower"])
                    maximum_mu = max(maximum_mu, info["log_norm_upper"])
                    maximum_defect = max(maximum_defect, info["defect_upper"])
                    maximum_radius = max(maximum_radius, float(np.max(info["apriori_radius"])))
                    branch_counts[info["active_branch"]] += 1
                    if len(info["attempts"]) > 1:
                        inflation_audits.append(dict(segment=index, time_h=float(cursor), attempts=info["attempts"]))
                    record_service(cursor, geo)
                    if cursor in extra_cuts or cursor == endpoint:
                        checkpoint(cursor, geo, index)
                    if retain_steps:
                        steps.append(dict(segment=index, mode=mode.name, **info))
                    if min(info["continuous_constraint_margins_lower"]) < 0:
                        binding = int(np.argmin(info["continuous_constraint_margins_lower"]))
                        failures.append(dict(kind="continuous_physical_constraint_not_certified", segment=index,
                                             time_h=float(cursor), mode=mode.name, constraint=geo["names"][binding],
                                             margin_lower=info["continuous_constraint_margins_lower"][binding],
                                             active_branch=info["active_branch"], failed_step=info))
                        break
                    last_safe_time = cursor
                if failures:
                    break
            support = self.coordinate_error(geo, radius)
            final_box = I(center) + I(-support.hi, support.hi)
            segments.append(dict(segment=index, mode=mode.name, time_h=float(cursor), center=center.tolist(),
                                 radius=np.asarray(radius).tolist(), coordinate_lower=final_box.lo.tolist(), coordinate_upper=final_box.hi.tolist()))
            if failures:
                break
        final_geo = self.geometry(selected[min(len(segments) - 1, len(selected) - 1)])
        support = self.coordinate_error(final_geo, radius)
        final_box = I(center) + I(-support.hi, support.hi)
        physical_ok = not failures
        regional_limits = None if terminal_regional_upper is None else np.broadcast_to(terminal_regional_upper, (2,)).astype(float)
        terminal_checks = {}
        if regional_limits is not None:
            for index in range(2):
                terminal_checks[f"n{index + 1}"] = dict(upper=float(final_box.hi[index]), limit=float(regional_limits[index]),
                                                        passed=bool(final_box.hi[index] <= regional_limits[index]))
        if terminal_queue_upper is not None:
            terminal_checks["exterior_queue"] = dict(upper=float(final_box.hi[6]), limit=float(terminal_queue_upper),
                                                      passed=bool(final_box.hi[6] <= terminal_queue_upper))
        terminal_ok = physical_ok and all(item["passed"] for item in terminal_checks.values())
        if physical_ok and not terminal_ok:
            failures.append(dict(kind="terminal_service_not_certified", checks=terminal_checks))
        service_ok = (physical_ok and len(service_results) == len(requested_checks)
                      and all(item["passed"] for item in service_results))
        if physical_ok and not service_ok:
            failures.append(dict(kind="scheduled_service_not_certified", checks=service_results))
        return dict(verified=physical_ok and terminal_ok and service_ok,
                    accepted=physical_ok and terminal_ok and service_ok, verified_continuous_physical=physical_ok,
                    verified_terminal_service=terminal_ok, proof_scope="finite prescribed plan only; no ODE tolerance used",
                    verified_scheduled_service=service_ok, service_check_results=service_results,
                    jacobian_evaluation=self.jacobian_evaluation,
                    metric_strategy=self.metric_strategy,
                    proof_id=proof_id, proof_descriptor=proof_descriptor, reachable_checkpoints=checkpoints,
                    metric_selection_audits={mode.name: self.geometry(mode)["metric_selection"] for mode in selected},
                    modes=[mode.name for mode in selected], durations_h=[float(value) for value in durations],
                    initial_time_h=float(fraction(initial_time_h)), certified_until_h=float(last_safe_time),
                    enclosed_until_h=float(cursor),
                    initial_metric_radius=np.asarray(initial_radius).tolist(), residual_bound_units_per_hour=residual_bound,
                    initial_box_units=np.broadcast_to(initial_box, (DIM,)).tolist() if initial_shape is None else None,
                    arrival=dict(amplitude=amplitude, duration_h=duration, start_h=pulse_start, ramp_h=ramp,
                                 exact_knot_fractions=[str(value) for value in pulse.knots]),
                    completed_interval_steps=completed_steps, branch_counts=branch_counts,
                    maximum_log_norm_upper=None if not math.isfinite(maximum_mu) else maximum_mu,
                    maximum_predictor_defect_upper=maximum_defect, maximum_metric_radius=maximum_radius,
                    continuous_constraint_margins_lower={name: float(value) if math.isfinite(value) else None
                                                        for name, value in zip(first_geo["names"], minimum_margins)},
                    terminal_coordinate_intervals={name: [float(final_box.lo[index]), float(final_box.hi[index])]
                                                   for index, name in enumerate(model.STATE_ORDER)},
                    terminal_checks=terminal_checks,
                    terminal_shape=dict(center=center.tolist(), metric_L=current_L.tolist(), radius=np.asarray(radius).tolist(),
                                        vector_radius=np.asarray(radius).tolist() if np.asarray(radius).ndim else None,
                                        exact_time_fraction=str(cursor),
                                        time_h=float(cursor), shape=("{x: abs(metric_L (x-center)) <= radius componentwise}"
                                                                   if np.asarray(radius).ndim else "{x: ||metric_L (x-center)||_2 <= radius}")),
                    failures=failures, segments=segments, mode_change_radius_audits=switch_records,
                    apriori_radius_inflation_audits=inflation_audits,
                    interval_steps=steps if retain_steps else None, wall_seconds=time.perf_counter() - started)


class EnclosureFailure(Exception):
    def __init__(self, kind: str, record: dict):
        self.kind, self.record = kind, record
        super().__init__(kind)


_DEFAULT_VERIFIERS = {}


def verify_plan(initial_state, modes, durations_h=None, *, metric_strategy="conservation_box", **kwargs):
    if metric_strategy not in _DEFAULT_VERIFIERS:
        _DEFAULT_VERIFIERS[metric_strategy] = PlanVerifier(metric_strategy=metric_strategy)
    return _DEFAULT_VERIFIERS[metric_strategy].verify_plan(initial_state, modes, durations_h, **kwargs)


def contains_information_box(shape, information_center, half_width=.01):
    """Outward check of a measurement box inside a retained reachable shape.

    A caller may retain a suffix only when its parent proof was accepted and its
    mode/clock/arrival assumptions remain unchanged. This function checks set
    inclusion only; it does not establish those execution-binding conditions.
    """
    center = np.asarray(information_center, dtype=float)
    L = np.asarray(shape["metric_L"], dtype=float)
    widths = np.broadcast_to(np.asarray(half_width, dtype=float), (DIM,))
    if center.shape != (DIM,) or L.shape != (DIM, DIM) or np.any(widths < 0):
        raise ValueError("seven-state center, metric, and nonnegative widths required")
    if not all(np.all(np.isfinite(value)) for value in (center, L, widths)):
        raise ValueError("finite information box required")
    transformed = imat(L, I(center) - I(np.asarray(shape["center"])))
    support = imat(I(abs(L)), I(widths)).hi
    bound = (I(transformed.mag()) + I(support)).hi
    radius = np.asarray(shape["radius"], dtype=float)
    passed = bool(np.all(bound <= radius)) if radius.ndim else bool(norm_upper(I(bound)) <= float(radius))
    return dict(passed=passed, transformed_absolute_error_upper=bound.tolist(),
                radius=radius.tolist(), shape_kind="componentwise_box" if radius.ndim else "euclidean_ball")


PROOF_NOTE = {
    "coordinates": "For conservation_box, y=Lx=(n11,n12,n10,n22+z/2,n21+z/2,n20,z). The admission incidence b=(0,1,0,1/2,0,0,-1) obeys Lb=-e7 exactly.",
    "predictor": "On every exact arrival-affine subinterval, choose an arbitrary binary-coefficient quadratic physical predictor p(t). Outward polynomial arithmetic bounds every coefficient of L f_branch(p(t),lambda_nom(t))-L p'(t), of degree at most eight. If p crosses the min branch, take the maximum componentwise residual magnitude over both smooth branches. Add |L e7|*eta for every measurable residual |w(t)|<=eta.",
    "jacobian": "On the candidate tube p(t)+L^{-1}[-R,R], interval polynomials enclose transformed Jacobians J=L Df_branch L^{-1}. At a min kink every generalized Jacobian is a convex combination of the two vertex Jacobians. Set a_i=max_branch sup J_ii and b_ij=max_branch sup |J_ij| for j!=i; b_ij>=0. These majorize every generalized Jacobian. For the ball alternative, convexity permits max of vertex logarithmic norm bounds; taking a componentwise matrix hull is unnecessary.",
    "comparison": "Before the first tube exit, v_i=|(L(x-p))_i| has upper Dini derivative D+v_i <= a_i v_i + sum_{j!=i} b_ij R_j + eta_i + defect_i. Let c_i(R) be the nonnegative last three terms. The scalar comparison solution is g_i(t)=exp(a_i*t)*r_i+phi(a_i,t)*c_i(R), with phi=(exp(a_i*t)-1)/a_i and phi(0,t)=t.",
    "self_inclusion": "Each scalar g_i is monotone on the step because its derivative has constant sign. Outward interval exponentials certify max(r_i,g_i(h))<R_i for every component. A first-exit contradiction therefore proves the whole continuous trajectory stays inside the trial tube and v_i(h)<=g_i(h). Endpoint predictor rounding is added componentwise with outward arithmetic. This is a finite-time validated integration argument, not an ODE solver error tolerance.",
    "outputs_and_constraints": "For every physical affine row a*x<=b, a*x is at most sup_t(a*p(t))+|a*L^{-1}|*R; interval lower slack must be nonnegative throughout every accepted step. Queue limit, state/class positivity, jam bounds, command boxes, and directional command limits are all checked. Endpoint coordinate bounds use |L^{-1}|*r. Scheduled service checks force exact rational time cuts and apply the same output support bound.",
    "switches_and_suffixes": "Plant states are continuous at mode changes. In fixed conservation coordinates the vector radius is unchanged; in general it transforms outward by |L_dest*L_source^{-1}|*r. Checkpoint shapes bind to a deterministic proof identifier, exact time, installed mode fingerprint, and the retained plan. Measurement-box inclusion uses |L*(xhat-center)|+|L|*width<=r componentwise. A retained suffix requires an accepted parent proof and unchanged exogenous/clock/plan assumptions.",
    "arithmetic_scope": "Binary floating-point installed coefficients define the mathematical plant. Arithmetic uses nextafter-outward sums/products and interval exponentials. Numerical inverses receive a verified Neumann correction; eigensolver bases are interval Gram-verified when used. Numerical metrics and predictor coefficients only choose coordinates/curves and are not accepted as certificates by their solver tolerances.",
    "limitations": "No infinite-time safety, recursive feasibility, incident capacity-loss model, parameter-calibration uncertainty, or optimality follows. A failed bound is not a physical counterexample. Every failed plan and its diagnostic trial bounds is retained. The ordinary class-coordinate box also passes the primary case; conservation sums improve sharpness but are not necessary for that pass.",
}


def implementation_self_checks(verifier, mode):
    """Numerical regression checks only; these do not replace interval proofs."""
    geo = verifier.geometry(mode)
    maximum_containment_excess = -np.inf
    for alpha in (0., .5, 1.):
        physical = alpha * model.BASE_STATE + (1 - alpha) * mode.center
        for queue in (.8, 2.):
            physical[6] = queue
            metric_point = geo["L"] @ (physical - mode.center)
            numerical_inverse = np.linalg.inv(geo["L"])
            for branch in ("gate", "supply"):
                interval_matrix = evaluate_transformed_jacobian(geo["jacobian_record"],
                                                                 I(metric_point - 1.e-8, metric_point + 1.e-8), branch,
                                                                 centered="exact")
                physical_jacobian = model.original.complex_step_jacobian(
                    lambda state: np.asarray(closed_field(list(state), mode, 60., branch)), physical)
                numerical = geo["L"] @ physical_jacobian @ numerical_inverse
                excess = max(float(np.max(interval_matrix.lo - numerical)),
                             float(np.max(numerical - interval_matrix.hi)))
                maximum_containment_excess = max(maximum_containment_excess, excess)
    assert maximum_containment_excess <= 0.
    scalar_to_box = verifier.switch_radius(.1, geo["L"], mode)
    assert np.array_equal(scalar_to_box, np.full(DIM, .1))
    common = dict(initial_state=model.BASE_STATE, modes=[mode], durations_h=[Fraction(1, 60)],
                  duration=1., initial_box=.01, residual_bound=.6)
    loose = verifier.verify_plan(**common, service_checks=[dict(time_h=0., coordinate=0, upper=31.)])
    tight = verifier.verify_plan(**common, service_checks=[dict(time_h=0., coordinate=0, upper=29.)])
    assert loose["accepted"] and not tight["accepted"] and loose["proof_id"] != tight["proof_id"]
    rejected_outside = False
    try:
        verifier.verify_plan(**common, service_checks=[dict(time_h=1., coordinate=0, upper=31.)])
    except ValueError:
        rejected_outside = True
    assert rejected_outside
    return dict(scope="implementation regression checks, not sampled proof", smooth_branch_jacobian_points=12,
                maximum_interval_containment_excess=maximum_containment_excess,
                same_L_ball_to_box_shape_conversion=True, service_contract_changes_proof_id=True,
                failed_service_retains_continuous_physical_audit=tight["verified_continuous_physical"],
                out_of_horizon_service_rejected=rejected_outside)


def run_commissioned_pilot():
    from mfd_incident_library_design import synthesize
    library = [synthesize(20, 22, 72, 3000), synthesize(22, 24, 78, 3000), synthesize(24, 26, 84, 3000)]
    verifier = PlanVerifier(library, metric_strategy="conservation_box")
    cases = []

    def run(name, active, schedule, durations, **kwargs):
        result = active.verify_plan(model.BASE_STATE, schedule, durations, duration=1.,
                                    residual_bound=.6, initial_box=.01, step_seconds=15., **kwargs)
        result["case_name"] = name
        cases.append(result)
        print(json.dumps(dict(case=name, accepted=result["accepted"], until=result["certified_until_h"],
                              seconds=result["wall_seconds"], service=result["service_check_results"],
                              failures=[failure["kind"] for failure in result["failures"]])), flush=True)
        return result

    primary = None
    for minutes in (15, 30, 60, 120):
        checks = [dict(time_h=1., coordinate=0, upper=28.5)] if minutes >= 60 else []
        result = run(f"commissioned_mode0_{minutes}min", verifier, [0], [Fraction(minutes, 60)],
                     service_checks=checks, timecuts_h=[Fraction(k, 12) for k in range(1, minutes // 5 + 1)])
        if minutes == 60:
            primary = result
    for index in (1, 2):
        run(f"commissioned_mode{index}_60min", verifier, [index], [Fraction(1)],
            service_checks=[dict(time_h=1., coordinate=0, upper=28.5)])
        run(f"commissioned_mode{index}_5min_then_mode0_55min", verifier, [index, 0],
            [Fraction(1, 12), Fraction(11, 12)], service_checks=[dict(time_h=1., coordinate=0, upper=28.5)])
    ablations = []
    for strategy in ("physical_box", "class_box", "installed", "common_branch"):
        active = PlanVerifier(library, metric_strategy=strategy)
        result = run(f"same_case_ablation_{strategy}", active, [0], [Fraction(1)],
                     service_checks=[dict(time_h=1., coordinate=0, upper=28.5)])
        ablations.append(dict(strategy=strategy, accepted=result["accepted"],
                              service=result["service_check_results"], case_name=result["case_name"]))
    ablations.append(dict(strategy="conservation_box", accepted=primary["accepted"],
                          service=primary["service_check_results"], case_name=primary["case_name"]))
    checkpoint = next(item for item in primary["reachable_checkpoints"] if item["exact_time_fraction"] == "1/2")
    continuation = verifier.verify_plan(checkpoint["center"], [0], [Fraction(1, 2)], initial_time_h=Fraction(1, 2),
                                        initial_shape=checkpoint, duration=1., residual_bound=.6,
                                        service_checks=[dict(time_h=1., coordinate=0, upper=28.5)])
    continuation["case_name"] = "reachable_box_continuation_without_radius_reset"
    cases.append(continuation)
    prior = None
    if OUTPUT.exists():
        prior = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if prior.get("status") == "commissioned_operating_library_interval_pilot":
            prior = prior.get("previous_high_target_pilot")
    return dict(status="commissioned_operating_library_interval_pilot", proof_note=PROOF_NOTE,
                configuration=dict(primitive_demand=model.INTERNAL_DEMAND.tolist(), actual_pulse_amplitude=.3,
                                   actual_pulse_duration_h=1., residual_bound_units_per_hour=.6,
                                   initial_box_half_width_units=.01, max_step_seconds=15.,
                                   library=[mode.record() for mode in library]),
                cases=cases, coordinate_ablation=ablations,
                implementation_self_checks=implementation_self_checks(verifier, library[0]),
                checkpoint_information_box_self_test=contains_information_box(checkpoint, checkpoint["center"], .01),
                previous_high_target_pilot=prior,
                summary=dict(cases=len(cases), accepted=sum(case["accepted"] for case in cases),
                             failures_retained=sum(not case["accepted"] for case in cases)))


def run_pilot(quick=False) -> dict:
    verifier = PlanVerifier()
    cases = []
    horizons = (15.,) if quick else (15., 20., 30., 60.)
    for minutes in horizons:
        for mode in range(3):
            case = verifier.verify_plan(model.BASE_STATE, [mode], [minutes / 60.],
                                        residual_bound=.6, initial_box=.01, step_seconds=15.,
                                        terminal_regional_upper=(35., 35.), terminal_queue_upper=8.)
            cases.append(case)
            print(json.dumps(dict(minutes=minutes, mode=mode, verified=case["verified"],
                                  wall_seconds=case["wall_seconds"], failures=case["failures"],
                                  terminal=case["terminal_coordinate_intervals"])), flush=True)
    if not quick:
        for schedule in ([0] * 20, [0] * 10 + [1] * 10, [1] * 10 + [0] * 10):
            case = verifier.verify_plan(model.BASE_STATE, schedule, residual_bound=.6, initial_box=.01,
                                        step_seconds=15., terminal_regional_upper=(35., 35.), terminal_queue_upper=8.)
            cases.append(case)
            print(json.dumps(dict(plan=case["modes"], verified=case["verified"], wall_seconds=case["wall_seconds"],
                                  failures=case["failures"])), flush=True)
    return dict(status="outward_interval_finite_plan_pilot", cases=cases,
                summary=dict(cases=len(cases), verified=sum(case["verified"] for case in cases)),
                limitations=["Installed binary floating coefficients define the mathematical plant; coefficient/calibration uncertainty is not included.",
                             "Only offered-arrival residual uncertainty is included, not incident MFD capacity degradation.",
                             "All accepted steps use strict self-inclusion and nonnegative continuous affine margins.",
                             "Failure means this enclosure did not certify the plan, not proof the physical plan is unsafe.",
                             "Finite-plan terminal service bounds do not establish recursive feasibility or safety beyond the certified endpoint."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--commissioned", action="store_true")
    args = parser.parse_args()
    results = run_commissioned_pilot() if args.commissioned else run_pilot(args.quick)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
