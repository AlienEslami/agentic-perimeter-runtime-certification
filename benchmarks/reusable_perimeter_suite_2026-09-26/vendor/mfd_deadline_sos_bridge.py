#!/usr/bin/env python3
"""Auditable bridge from the rounded SOS polynomial to the exact binary MFD.

Original numerical records are read, never changed.  The two fixed SOS levels
are re-solved and the returned Gram witnesses are independently interval
verified.  Exact-binary field coefficients are then enclosed and compared to
the exact binary polynomial used in those fresh SOS solves.  The resulting
roundoff forcing and rate correction close the near-equilibrium degree-one
residual that homogeneous nominal claims cannot absorb.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
from mpmath import iv

import mfd_cubic_nonstationary_study as study
import mfd_cubic_pilot as plant
import mfd_cubic_sos_pilot as sos
from mfd_deadline_certificates import I, Poly, field, imat, isum


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_deadline_sos_bridge_results.json"
ZERO = (0,) * 6


def polynomial_sum(items):
    total = Poly({})
    for item in items:
        total = total + item
    return total


def eigen_upper(matrix):
    """Generalized Gershgorin; handles the 6-, 27-, and 83-state matrices."""
    matrix = I(matrix)
    dimension = matrix.lo.shape[0]
    _, basis = np.linalg.eigh(matrix.mid())
    rotated = imat(basis.T, imat(matrix, basis))
    gram = imat(basis.T, basis)
    numerator = max(float((I(rotated[k, k].hi) + isum(
        I(rotated[k, j].mag()) for j in range(dimension) if j != k
    )).hi) for k in range(dimension))
    denominator_lower = min(float((I(gram[k, k].lo) - isum(
        I(gram[k, j].mag()) for j in range(dimension) if j != k
    )).lo) for k in range(dimension))
    denominator_upper = max(float((I(gram[k, k].hi) + isum(
        I(gram[k, j].mag()) for j in range(dimension) if j != k
    )).hi) for k in range(dimension))
    assert denominator_lower > 0
    denominator = denominator_lower if numerator >= 0 else denominator_upper
    return float((iv.mpf(numerator) / iv.mpf(denominator)).b)


def interval_gram_coefficients(matrix, basis):
    result = {}
    for k, left in enumerate(basis):
        for j, right in enumerate(basis):
            exponent = tuple(a + b for a, b in zip(left, right))
            result[exponent] = result.get(exponent, I(0.)) + I(matrix[k, j])
    return result


def add_interval_polynomials(left, right, scale=1.):
    result = left.copy()
    for exponent, value in right.items():
        result[exponent] = result.get(exponent, I(0.)) + value * I(scale)
    return result


def verify_sos(polynomial, proof_margin, solver="CLARABEL"):
    """Re-solve the unchanged fixed-level SOS and verify its actual witnesses."""
    multiplier_basis = sos.monomial_basis(2, min_degree=1)
    target_basis = sos.monomial_basis(3, min_degree=1)
    q_multiplier = cp.Variable((len(multiplier_basis), len(multiplier_basis)), symmetric=True)
    q_target = cp.Variable((len(target_basis), len(target_basis)), symmetric=True)
    sigma = sos.gram_coefficients(q_multiplier, multiplier_basis)
    target = sos.gram_coefficients(q_target, target_basis)
    unit_value = sos.quadratic_polynomial(np.eye(6))
    domain = sos.padd({ZERO: 1.}, sos.pscale(unit_value, -1.))
    product = sos.expression_product(sigma, domain)
    target_polynomial = sos.padd(polynomial, sos.pscale(unit_value, -proof_margin))
    exponents = set(target_polynomial) | set(product) | set(target)
    constraints = [q_multiplier >> 0., q_target >> 0.]
    constraints.extend(target_polynomial.get(key, 0.) - product.get(key, 0.) == target.get(key, 0.) for key in exponents)
    problem = cp.Problem(cp.Minimize(cp.trace(q_multiplier) + 1.e-9 * cp.trace(q_target)), constraints)
    started = time.perf_counter()
    sos.solve_problem(problem, solver, False)
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    multiplier = sos.symmetrize(np.asarray(q_multiplier.value))
    target_matrix = sos.symmetrize(np.asarray(q_target.value))
    multiplier_lower = -eigen_upper(-I(multiplier))
    target_lower = -eigen_upper(-I(target_matrix))
    sigma_interval = interval_gram_coefficients(multiplier, multiplier_basis)
    target_interval = interval_gram_coefficients(target_matrix, target_basis)
    product_interval = sigma_interval.copy()
    for coordinate in range(6):
        shift = [0] * 6
        shift[coordinate] = 2
        shifted = {tuple(a + b for a, b in zip(key, shift)): value for key, value in sigma_interval.items()}
        product_interval = add_interval_polynomials(product_interval, shifted, -1.)
    residual = {key: I(value) for key, value in polynomial.items()}
    residual = add_interval_polynomials(residual, {key: I(value) * I(proof_margin) for key, value in unit_value.items()}, -1.)
    residual = add_interval_polynomials(residual, product_interval, -1.)
    residual = add_interval_polynomials(residual, target_interval, -1.)
    assert all(sum(key) >= 2 for key in residual)
    residual_norm = float(isum(I(value.mag()) for value in residual.values()).hi)
    lower = float((I(proof_margin) +
        I(min(0., multiplier_lower)) * I(len(multiplier_basis)) +
        I(min(0., target_lower)) * I(len(target_basis)) - I(residual_norm)).lo)
    assert lower > 0, f"Interval SOS verification failed: {lower}"
    return dict(
        solver=solver, solver_status=problem.status,
        wall_seconds=time.perf_counter() - started,
        reserved_normalized_proof_margin=proof_margin,
        interval_multiplier_eigenvalue_lower=multiplier_lower,
        interval_target_eigenvalue_lower=target_lower,
        interval_coefficient_residual_one_norm_upper=residual_norm,
        interval_verified_normalized_lower_bound=lower,
        verified_on_unit_ball=True,
        multiplier_basis=[list(key) for key in multiplier_basis],
        target_basis=[list(key) for key in target_basis],
        multiplier_gram=multiplier.tolist(), target_gram=target_matrix.tolist(),
        coefficient_residuals=[dict(exponent=list(key), lower=float(value.lo), upper=float(value.hi)) for key, value in sorted(residual.items())],
    )


def mode_bridge(mode, gain, matrix, rate, level, old_record):
    values, vectors = np.linalg.eigh(matrix)
    inverse_sqrt = vectors @ np.diag(1. / np.sqrt(values)) @ vectors.T
    coordinate_map = math.sqrt(level) * inverse_sqrt
    numerical = sos.pscale(sos.linear_substitution(
        sos.dissipation_polynomial(gain, matrix, rate), coordinate_map), 1. / level)
    assert numerical.get(ZERO, 0.) == 0
    variables = []
    for coordinate in range(6):
        exponent = [0] * 6
        exponent[coordinate] = 1
        variables.append(Poly({tuple(exponent): I(1.)}))
    errors = [polynomial_sum(variables[j] * I(coordinate_map[k, j]) for j in range(6)) for k in range(6)]
    physical = [errors[k] + I(plant.X_STAR[k]) for k in range(6)]
    exact_field = field(physical, gain)
    symmetric = (I(matrix) + I(matrix.T)) * I(.5)
    value = polynomial_sum(errors[k] * polynomial_sum(errors[j] * symmetric[k, j] for j in range(6)) for k in range(6))
    gradient = [polynomial_sum(errors[j] * symmetric[k, j] * I(2.) for j in range(6)) for k in range(6)]
    derivative = polynomial_sum(gradient[k] * exact_field[k] for k in range(6))
    reciprocal = iv.mpf(1.) / iv.mpf(level)
    exact = (-derivative - value * I(rate)) * I(float(reciprocal.a), float(reciprocal.b))
    # There is no constant term: V and its gradient are formed from homogeneous
    # degree-two/degree-one expressions, with no numerically inferred pruning.
    assert ZERO not in exact.terms
    keys = sorted(set(exact.terms) | set(numerical))
    discrepancies = {key: exact.terms.get(key, I(0.)) - I(numerical.get(key, 0.)) for key in keys}
    tau1 = float(isum(I(value.mag()) for key, value in discrepancies.items() if sum(key) == 1).hi)
    tau2 = float(isum(I(value.mag()) for key, value in discrepancies.items() if sum(key) >= 2).hi)
    gram = imat(coordinate_map.T, imat(symmetric, coordinate_map))
    gmin = -eigen_upper(-gram)
    gmax = eigen_upper(gram)
    assert gmin > 0
    reduction = float((iv.mpf(level) * iv.mpf(tau2) / iv.mpf(gmin)).b)
    gamma_roundoff = float((iv.mpf(level) * iv.mpf(tau1) / (2 * iv.sqrt(iv.mpf(gmin)))).b)
    effective_rate = float((I(rate) - I(reduction)).lo)
    assert effective_rate > 0
    witness = verify_sos(numerical, old_record['reserved_normalized_proof_margin'])
    coefficients = [dict(exponent=list(key), degree=sum(key), numerical=float(numerical.get(key, 0.)),
        exact_lower=float(exact.terms.get(key, I(0.)).lo), exact_upper=float(exact.terms.get(key, I(0.)).hi),
        discrepancy_lower=float(discrepancies[key].lo), discrepancy_upper=float(discrepancies[key].hi)) for key in keys]
    hash_payload = json.dumps([[list(key), value] for key, value in sorted(numerical.items())], separators=(',', ':')).encode()
    return dict(
        mode=mode + 1, original_individual_level=level,
        original_rate_per_hour=rate, coordinate_map=coordinate_map.tolist(),
        coordinate_gram_eigenvalue_lower=gmin, coordinate_gram_eigenvalue_upper=gmax,
        degree_one_coefficient_discrepancy_sum_upper=tau1,
        degree_two_and_higher_coefficient_discrepancy_sum_upper=tau2,
        rate_reduction_upper_per_hour=reduction, effective_rate_lower_per_hour=effective_rate,
        certificate_radius_roundoff_forcing_upper_per_hour=gamma_roundoff,
        exact_constant_term_structurally_zero=True,
        regenerated_numerical_polynomial_sha256=hashlib.sha256(hash_payload).hexdigest(),
        original_sos_summary={key: old_record[key] for key in (
            'minimum_gram_eigenvalue', 'multiplier_minimum_eigenvalue', 'target_minimum_eigenvalue',
            'maximum_coefficient_residual', 'coefficient_residual_one_norm', 'reserved_normalized_proof_margin',
            'verified_normalized_lower_bound', 'a_posteriori_residual_bound_passed')},
        fresh_interval_sos_witness=witness, coefficient_comparison=coefficients,
    )


def main():
    started = time.perf_counter()
    gains, certificate, tube = study.load_installed()
    old = json.loads((ROOT / 'reproducibility/results/mfd_cubic_sos_pilot_results.json').read_text())
    deadline_level = float((I(tube['common_level']) * (I(1.) - I(1.e-8))).lo)
    records = []
    for mode in range(2):
        old_record = old['sos_certificates'][mode]
        record = mode_bridge(mode, gains[mode], certificate['matrices'][mode], certificate['rate'], old_record['level'], old_record)
        record['deadline_level_inside_verified_coordinate_domain'] = deadline_level <= record['coordinate_gram_eigenvalue_lower']
        assert record['deadline_level_inside_verified_coordinate_domain']
        records.append(record)
        print(json.dumps({key: record[key] for key in ('mode', 'coordinate_gram_eigenvalue_lower',
            'degree_one_coefficient_discrepancy_sum_upper', 'degree_two_and_higher_coefficient_discrepancy_sum_upper',
            'effective_rate_lower_per_hour', 'certificate_radius_roundoff_forcing_upper_per_hour')}), flush=True)
    result = dict(
        scope='Exact-binary coefficient bridge with independently re-solved and interval-verified fixed-level SOS witnesses',
        exact_model='Stored binary floats are interpreted as exact real coefficients; symmetry uses (P+P.T)/2',
        original_common_level=tube['common_level'], deadline_admission_level=deadline_level,
        deadline_level_shrink_fraction=1.e-8, modes=records,
        all_coefficient_discrepancies_recorded=True,
        all_fresh_sos_witnesses_interval_verified=True,
        wall_seconds=time.perf_counter() - started,
        flow_statement='D+sqrt(V_m) <= -(lambda_effective_m/2)*sqrt(V_m) + gamma_roundoff_m + sup_w ||Sym(P_m)^(1/2) E w|| on V_m <= coordinate_gram_lower_m',
    )
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Wrote {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
