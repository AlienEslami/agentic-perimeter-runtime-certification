"""Independent high-precision diagnostics for the affine matrix primitives.

These numerical comparisons supplement the analytic norm-tail argument. They
do not certify a matrix function by sampling or replace the interval backend.
No source or scientific result files are modified.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from mpmath import mp, iv
from mfd_incident_verifier import interval_fraction


def run(module_path=None):
    path = Path(module_path) if module_path else Path(__file__).with_name('mfd_incident_affine_reachability.py')
    spec = importlib.util.spec_from_file_location('affine_matrix_review_target', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(714911)
    cases = []
    with mp.workdps(90):
        for case in range(6):
            matrix = rng.normal(0, 5, (7, 7))-np.diag(rng.uniform(0, 60, 7))
            if case >= 3:
                matrix = np.abs(matrix)
                np.fill_diagonal(matrix, -rng.uniform(0, 60, 7))
            h = Fraction(1, 240)
            exponential, integral, _ = module.exponential_and_integral(matrix, interval_fraction(h))
            exact_time = mp.mpf(h.numerator)/h.denominator
            exact_matrix = mp.matrix(matrix.tolist())
            augmented = mp.matrix(14)
            for i in range(7):
                for j in range(7):
                    augmented[i, j] = exact_matrix[i, j]
                augmented[i, i+7] = 1
            reference = mp.expm(augmented*exact_time)
            for i in range(7):
                for j in range(7):
                    assert mp.mpf(float(exponential.lo[i, j])) <= reference[i, j] <= mp.mpf(float(exponential.hi[i, j]))
                    assert mp.mpf(float(integral.lo[i, j])) <= reference[i, j+7] <= mp.mpf(float(integral.hi[i, j]))
            rows, generators = rng.normal(0, 2, (3, 7)), rng.normal(0, .2, (7, 7))
            flow, _ = module.signed_flow_range(rows, matrix, generators, interval_fraction(h))
            exact_rows, exact_generators = mp.matrix(rows.tolist()), mp.matrix(generators.tolist())
            for fraction in (mp.mpf(0), mp.mpf(1)/7, mp.mpf(1)/2, mp.mpf(1)):
                reference = exact_rows*mp.expm(exact_matrix*exact_time*fraction)*exact_generators
                for i in range(3):
                    for j in range(7):
                        assert mp.mpf(float(flow.lo[i, j])) <= reference[i, j] <= mp.mpf(float(flow.hi[i, j]))
            cases.append(dict(case=case, metzler=case >= 3, exact_time_h=str(h),
                              endpoint_exponential_and_integral=True, signed_range_time_points=4))
    return dict(passed=True, source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference_decimal_digits=90, iv_precision=iv.prec, matrix_cases=len(cases),
                matrix_entry_checks=6*(98+4*21), signed_time_points=24, cases=cases,
                scope='independent numerical diagnostics; analytical tail proof remains the certificate basis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-path', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.module_path), indent=2))
