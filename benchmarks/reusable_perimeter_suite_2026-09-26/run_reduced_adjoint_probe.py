"""Five-state exact quotient and interval-adjoint sign probe for fixed plans.

The quotient is exact when u20=1; the inherited seven-state physical tube
provides an outer box for its Jacobian. This probe does not certify service
unless all needed interval signs and a validated extremal value are obtained.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import time

import numpy as np
from mpmath import iv

from common import HERE, START, DURATIONS, fixture, save
from run_falsification import records
from run_interval_adjoint_probe import CaptureVerifier
from mfd_incident_affine_reachability import I, imat, norm_inf_upper, reciprocal, upward
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_verifier import interval_fraction

OUT = HERE / "results/directional_service_2026-10-01"


def reduced_jacobian(box, command, branch):
    assert command[3] == 1.0
    x, y, a, b, z = box[0], box[1], box[2], box[3]+box[5], box[6]
    u12, u21 = I(command[0]), I(command[1])
    h1 = I(.0014877)*x*x-I(.29815)*x+I(15.0912)
    h2 = I(.0014877)*y*y-I(.29815)*y+I(15.0912)
    hp1 = I(.0029754)*x-I(.29815)
    hp2 = I(.0029754)*y-I(.29815)
    p, q = x-a, y-b
    az = I(60.) if branch == "supply" else I(0.) if branch == "gate" else I(0., 60.)
    entries = [
        [-(a+u12*p)*hp1-u12*h1, u21*(h2+q*hp2), -(I(1.)-u12)*h1, -u21*h2, I(0.)],
        [u12*(h1+p*hp1), -u21*h2-(b+u21*q)*hp2, -u12*h1, -(I(1.)-u21)*h2, az],
        [-a*hp1, u21*(h2+q*hp2), -h1, -u21*h2, I(0.)],
        [u12*(h1+p*hp1), -b*hp2, -u12*h1, -h2, I(.5)*az],
        [I(0.), I(0.), I(0.), I(0.), -az],
    ]
    return I(np.array([[v.lo for v in row] for row in entries]),
             np.array([[v.hi for v in row] for row in entries]))


def series_flow(matrix, h, order=18):
    n = matrix.lo.shape[0]
    term = I(np.eye(n)); total = term
    for k in range(1, order+1):
        term = imat(matrix, term) * h * reciprocal(k)
        total = total + term
    z = iv.mpf(norm_inf_upper(matrix))*iv.mpf([float(h.lo), float(h.hi)])
    tail = upward((iv.exp(z)*z**(order+1)/math.factorial(order+1)).b)
    return total+I(-tail,tail)


class ReducedCapture(CaptureVerifier):
    def step(self, center, radius, mode, left, right, pulse, residual_bound):
        new_center, new_radius, info = super().step(
            center, radius, mode, left, right, pulse, residual_bound)
        box = I(info["captured_physical_lower"], info["captured_physical_upper"])
        matrix = reduced_jacobian(box, mode.feedforward, info["active_branch"])
        info["reduced_jacobian_lower"] = matrix.lo.tolist()
        info["reduced_jacobian_upper"] = matrix.hi.tolist()
        return new_center, new_radius, info


def run_case(path, record):
    c = record["contract"]
    commands = np.asarray(record["policies"]["full_scenario_mpc"]["commands"])
    assert np.all(commands[:, 3] == 1.)
    started = time.perf_counter()
    result = verify_command_plan(
        np.asarray(fixture()["center"]), commands, DURATIONS,
        initial_time_h=START, amplitude=c["amplitude"], duration=c["duration_h"],
        initial_box=c["width_vehicles"]/100,
        residual_bound=c["residual_veh_h"]/100, step_seconds=15,
        verifier=ReducedCapture(), retain_steps=True,
        service_checks=[dict(time_h=1, coordinate=0, upper=c["deadline_vehicles"]/100)],
    )
    steps = result["interval_steps"]
    assert result["verified_continuous_physical"] and len(steps) == 160
    row = I(np.eye(5)[0]); first_kernel = []; second_kernel = []
    h = interval_fraction(Fraction(1,240))
    for item in reversed(steps):
        j = I(item["reduced_jacobian_lower"],item["reduced_jacobian_upper"])
        partial = series_flow(j,I(0.,h.hi))
        k = imat(I(row.lo[None,:],row.hi[None,:]),partial)[0,4]
        (first_kernel if item["segment"] == 0 else second_kernel).append([float(k.lo),float(k.hi)])
        phi = series_flow(j,h)
        row = imat(I(row.lo[None,:],row.hi[None,:]),phi)[0]
    expected = (1,1,-1,-1,1)
    resolved = [bool(row.lo[i]>0 if s==1 else row.hi[i]<0) for i,s in enumerate(expected)]
    return dict(id=c["id"], quotient_states=["n1","n2","n11","n22_plus_n20","queue"],
                initial_gradient_lower=row.lo.tolist(),initial_gradient_upper=row.hi.tolist(),
                expected_signs=expected,initial_signs_resolved=resolved,
                first_arrival_kernel_lower=min(v[0] for v in first_kernel),
                first_arrival_kernel_upper=max(v[1] for v in first_kernel),
                second_arrival_kernel_lower=min(v[0] for v in second_kernel),
                second_arrival_kernel_upper=max(v[1] for v in second_kernel),
                branch_counts=result["branch_counts"],elapsed_seconds=time.perf_counter()-started,
                full_service_certificate=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",default="D027")
    args=parser.parse_args()
    path,record=next((p,r) for p,r in records() if r["contract"]["id"]==args.case)
    result=run_case(path,record)
    save(OUT/f"reduced_adjoint_probe_{args.case}.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
