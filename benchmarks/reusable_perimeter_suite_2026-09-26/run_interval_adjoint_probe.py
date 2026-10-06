"""Pilot: whole-corridor interval adjoint for an unchanged deadline plan.

This is intentionally a probe. A certificate is reported only when every
required interval sign and the final validated trajectory inequality passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction

import numpy as np

from common import HERE, START, DURATIONS, fixture, save
from run_falsification import records
from mfd_incident_affine_reachability import (
    AffineGeneratorVerifier, I, exponential_and_integral, imat,
)
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_verifier import (
    branch_for_box, closed_field, interval_fraction, jacobian,
)

OUT = HERE / "results/directional_service_2026-10-01"


class CaptureVerifier(AffineGeneratorVerifier):
    def step(self, center, radius, mode, left, right, pulse, residual_bound):
        next_center, next_radius, info = super().step(
            center, radius, mode, left, right, pulse, residual_bound)
        geo = self.geometry(mode)
        h = interval_fraction(right-left)
        tau = I(0., h.hi)
        arrival0, arrival_slope = pulse.affine(left, right)
        center_branch, _ = branch_for_box(I(center), mode)
        chosen = "gate" if center_branch == "both" else center_branch
        values = closed_field([I(value) for value in center], mode, arrival0, chosen)
        fbox = I(np.array([v.lo for v in values]), np.array([v.hi for v in values]))
        velocity = fbox.mid()
        acceleration_box = imat(jacobian(I(center), mode, chosen), fbox)
        forcing_slope = np.zeros(7); forcing_slope[6] = 1.
        acceleration = ((acceleration_box + I(forcing_slope) * arrival_slope) * I(.5)).mid()
        curve_box = I(center) + I(velocity) * tau + I(acceleration) * tau * tau
        support = imat(I(geo["T"].mag()), I(info["apriori_radius"])).hi
        physical_box = curve_box + I(-support, support)
        queue_lo, queue_hi = info["independent_physics"]["mean_value_queue_hull"]
        physical_box.lo[6] = max(float(physical_box.lo[6]), queue_lo)
        physical_box.hi[6] = min(float(physical_box.hi[6]), queue_hi)
        assert physical_box.lo[6] <= physical_box.hi[6]
        branch, _ = branch_for_box(physical_box, mode)
        assert branch == info["active_branch"], (branch, info["active_branch"])
        j = jacobian(physical_box, mode, branch)
        info["captured_physical_lower"] = physical_box.lo.tolist()
        info["captured_physical_upper"] = physical_box.hi.tolist()
        info["captured_jacobian_lower"] = j.lo.tolist()
        info["captured_jacobian_upper"] = j.hi.tolist()
        return next_center, next_radius, info


def run_case(path, record):
    contract = record["contract"]
    policy = record["policies"]["full_scenario_mpc"]
    commands = np.asarray(policy["commands"])
    started = time.perf_counter()
    result = verify_command_plan(
        np.asarray(fixture()["center"]), commands, DURATIONS,
        initial_time_h=START, amplitude=contract["amplitude"],
        duration=contract["duration_h"], initial_box=contract["width_vehicles"]/100,
        residual_bound=contract["residual_veh_h"]/100, step_seconds=15,
        verifier=CaptureVerifier(), retain_steps=True,
        service_checks=[dict(time_h=1, coordinate=0, upper=contract["deadline_vehicles"]/100)],
    )
    steps = result["interval_steps"]
    assert result["verified_continuous_physical"] and len(steps) == 160
    row = I(np.eye(7)[0])
    initial_row = None
    mixed = 0
    gate = 0
    supply = 0
    first_block_kernel = []
    second_block_kernel = []
    max_width = 0.
    for item in reversed(steps):
        j = I(item["captured_jacobian_lower"], item["captured_jacobian_upper"])
        # The installed 15-second grid is exactly 1/240 hour in both blocks.
        # Reconstruct that rational duration instead of subtracting rounded
        # JSON time stamps.
        assert abs(item["right_h"]-item["left_h"]-1/240) < 1e-14
        h = interval_fraction(Fraction(1, 240))
        # The Peano--Baker series has the same interval-matrix product terms
        # as exp(J h). Since every time-varying Jacobian factor lies in J,
        # its interval products and the norm-series tail enclose the flow.
        phi, _, _ = exponential_and_integral(j, h)
        partial, _, _ = exponential_and_integral(j, I(0., h.hi))
        kernel = imat(I(row.lo[None, :], row.hi[None, :]), partial)[0, 6]
        target = first_block_kernel if item["segment"] == 0 else second_block_kernel
        target.append((float(kernel.lo), float(kernel.hi)))
        row = imat(I(row.lo[None, :], row.hi[None, :]), phi)[0]
        max_width = max(max_width, float(np.max(row.hi-row.lo)))
        mixed += item["active_branch"] == "both"
        gate += item["active_branch"] == "gate"
        supply += item["active_branch"] == "supply"
    initial_row = row
    return {
        "id": contract["id"],
        "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "whole_physical_corridor_verified": True,
        "interval_adjoint_initial_lower": initial_row.lo.tolist(),
        "interval_adjoint_initial_upper": initial_row.hi.tolist(),
        "first_block_arrival_kernel_lower": min(x[0] for x in first_block_kernel),
        "first_block_arrival_kernel_upper": max(x[1] for x in first_block_kernel),
        "second_block_arrival_kernel_lower": min(x[0] for x in second_block_kernel),
        "second_block_arrival_kernel_upper": max(x[1] for x in second_block_kernel),
        "branch_step_counts": dict(mixed=mixed, gate=gate, supply=supply),
        "max_adjoint_interval_width": max_width,
        "expected_initial_signs": [1, 1, -1, -1, 0, -1, 1],
        "initial_signs_resolved": [bool(initial_row.lo[k] > 0 if s == 1 else initial_row.hi[k] < 0 if s == -1 else
                                        initial_row.lo[k] == 0 and initial_row.hi[k] == 0)
                                   for k, s in enumerate((1, 1, -1, -1, 0, -1, 1))],
        "elapsed_seconds": time.perf_counter()-started,
        "new_certificate": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="D027")
    args = parser.parse_args()
    path, record = next((p, r) for p, r in records() if r["contract"]["id"] == args.case)
    result = run_case(path, record)
    save(OUT / f"interval_adjoint_probe_{args.case}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
