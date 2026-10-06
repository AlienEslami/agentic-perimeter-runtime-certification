#!/usr/bin/env python3
"""Run the MFD policies from the locally synthesized SOS tube.

Run ``mfd_certificate_synthesis.py`` first.  This script reads its matrices and
SOS levels, initializes the nonlinear plant at 45% of the common synthesized
tube along the verified switching-witness direction, and then repeats the
held-mode, unrestricted, and certified comparisons. It remains separate from
the baseline audit so the solver-dependent and checked-in analytic records are
reported independently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mfd_fixed_gain_validation import (
    PROPOSAL_INTERVAL,
    json_ready,
    out_of_sample_demand,
    plot_results,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reproducibility" / "results"
SYNTHESIS_PATH = RESULTS_DIR / "mfd_certificate_synthesis_results.json"


def summary(result: dict) -> dict:
    return {
        "end_time": float(result["times"][-1]),
        "final_error_norm": float(np.linalg.norm(result["states"][-1] - np.array([30, 30, 15, 15, 5, 5]))),
        "minimum_margin": float(np.min(result["margins"])),
        "maximum_tube_occupancy": float(np.max(result["occupancies"])),
        "accepted_switches": len(result["accepted_times"]),
        "accepted_times": result["accepted_times"],
        "violation": result["violation"],
        "transportation": result["transportation"],
    }


def main() -> None:
    if not SYNTHESIS_PATH.exists():
        raise SystemExit(
            "Missing mfd_certificate_synthesis_results.json. Run "
            "mfd_certificate_synthesis.py first."
        )
    synthesis = json.loads(SYNTHESIS_PATH.read_text(encoding="utf-8"))
    quadratic = synthesis["quadratic_codesign"]
    matrices = [np.asarray(value, dtype=float) for value in quadratic["matrices"]]
    common_level = min(float(record["level"]) for record in synthesis["sos_certificates"])
    rate = float(quadratic["rate"])
    factor = float(quadratic["transition_factor"])
    dwell = math.log(factor) / rate
    realized = math.ceil(dwell / PROPOSAL_INTERVAL - 1.0e-12) * PROPOSAL_INTERVAL
    certificate = {
        "rate": rate,
        "matrices": matrices,
        "transition_factor": factor,
        "dwell_time": dwell,
        "realized_dwell": realized,
        "sampled_contraction": factor * math.exp(-rate * realized),
    }
    tube = {"common_level": common_level}
    simulations = {
        name: simulate(name, certificate, tube)
        for name in ("fixed_1", "fixed_2", "unconditional", "supervised")
    }
    if simulations["supervised"]["violation"] is not None:
        raise AssertionError("The SOS-certified supervisor violated an installed constraint")
    if np.max(simulations["supervised"]["occupancies"]) > 1.0 + 2.0e-5:
        raise AssertionError("The supervised trajectory left the common SOS tube")
    stress = {
        name: simulate(name, certificate, tube, demand_profile=out_of_sample_demand)
        for name in ("fixed_1", "supervised")
    }
    plot_results(simulations, suffix="_sos")
    output = {
        "source_synthesis_record": SYNTHESIS_PATH.name,
        "common_sos_level": common_level,
        "certificate": {key: value for key, value in certificate.items() if key != "matrices"},
        "simulation": {name: summary(result) for name, result in simulations.items()},
        "out_of_sample_demand_test": {
            "certificate_claim": False,
            "simulation": {name: summary(result) for name, result in stress.items()},
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "mfd_post_synthesis_results.json").write_text(
        json.dumps(json_ready(output), indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        RESULTS_DIR / "mfd_post_synthesis_trajectories.npz",
        **{
            f"{name}_{field}": result[field]
            for name, result in simulations.items()
            for field in ("times", "states", "modes", "margins", "occupancies", "production", "completions")
        },
    )
    print(json.dumps(json_ready(output), indent=2))


if __name__ == "__main__":
    main()
