#!/usr/bin/env python3
"""Deterministic assertions for the Phase-1 bounded-MFD study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mfd_incident_model as model


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reproducibility/results/mfd_uncertainty/mfd_uncertainty_results.json"


def test_exact_field_decomposition():
    rng = np.random.default_rng(55)
    for _ in range(100):
        x = model.BASE_STATE + rng.uniform(-.5, .5, 7)
        u = rng.uniform(.1, .9, 4)
        delta = rng.uniform(-.08, .08, 2)
        nominal = model.field(x, u)
        f1, f2 = model.uncertainty_channels(x, u)
        composed = nominal + delta[0] * f1 + delta[1] * f2
        assert np.allclose(model.field(x, u, delta=delta), composed, rtol=1e-14, atol=1e-12)


def test_zero_regression_and_grid_monotonicity():
    record = json.loads(RESULT.read_text())
    regression = record["zero_uncertainty_regression"]
    assert regression["exact_decision_match"]
    assert regression["exact_certified_until_match"]
    assert regression["service_upper_absolute_difference_vehicles"] == 0.
    assert regression["maximum_radius_absolute_difference"] < 1e-14
    assert record["monotonicity"]["acceptance_downward_closed_on_grid"]


def test_sampled_containment():
    record = json.loads(RESULT.read_text())["sampled_containment"]
    assert record["samples"] >= 200
    assert record["all_contained"]
    assert record["maximum_ratio"] <= 1 + 1e-8
    assert record["physical_violations"] == 0
    assert record["service_violations"] == 0


def test_structural_properties():
    record = json.loads(RESULT.read_text())["structural_tests"]
    assert record["regional_transfer_cancellation_and_completion_sign"]
    assert record["quasipositivity_passed"]
    assert record["queue_equation_contains_no_mfd_uncertainty"]


if __name__ == "__main__":
    tests = [test_exact_field_decomposition, test_zero_regression_and_grid_monotonicity,
             test_sampled_containment, test_structural_properties]
    for test in tests:
        test()
        print("PASS", test.__name__)
