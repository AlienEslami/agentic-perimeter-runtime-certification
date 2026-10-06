#!/usr/bin/env python3
"""Cross-file audit for the literature-cubic MFD manuscript migration."""

from __future__ import annotations

import json
import math
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility" / "results"
LIVE = ROOT / "live_llm_results" / "cubic"


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(value: float, expected: float, atol: float = 5e-7) -> None:
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=atol), (
        value,
        expected,
    )


def main() -> None:
    plant = read_json(RESULTS / "mfd_cubic_pilot_results.json")
    selected = plant["controller_search"]["operational_frontier"]["selected"]
    model = plant["model"]
    certificate = plant["certificate"]
    close(model["critical_accumulation_vehicles"], 3391.9308068470286)
    close(model["maximum_trip_completion_vehicles_per_second"], 6.3031365453)
    assert model["equilibrium_residual_inf"] == 0.0
    assert model["jacobian_centered_vs_complex_A_inf"] < 2e-9
    assert model["jacobian_centered_vs_complex_B_inf"] < 2e-9
    assert max(selected["closed_loop_spectral_abscissae"]) < 0.0
    close(selected["one_minute_monodromy_radius"], 1.005954864348913)
    close(selected["peak_residence_minutes"], 1.5939964131446893)
    close(selected["peak_monodromy_radius"], 1.007244815089263)
    close(selected["dwell_minutes"], 50.68997906518644)
    close(certificate["realized_dwell"], 51.0 / 60.0)
    close(certificate["sampled_contraction"], 0.9919846478372233)

    sos = read_json(RESULTS / "mfd_cubic_sos_pilot_results.json")
    close(sos["common_level"], 0.27818211421775113)
    expected_radii = (24.458062222313135, 38.47634341140599)
    for record, radius in zip(sos["sos_certificates"], expected_radii):
        assert record["feasible"]
        assert record["status"] == "optimal"
        close(record["inscribed_euclidean_radius_vehicles"], radius)
        assert record["maximum_coefficient_residual"] < 1.5e-11
        assert record["coefficient_residual_one_norm"] < 9e-10
        assert record["minimum_gram_eigenvalue"] > -9e-9
        assert record["a_posteriori_residual_bound_passed"]
        assert record["verified_normalized_lower_bound"] > 9.9e-5
        assert record["random_falsification"]["sample_count"] == 20_000
        assert record["random_falsification"]["violations_below_minus_1e-8"] == 0

    frontier_audit = read_json(RESULTS / "mfd_cubic_controller_frontier_results.json")
    assert frontier_audit["candidate_count"] == 9
    assert frontier_audit["one_minute_amplification_count"] == 8
    assert frontier_audit["audit_threshold_count"] == 6
    close(frontier_audit["selected_scale"], 12.5)

    directional = read_json(RESULTS / "mfd_cubic_directional_pilot_results.json")
    for record in directional["modes"]:
        assert record["sample_count"] == 100_000
        assert record["directions_with_flow_crossing_before_constraint"] == 0

    runtime = read_json(RESULTS / "mfd_cubic_runtime_pilot_results.json")
    stress = runtime["witness_runs"]
    expected_stress = {
        "unrestricted_one_minute": (359, 5.0079389667, 95.6276458236),
        "landing_only": (315, 0.9626797695, 52.2463310629),
        "time_envelope": (151, 0.45, 0.3348241339),
        "dwell_certified": (7, 0.45, 0.0001700947),
    }
    for policy, (changes, occupancy, error) in expected_stress.items():
        assert stress[policy]["executed_changes"] == changes
        close(stress[policy]["maximum_tube_occupancy"], occupancy)
        close(stress[policy]["final_error_vehicles"], error)
    close(stress["unrestricted_one_minute"]["first_tube_exit_time_h"], 2.1833333333)
    for policy in ("landing_only", "time_envelope", "dwell_certified"):
        assert stress[policy]["first_tube_exit_time_h"] is None

    congestion = read_json(RESULTS / "mfd_cubic_congestion_sweep_results.json")
    assert congestion["all_points_hurwitz"]
    assert congestion["dwell_range_minutes_over_hurwitz_points"] == [25.0, 69.0]
    close(congestion["installed_record"]["proposal_grid_dwell_minutes"], 51.0)
    close(congestion["installed_record"]["one_minute_monodromy_radius"], 1.0059548643)

    predictive = read_json(
        RESULTS / "mfd_cubic_predictive_filter_baseline_results.json"
    )
    predictive_30 = predictive["stress_runs"]["30.0"]
    assert predictive_30["executed_changes"] == 359
    close(predictive_30["maximum_tube_occupancy"], 5.0079389667)
    close(predictive_30["final_error_vehicles"], 95.6276458236)
    assert predictive_30["minimum_physical_margin"] > 0.42
    predictive_15 = predictive["stress_runs"]["15.0"]
    assert predictive_15["rejected_requested_changes"] == 11
    assert predictive_15["maximum_tube_occupancy"] > 2.7

    operational = runtime["operational_priority_scenario"]["runs"]
    assert operational["time_envelope"]["executed_changes"] == 24
    assert operational["dwell_certified"]["executed_changes"] == 6
    envelope_cost = operational["time_envelope"]["transportation"][
        "time_varying_priority_tracking_cost"
    ]
    dwell_cost = operational["dwell_certified"]["transportation"][
        "time_varying_priority_tracking_cost"
    ]
    close(envelope_cost, 1.9700646425619803)
    close(dwell_cost, 2.2526389033295082)
    close(100.0 * (dwell_cost - envelope_cost) / dwell_cost, 12.5441436863)

    persistence = read_json(RESULTS / "mfd_cubic_priority_persistence_results.json")
    assert set(persistence["profiles"]) == {
        "30_min_episodes",
        "60_min_episodes",
        "90_min_episodes",
        "incident_recovery",
    }
    priority_gains = []
    imbalance_gains = []
    network_differences = []
    trip_differences = []
    for profile in persistence["profiles"].values():
        comparison = profile["envelope_vs_dwell"]
        priority_gains.append(comparison["priority_cost_improvement_percent"])
        imbalance_gains.append(comparison["regional_imbalance_improvement_percent"])
        network_differences.append(comparison["network_vehicle_hours_difference_percent"])
        trip_differences.append(comparison["completed_trips_difference_percent"])
        for policy in ("time_envelope", "dwell_certified"):
            assert profile["runs"][policy]["maximum_tube_occupancy"] <= 0.8000000001
    assert min(priority_gains) > 24.5 and max(priority_gains) < 38.0
    assert min(imbalance_gains) > 38.3 and max(imbalance_gains) < 41.5
    assert min(network_differences) > 0.08 and max(network_differences) < 0.09
    assert max(abs(value) for value in trip_differences) < 0.001

    transfers = runtime["structured_transfer"]
    assert len(transfers) == 8
    unrestricted_exits = sum(
        case["runs"]["unrestricted_one_minute"]["first_tube_exit_time_h"]
        is not None
        for case in transfers
    )
    assert unrestricted_exits == 5
    for case in transfers:
        for policy in ("landing_only", "time_envelope", "dwell_certified"):
            assert case["runs"][policy]["first_tube_exit_time_h"] is None

    attack = read_json(RESULTS / "mfd_cubic_agent_attack_results.json")
    assert attack["audit"]["matched_proposal_stream"]
    assert attack["audit"]["certified_tube_contained"]
    assert attack["audit"]["certified_physical_constraints_satisfied"]
    assert attack["audit"]["unrestricted_left_certified_tube"]
    direct_attack = attack["runs"]["unrestricted"]
    certified_attack = attack["runs"]["certified_envelope"]
    assert direct_attack["executed_changes"] == 240
    close(direct_attack["maximum_tube_occupancy"], 4.0360585034)
    close(direct_attack["first_tube_exit_time_h"], 43.0 / 60.0)
    assert certified_attack["executed_changes"] == 98
    assert certified_attack["rejected_changes"] == 72
    assert certified_attack["no_op_reselections"] == 190
    assert certified_attack["attack_proposals_outside_envelope"] == 72
    assert certified_attack["rejected_attack_change_proposals"] == 72
    assert certified_attack["first_tube_exit_time_h"] is None
    assert certified_attack["maximum_tube_occupancy"] <= 0.8000000001

    architecture = read_json(RESULTS / "mfd_architecture_results.json")
    assert max(architecture["spectral_abscissae"].values()) < 0.0
    assert max(architecture["augmented_jacobian_audit"].values()) < 2.3e-9
    assert architecture["simulation"]["p_to_pi"]["accepted"]
    assert architecture["simulation"]["p_to_pi"]["command_jump_inf"] == 0.0
    requests = architecture["simulation"]["adaptation_requests"]
    assert requests[0]["accepted"] is False
    assert requests[1]["accepted"] is True
    close(requests[0]["robust_landing_ratio"], 1.5625, atol=5e-6)
    close(requests[1]["robust_landing_ratio"], 0.25, atol=5e-6)

    average_dwell = read_json(RESULTS / "mfd_cubic_average_dwell_results.json")
    close(
        average_dwell["robust_average_dwell_threshold_minutes"],
        101.37995813037288,
    )
    close(average_dwell["selected"]["average_dwell_minutes"], 105.0)
    assert average_dwell["selected"]["burst_capacity"] == 2
    for amplitude in average_dwell["grid"].values():
        for episode in amplitude.values():
            for case in episode.values():
                assert case["safe"]
                run = case["run"]
                assert run["first_tube_exit_time_h"] is None
                assert run["physical_violation"] is None
                assert run["average_dwell"]["count_bound_verified"]
    adt_30 = average_dwell["moderate_comparisons"]["30"]["adt"]
    adt_60 = average_dwell["moderate_comparisons"]["60"]
    close(adt_30["average_dwell"]["minimum_executed_interval_minutes"], 30.0)
    assert adt_60["adt"]["executed_changes"] == 6
    close(
        adt_60["adt"]["average_dwell"]["minimum_executed_interval_minutes"],
        60.0,
    )
    close(
        adt_60["adt_vs_dwell"]["priority_cost_improvement_percent"],
        1.438645255389272,
    )
    close(
        adt_60["adt_vs_dwell"]["regional_imbalance_improvement_percent"],
        8.771092893525424,
    )

    uncertainty = read_json(
        RESULTS / "mfd_cubic_calibration_uncertainty_results.json"
    )
    close(
        uncertainty["external_basis"][
            "reported_loading_to_recovery_capacity_drop_percent"
        ],
        15.0,
    )
    close(uncertainty["selected_family_rate_per_hour"], 1.3)
    close(
        uncertainty["selected_family_dwell_threshold_minutes"],
        60.730652784519734,
    )
    endpoint = uncertainty["selected_endpoint_certificate"]
    assert endpoint["passed"]
    assert max(endpoint["relative_polynomial_affinity_residuals"]) < 7e-15
    for records in endpoint["endpoint_sos"].values():
        for record in records:
            assert record["feasible"]
            assert record["a_posteriori_residual_bound_passed"]
            assert record["verified_normalized_lower_bound"] > 9.8e-5
    assert all(uncertainty["safety_audit"].values())
    close(
        uncertainty["effect_scale"][
            "completed_trip_between_calibration_spread_vehicles"
        ],
        29283.781819669908,
    )
    assert (
        uncertainty["effect_scale"]["policy_to_calibration_spread_ratio"]
        < 1.3e-7
    )

    sequential = read_json(
        LIVE
        / "sequential_semantic_agent_v3"
        / "mfd_sequential_semantic_agent_aggregate.json"
    )
    assert sequential["prompt_version"] == "mfd-sequential-semantic-agent-v3-five-trace"
    assert sequential["inclusion_audit"]["all_calls_valid"]
    assert sequential["inclusion_audit"]["all_executed_trajectories_safe"]
    assert sequential["inclusion_audit"]["total_live_calls"] == 240
    assert sequential["inclusion_audit"]["context_requests_per_variant"] == 45
    assert sequential["inclusion_audit"]["stateful_context_requests_correct"] == 33
    assert sequential["inclusion_audit"]["memoryless_context_requests_correct"] == 1
    assert sequential["inclusion_audit"]["proposals_per_variant"] == 120
    assert sequential["inclusion_audit"]["stateful_true_objective_proposals"] == 115
    assert sequential["inclusion_audit"]["memoryless_true_objective_proposals"] == 99
    close(
        sequential["inclusion_audit"]["mean_stateful_context_accuracy"],
        33.0 / 45.0,
    )
    close(
        sequential["inclusion_audit"]["mean_memoryless_context_accuracy"],
        1.0 / 45.0,
    )
    expected_context = {
        "llama3.1:latest": (10.0 / 15.0, 0.0),
        "qwen3:8b": (14.0 / 15.0, 1.0 / 15.0),
        "mistral:7b": (9.0 / 15.0, 0.0),
    }
    sequential_calls = 0
    for model in sequential["models"]:
        for variant in ("stateful", "memoryless"):
            run = sequential["runs"][model][variant]
            assert run["scenario_count"] == 5
            assert len(run["scenarios"]) == 5
            assert run["summary"]["call_count"] == 40
            assert run["summary"]["explicit_request_count"] == 25
            assert run["summary"]["context_request_count"] == 15
            assert run["summary"]["valid_call_rate"] == 1.0
            assert run["summary"]["explicit_tool_request_accuracy"] == 1.0
            assert run["summary"]["count_bound_verified"]
            assert run["summary"]["maximum_tube_occupancy"] <= 0.8000000001
            for scenario in run["scenarios"].values():
                assert len(scenario["decisions"]) == 8
                assert scenario["summary"]["valid_call_rate"] == 1.0
                assert scenario["summary"]["count_bound_verified"]
                assert scenario["summary"]["maximum_tube_occupancy"] <= 0.8000000001
                for decision in scenario["decisions"]:
                    assert decision["tool_call"]["raw_response"].strip()
                    sequential_calls += 1
        close(
            sequential["runs"][model]["stateful"]["summary"][
                "context_tool_request_accuracy"
            ],
            expected_context[model][0],
        )
        close(
            sequential["runs"][model]["memoryless"]["summary"][
                "context_tool_request_accuracy"
            ],
            expected_context[model][1],
        )
    assert sequential_calls == 240

    factorial_dir = LIVE / "factorial"
    live = read_json(factorial_dir / "mfd_cubic_factorial_aggregate.json")
    assert live["experiment"] == "cubic-MFD live prompt/scenario factorial"
    assert live["models"] == ["llama3.1:latest", "qwen3:8b", "mistral:7b"]
    assert live["prompt_templates"] == ["rule_first", "costs_first", "compact"]
    assert live["scenario_seeds"] == [20260812, 20260813, 20260814, 20260815, 20260816]
    assert live["configuration_count"] == 45
    assert live["total_live_calls"] == 1_080
    overall = live["overall"]
    close(overall["mean_lower_cost_match_rate"], 0.7148148148148149)
    close(overall["minimum_lower_cost_match_rate"], 0.5)
    close(overall["maximum_lower_cost_match_rate"], 1.0)
    assert overall["envelope_executed_changes_min"] == 0
    assert overall["envelope_executed_changes_max"] == 14
    assert overall["dwell_executed_changes_min"] == 0
    assert overall["dwell_executed_changes_max"] == 6
    assert overall["maximum_certified_tube_occupancy"] <= 0.8000000001
    assert overall["all_certified_paths_inside_tube"]

    expected_cell_matches = {
        ("llama3.1:latest", "rule_first"): 0.7916666666666666,
        ("llama3.1:latest", "costs_first"): 0.7333333333333334,
        ("llama3.1:latest", "compact"): 0.725,
        ("qwen3:8b", "rule_first"): 0.975,
        ("qwen3:8b", "costs_first"): 0.85,
        ("qwen3:8b", "compact"): 0.675,
        ("mistral:7b", "rule_first"): 0.6833333333333333,
        ("mistral:7b", "costs_first"): 0.5,
        ("mistral:7b", "compact"): 0.5,
    }
    assert len(live["cells"]) == 9
    for cell in live["cells"]:
        key = (cell["model"], cell["prompt_template"])
        close(cell["lower_cost_match_rate_mean"], expected_cell_matches[key])
        assert cell["scenario_count"] == 5
        assert cell["live_calls"] == 120
        assert cell["maximum_certified_tube_occupancy"] <= 0.8000000001

    traces = sorted(factorial_dir.glob("*/*_trace.json"))
    assert len(traces) == 45
    audited_calls = 0
    for path in traces:
        trace = read_json(path)
        assert trace["number_of_live_calls"] == 24
        assert len(trace["calls"]) == 24
        assert trace["prompt_template"] in live["prompt_templates"]
        assert trace["scenario_seed"] in live["scenario_seeds"]
        schedule = trace["summary"]["priority_schedule"]
        assert len(schedule) == 24
        assert schedule.count(0) == schedule.count(1) == 12
        for call in trace["calls"]:
            assert call["raw_response"].strip()
            assert len(call["prompt_sha256"]) == 64
            assert call["prompt"].strip()
            audited_calls += 1
    assert audited_calls == 1_080

    improvement = read_json(
        LIVE
        / "prompt_improvement"
        / "mfd_cubic_prompt_improvement_aggregate.json"
    )
    assert improvement["development"]["selected_variant"] == "grounded_numeric"
    development = improvement["development"]["variants"]["grounded_numeric"]["overall"]
    assert development["total_live_calls"] == 144
    assert development["lower_cost_matches"] == 144
    held_out = improvement["held_out"]["variants"]
    baseline_held_out = held_out["current_rule_first"]["overall"]
    grounded_held_out = held_out["grounded_numeric"]["overall"]
    assert baseline_held_out["lower_cost_matches"] == 297
    assert grounded_held_out["lower_cost_matches"] == 360
    close(baseline_held_out["lower_cost_match_rate"], 0.825)
    close(grounded_held_out["lower_cost_match_rate"], 1.0)
    close(grounded_held_out["envelope_cost_improvement_percent_mean"], 3.9761664538)
    assert grounded_held_out["envelope_cost_improvement_percent_min"] > 0.06
    assert grounded_held_out["maximum_cost_copy_error"] < 5e-9
    assert grounded_held_out["all_certified_paths_inside_tube"]

    # Retain and validate the earlier 72-call operational pilot as an archive.
    pilot = read_json(LIVE / "mfd_cubic_live_v2_aggregate.json")
    assert pilot["prompt_version"] == "mfd-cubic-operational-cost-v2"
    assert pilot["total_live_calls"] == 72
    assert pilot["all_certified_paths_inside_tube"]

    pdf_path = ROOT / "FOUNDATION_MFD_PARTB_ELSEVIER.pdf"
    reader = PdfReader(pdf_path)
    assert 35 <= len(reader.pages) <= 65  # Includes the new queue-aware study and full enclosure appendix.
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_pdf_text = " ".join(pdf_text.split())
    required = (
        "A Control-Theoretic Mathematical Foundation",
        "Literature-Cubic Two-Region MFD Numerical Record",
        "50.69",
        "1.005954864",
        "24.46",
        "38.48",
        "25 to 69 min",
        "Predictive filter",
        "anisotropy",
        "Burst scheduling and production uncertainty",
        "Additional Burst and Production-Uncertainty",
        "105-min",
        "0.02685",
        "11.21",
        "15%",
        "Sequential live language-to-tool proposals",
        "240 live calls",
        "33/45",
        "115/120",
        "Memory-poisoning attack on controller choice",
        "All 72 attack-period proposals",
        "1,296",
        "Conservation and performance",
        "class-composition",
        "potential outflow",
        "Performance measures",
    )
    for phrase in required:
        assert phrase in normalized_pdf_text, phrase
    obsolete = (
        "5.8111",
        "0.205 h",
        "3.96",
        "47.78",
        "28.8×",
        "118 proposals",
        "two-hour heuristic",
        "not an empirical claim",
        "not claimed to be a calibrated welfare model",
        "not a claim about typical forecast accuracy",
        "does not mislabel this stress run",
        "not a claim about the vehicle",
        "hours and seconds are directly comparable",
        "economic ordering, rather than the prompt",
        "Seventy-two live cubic-MFD proposal calls",
        "72 genuinely live language-model calls",
        "12.63%",
        "optimal inaccurate",
        "conserved network direction",
        "companion boundary command",
    )
    for phrase in obsolete:
        assert phrase not in normalized_pdf_text, phrase

    source = (ROOT / "FOUNDATION_MFD_PARTB_ELSEVIER.tex").read_text(encoding="utf-8")
    assert "\\iffalse" not in source
    abstract = source.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    assert len(abstract.split()) <= 250
    foundations = read_json(RESULTS / "mfd_review_foundations_audit.json")
    assert foundations["passed"]
    assert foundations["network_balance_residual_units_per_hour"] < 1e-10
    assert foundations["independent_integrated_balance_residual_vehicles"] < 1e-7

    bbl_text = (ROOT / "FOUNDATION_MFD_PARTB_ELSEVIER.bbl").read_text(
        encoding="utf-8", errors="replace"
    )
    assert bbl_text.count("\\bibitem") == 57

    log_text = (ROOT / "FOUNDATION_MFD_PARTB_ELSEVIER.log").read_text(
        encoding="utf-8", errors="replace"
    )
    for phrase in (
        "undefined references",
        "Citation `",
        "There were undefined citations",
        "Overfull \\hbox",
        "Rerun to get cross-references right",
    ):
        assert phrase not in log_text, phrase

    print("PASS: cubic plant, controller frontier, residual-validated SOS records,")
    print("      congestion sweep, predictive-filter comparator, runtime studies,")
    print("      average dwell, production uncertainty, all live calls,")
    print("      partial observation, manuscript text, and PDF log.")


if __name__ == "__main__":
    main()
