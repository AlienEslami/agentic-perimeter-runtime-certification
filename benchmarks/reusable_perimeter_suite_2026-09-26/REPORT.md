# Complete reusable perimeter-control benchmark report

26 September 2026. The previous manuscript is preserved in `archive/pre_reusable_benchmark_2026-09-26`.

## Completed scope

- 24 native regulation cases, both CLF and tracking MPC.
- 12 queue-only finite-plan contracts.
- 72 development and 8 held-out deadline contracts.
- 640 corner/profile realizations per constant plan; separate 80-run CLF and one-run CBF diagnostic denominators.
- Both a five-scenario and a 641-scenario predictive comparator, with every solver and proof outcome retained.
- 96 OpenAI interpretation/routing calls and eight end-to-end live planning traces.

## Main results

Regional regulation: both policies completed all 24 cases within the tested limits. Tracking MPC reduced integrated regulation error by 21.0-95.9% (median 73.5%). This comparison concerns the controllers, separately from runtime certification.

Queue-only planning: 12/12 nominal plans were certified unchanged. The 640-realization tests passed in every case. Against the declared CLF queue/exit extension, matched feasible-case TTS reduction was 23.2-24.9% (median 24.0%).

Deadline replacement: 36/80 nominal proposals passed the uncertainty-family checker. 36/80 passed every sampled physical and service test. Certified replacement/repair remained available in 74/80 contracts, with a verified continuation available in 68/80. The physical CBF diagnostic met its physical rows in 80/80 adverse trajectories and the regional deadline in 44/80.

The all-scenario planner was numerically feasible across the full replay ensemble in 80/80 contracts and directly certified in 44/80. Certification plus repair provided 74/80 certified outputs. This stronger comparator separates finite-scenario coverage from uncertainty-family admission.

## Matched primary case

The table uses the same all-high initial corner and constant-high residual, with a 2000-vehicle exterior queue limit and a 2850-vehicle Region-1 deadline. Costs of infeasible rows are descriptive, and feasible performance is compared only after applying both constraint tests.

| Policy | TTS (veh-h) | Peak queue (veh) | Region 1 at deadline (veh) | Physical / service | Family certificate |
| --- | ---: | ---: | ---: | --- | --- |
| nominal_mpc | 3300.0 | 2180.0 | 2852.2 | fail / fail | False |
| continuation | 2694.9 | 194.4 | 1727.3 | pass / pass | True |
| grid_repair | 3096.9 | 1682.7 | 2310.1 | pass / pass | True |
| scenario_mpc | 3267.8 | 2000.0 | 2850.0 | fail / pass | False |
| physical_CBF | 3295.2 | 1843.8 | 2917.4 | pass / fail | None |
| full_scenario_mpc | 3267.2 | 1999.0 | 2848.0 | pass / pass | False |
| full_scenario_repair | 3074.6 | 1547.0 | 2314.5 | pass / pass | True |

## Verified reserve availability

The archived continuation failed to close a proof in 12 contracts. A separately planned reserve certified in 12/12. With this explicit initialization/reserve-refresh tool, certified outputs were available in 80/80 deadline contracts. The original availability outcomes remain in the released data. This establishes existence on the tested finite grid; the applicable information/disturbance contract must be checked before authority is granted.

The released planning-tool portfolio selects the lowest nominal center-forecast task cost among the independently certified same-contract candidates. It supplied 80/80 outputs. Realized performance is excluded from the selection rule.

## What the paper can use

The strongest finding is the separation between a plan's nominal economic appeal, its finite-scenario behavior, and its authority to replace an existing verified plan. The released problem definitions make these distinct questions reproducible. Certified repair preserves operational choice while retaining a checked continuation through a named absolute expiry.

Priority performance and network waiting are separate objectives. On repaired proposals, median priority-cost improvement relative to retaining the continuation was 5.68%, while median TTS change was +16.07%. This is an operator tradeoff, with physical storage and the regional deadline enforced in both certified policies.

The literature benchmark includes the native reproduced CLF equations and local predictive-controller implementations, with model extensions and information choices identified. The seven-state CLF row includes an explicit queue/exit feedback extension. These results support a reusable task-and-contract benchmark, with comparison boundaries stated for each method.

The API pilot records contextual interpretation separately from physical admission. History-only objective accuracy was 19/24 clean and 15/24 poisoned; the explicit task record achieved 24/24 in each group. The separate live traces actually run the requested optimization and checking tools; their complete outcomes appear in `results/end_to_end_api/summary.json`.

## Validation and scope

The model/Jacobian, command-compilation, and independent adaptive-replay checks are archived. The cubic MFD and synthetic internal demand remain fixed while sensing and arrival residuals vary. The finite-plan deadline study uses one inherited minute-20 state; regulation uses three initial regional-total pairs. These are fully specified computational benchmarks for later comparisons. Closed-loop city-network calibration and parameter-uncertainty validation are separate deployment studies.

Individual proof checks took a median 8.7 seconds (maximum 11.8) under concurrent desktop load. The simulations use precomputed activation-time commands and report processor/API wall time separately from traffic time. A delayed deployment handover requires propagating the retained plan to the scheduled activation and checking that inherited state set.

API known token cost: USD 0.0900; conservative total accounting: USD 0.0900; authorized ceiling: USD 10. Development calls and any retained reservations are included in the accounting.

## Suggested manuscript changes after review

1. Add the three benchmark problem definitions before the architecture and explain how each relates to the perimeter-control literature.
2. Keep one main matched deadline table, the 12 queue-only unchanged-admission cases, and the uncertainty/repair summary. Put full grids and detailed trajectories in the supplement and released benchmark.
3. Present predictive repair and scenario planning as tools within the same certified execution architecture. Lead with guaranteed storage/service feasibility and conditional operational tradeoffs.
4. Separate semantic task-state evidence from physical guarantees; use the live tool traces as an interface demonstration.

The current manuscript source is unchanged. The experiment package contains `benchmark_formulation.tex`, full raw results, structured summaries and reproducible scripts. No GitHub push is performed by this study.
