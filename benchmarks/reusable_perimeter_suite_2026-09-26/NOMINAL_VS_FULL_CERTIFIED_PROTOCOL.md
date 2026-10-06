# Paired certified-planner comparison

Set on 1 October 2026 before computing the paired counts.

Compare the existing 80 Task D contracts. No plant, plan, certificate,
uncertainty envelope, simulation or manuscript is changed. The primary cohort
is every contract with **both** an accepted ordinary-MPC-route output and an
accepted full-scenario-MPC-route output. In the ordinary route, select the
already recorded `certified_repair_source` (unchanged nominal MPC or certified
grid repair). In the stronger route, select the already recorded
`certified_source` (unchanged full-scenario MPC or its certified repair).
Exclude a contract from the primary comparison if either route lacks a
certificate; list its ID. Six contracts using the same newly certified reserve
may be shown separately as ties, not as planner wins.

For each common contract, align all 640 inherited-state-corner/residual-profile
realizations by their keys. Verify that each selected plan is accepted and all
sampled physical and deadline tests pass. Compare the median paired difference
in priority cost (the optimization objective) and, separately, total time
spent. Lower is better. Count one vote per contract; do not treat the 640
correlated realizations as 640 independent problems. A difference of at most
1e-6 vehicle-hours is a numerical tie. Report win/loss/tie counts, median
relative difference, ranges and the held-out subset. Show whether advantage
depends on sensing half-width and residual bound.

Also report direct unchanged-plan certificate coverage, and the all-80
fallback-complete count. The raw nominal candidate's lower cost is not an
operational win where its constraints or certificate fail. No safety or
performance claim should be inferred from the other planner's planned
scenario set alone. Report unfavorable outcomes and both metrics.

As a separate, narrower interpretation of "ordinary MPC with certification,"
compare the recorded `certified_retention_source` (nominal if accepted,
otherwise the preverified continuation) against the same stronger route on
the contracts where both exist. Keep this result separate from the repaired
ordinary route; never combine the two denominators.
