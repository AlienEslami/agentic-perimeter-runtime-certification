# Existing certified-tool portfolio: paired reanalysis

Set 1 October 2026 before computing the paired counts. Reuse the released
80 Task D contracts and the existing prospective selection recorded in
`results/certified_tool_selection.json`. That selector chooses minimum
center-forecast priority cost among already certified candidates, before
seeing any realization replay. Compare it with two fixed tool routes: (1)
ordinary MPC, with its previously recorded certified repair and the common
freshly checked reserve for the six uncovered contracts; (2) 641-scenario MPC,
with its previously recorded certified repair and the same reserve. All
selected plans must be accepted by the same checker.

For each contract align the 640 existing realizations by corner/profile key.
Report contract-level median paired differences for priority cost, total time
spent and queue time, with lower better and a 1e-6 veh-h numerical tie. Count
one vote per contract, include held-out contracts, and report all losses.
This is a reuse analysis, not a new scenario or a claim of global optimality.
