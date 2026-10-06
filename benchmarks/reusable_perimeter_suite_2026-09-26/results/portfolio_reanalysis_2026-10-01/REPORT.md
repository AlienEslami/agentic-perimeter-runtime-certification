# What the existing certified planner portfolio shows

1 October 2026. This reanalysis follows
`../../PORTFOLIO_REANALYSIS_PROTOCOL.md`, using the previously recorded
prospective selection in `../certified_tool_selection.json`. It adds no new
traffic case, optimizer output or manuscript claim.

The 80-contract selector chooses the lowest center-forecast priority cost
among independently certified candidates, before the 640 realization replays
are examined. It selects 61 original-tool plans, 13 full-scenario-tool plans
and six refreshed reserve plans. Every selected output has an accepted
physical-and-service certificate.

| Matched comparator | Portfolio lower priority cost | Comparator lower | Ties | Portfolio lower total time spent | Comparator lower | Ties |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary MPC with certified repair/reserve | 17 | 0 | 63 | 0 | 17 | 63 |
| 641-scenario MPC with certified repair/reserve | 25 | 0 | 55 | 1 | 24 | 55 |

Each vote is one contract, based on the median paired difference over its 640
matched realizations. The largest observed priority improvement against the
stronger-only route is 0.41%; the held-out improvements are much smaller
(at most 0.00162%). Thus the portfolio's useful story is certified choice
among tools, with task-priority gains and an explicit delay trade-off. It is
not a large network-performance improvement. The realized priority wins are
empirical; only the center-forecast selection criterion is guaranteed to be
nonworsening when an additional certified candidate is available.

The ordinary-versus-stronger comparison has 21 ordinary priority wins,
17 stronger wins and 36 identical plans among the 74 contracts with both
routes certified. All 21 ordinary wins arise **after both proposals are
repaired**. Direct unchanged ordinary MPC and directly certified stronger
MPC use the same compiled plan in the 36 tie cases. This is a repair-path and
tool-selection result, not evidence that ordinary MPC solves the same robust
optimization more effectively.

The six uncovered contracts use the same freshly certified reserve repair
under both fixed-tool comparators. All portfolio and comparator outcomes are
evaluated under the same contract; the stronger planner's 641 finite planning
scenarios do not themselves constitute its continuous whole-family proof.

`../../run_portfolio_reanalysis.py` reproduces the comparison from archived
plans and replays. `summary.json` and `paired_contracts.json` retain all
contract-level outcomes.
