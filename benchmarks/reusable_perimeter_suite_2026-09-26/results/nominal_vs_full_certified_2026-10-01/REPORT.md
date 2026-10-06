# Ordinary versus 641-scenario MPC under the same certifier

1 October 2026. The predeclared comparison is in
`../../NOMINAL_VS_FULL_CERTIFIED_PROTOCOL.md`. It reuses the 80 fixed Task D
contracts, installed plans, certificates and 640 matched realization replays
per contract. No optimizer was rerun, no case was discarded after seeing a
traffic metric, and the manuscript was not changed. The stronger planner is a
641-**scenario** MPC; finite-scenario feasibility by itself is not a
whole-family robust certificate.

## Main like-for-like result

Both planners have an already accepted output in 74 contracts. The ordinary
route uses unchanged nominal MPC when accepted and its existing certified
repair otherwise. The stronger route uses unchanged scenario MPC when accepted
and its existing certified repair otherwise. All selected outputs pass both
the whole-family checker and all 640 sampled physical/deadline tests. Each
contract contributes one vote using the median *paired* realization difference.

| Lower is better | Ordinary route wins | Stronger route wins | Identical plans/ties |
|---|---:|---:|---:|
| Priority cost: the planners' regional objective | 21 | 17 | 36 |
| Total time spent, including the exterior queue | 18 | 20 | 36 |
| Exterior queue vehicle-hours | 17 | 21 | 36 |

The 36 ties execute exactly the same compiled commands. Among ordinary-route
priority wins, the median advantage is 0.22%; among stronger-route wins, the
median advantage is 0.10%, with one larger stronger-route advantage of 4.34%.
Across all 74 contracts, the median difference is zero because so many plans
are identical. The eight held-out contracts all favor the stronger route on
priority cost by 0.06–0.10%, while all eight favor the ordinary route on total
time spent by 0.18–0.30%. These metrics expose different operator preferences;
they do not support an unqualified performance ranking.

Direct unchanged-plan certification covers 36/80 ordinary MPC plans and
44/80 stronger MPC plans. With their previously checked repairs each route
has a certified output in 74/80. The remaining six contracts use the same
newly certified reserve repair and therefore add six ties to an all-80
fallback-complete accounting.

## Certificate with retention only

If an ordinary MPC rejection simply retains the preverified continuation,
instead of trying repair, the 74 matched results change sharply: ordinary
retention has lower total time spent in all 38 differing contracts, but the
stronger route has lower regional priority cost in all 38. The other 36 are
identical plans. This is a **priority-versus-delay trade-off**, not evidence
that one route dominates. In particular, reporting only total time spent
would omit the planning task that motivated the reconfiguration.

## Interpretation

The current fixed benchmark does **not** establish that ordinary MPC plus
certification outperforms the stronger planner in most contracts. It does
show that ordinary MPC is often reusable: 36 direct plans are unchanged and
certified; repair yields 21 priority wins and 18 total-delay wins among the
74 matched certified routes. The stronger planner obtains more direct
certificates and wins every held-out priority comparison.

The stronger paper claim is tool flexibility with a common physical/service
contract: a checker can admit an ordinary or stronger plan, repair a rejected
plan, and retain or use a reserve when needed. The existing prospective
certified-tool portfolio chooses 61 original-tool plans, 13 stronger-tool
plans and six fresh-reserve plans, all under that same contract. Its choice is
based on nominal planning cost before replay outcomes, so it should be
presented as certified selection, not as a hindsight performance win.

Reproduce the counts with `../../run_nominal_vs_full_certified.py`.
Contract-level paired results are in `certified_repair_pairs.json` and
`certified_retention_pairs.json`; `summary.json` includes uncertainty and
held-out breakdowns.
