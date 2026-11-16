# Three-region recovery follow-up, 16 November 2026

This is a separate exploratory follow-up, authorized by the request to rerun
the three-region case and seek a more informative Figure-4-style comparison.
It does not change the frozen earlier connected-ring protocol or its results.
All attempts, including inconclusive certificates, are retained.

First reproduce the existing three-region certificate (regional total error
25 vehicles, queue error 10 vehicles, arrival error 60 veh/h per queue).
For the new traffic case, retain the same three-region directed-ring model,
MFD, destination allocation and internal demand. Start every region at 4500
vehicles (half local destination) and every exterior queue at 500 vehicles.
Use a one-hour horizon, 2000-vehicle queue storage, 10000-vehicle regional
limits and a 3300-vehicle deadline in each region. The deadline is fixed
before examining the new trajectories.

Explore base arrivals of 4000 and 6000 veh/h per queue, with a 30% surge at
Queue 1 over minutes 15--45. Prefer the 6000-veh/h case if it admits a verified
backup. The first control block lasts five minutes; the second ends at minute
60. First-block entry closure is an explicit design choice, not computation
delay. Numerical planning may choose ring and second-block entrance inputs.
Compare a regional-priority proposal, a network-efficiency backup and checked
convex interpolations on the existing 0.05 repair grid. Any optimizer failure
and any relaxation of a planning choice must be recorded.

Seek certificates with regional total error 75 vehicles (25 per destination
component), queue error 10 vehicles and independent arrival-error bounds of
300 veh/h per queue. Also test regional error 25 vehicles and retain both
outcomes. These are tested bounds, not field-calibrated estimation claims.
Certificates, not sampled trajectories, determine acceptance. The unchanged
comparison theorem is used with destination or conservation coordinates.
Use the existing outward-rounded arithmetic and scalar queue bounds. The
generalized driver must split at every plan and arrival discontinuity and
record the actual arrival limits, initial data, deadline and coefficients.

Compare identical initial states and arrival signals across plans. Nominal
and adverse examples are explanatory trajectories. Any sampled ensemble must
state its sample count and is not an exhaustive uncertainty guarantee. Do not
call an inconclusive check a proven violation. Report timings as observed runs,
not a latency guarantee. Use exact queue balances for overflow witnesses where
possible. Keep the old Figure 5 available until the new evidence is validated.
