# Diagnostic of the 36 unresolved MPC deadline certificates

27 September 2026. The original manuscript, traffic model, plans and benchmark
results are preserved. No new traffic scenario or paid API call was introduced.

## Outcome

The new search found zero deadline violations in all 36 unresolved
full-scenario MPC plans. Every one of these plans already had a passing
continuous physical-limit certificate; the unresolved certificate concerned
the Region-1 deadline only.

The tightest independently replayed deadline margin was **0.9963 vehicles**,
in D071: 50-vehicle sensing half-width, 60 vehicles/hour arrival residual, and
a 2,950-vehicle deadline limit. Across all cases the smallest margins ranged
from 0.9963 to 1.0009 vehicles. The strongest discovered candidate did not
exceed the original constant-positive corner maximum beyond numerical
integration disagreement (at most 0.0000013 vehicles).

## What was searched

| Search family | Trajectory checks across 36 plans |
|---|---:|
| All 128 initial corners, constant positive residual; reproduced baseline | 4,608 |
| All 256 five-minute high/low sequences at the eight strongest baseline corners | 73,728 |
| Random interior initial states and one-minute residual levels | 73,728 |
| Random initial corners and one-minute high/low residuals | 73,728 |
| Total fixed-family checks | **225,792** |

Thus 221,184 fixed-family checks were additional to the reproduced baseline.
There were also **288** bounded local optimization runs, with 768 terminal-map
evaluations, using five-minute and one-minute residual parameterizations.
All local solvers reported convergence. The local searches maximized the
deadline accumulation while jointly varying the initial state and residual
profile. Their convergence establishes numerical stationarity, not global
optimality over the uncertainty family.

Every candidate stayed within its original seven-coordinate initial box and
arrival-residual amplitude. All candidate initial states passed the physical
rows. No state was clipped. No sampled physical-row violations were found.

## Numerical verification

The strongest candidate for every plan was independently replayed with the
original scalar plant using adaptive DOP853, split at every residual and
command change. Ten-second, two-second and one-second RK4 replays agreed with
that independent calculation to at most **1.3e-6 vehicles** at the deadline.
The maximum vehicle-conservation residual was **2.5e-10 vehicles**.

The vectorized and original scalar fields agreed exactly at the checked
points. Forward trajectory gradients agreed with centered finite differences
to approximately 1.2e-7 in the tested five-minute and one-minute
parameterizations. Constant-positive baseline replays reproduced the archived
results within the required 1e-7-vehicle tolerance.

## Interpretation for the paper

The search supports **deadline-bound conservatism** as the explanation for
these 36 unresolved certificates. It has not produced an unsafe MPC trajectory
in this group. Piecewise-constant test families and local optimization cover a
finite subset of the allowed measurable residuals, so the complete-family
deadline statement still requires a mathematical certificate.

For example, in D027 the deadline limit is 2,850 vehicles and the independently
replayed strongest candidate ends at about 2,849 vehicles. The checker's
deadline upper bound is about 3,437 vehicles. That bound fails to establish
the deadline even though the expanded search continues to meet it.

Repair supplies a different plan with a passing physical-and-deadline
certificate. Here its demonstrated benefit is obtaining that guarantee,
potentially at a traffic-performance cost. The present diagnostic does not
support describing repair as eliminating an observed violation in these 36
plans. Existing observed violations of the ordinary nominal MPC are a
separate comparison and remain unchanged.

The complete 80-case certified planning portfolio uses MPC, checked repair
and freshly verified reserve plans. Six selected plans use the refreshed
reserve source; the all-80 result is a portfolio result, not repair alone.

## Preserved records

- Search specification: `../../FALSIFICATION_PROTOCOL.md`
- Implementation: `../../run_falsification.py`
- Numerical implementation checks: `implementation_audit.json`
- Aggregate results and case margins: `summary.json`
- Complete candidate inputs, commands, optimizer diagnostics and independent
  replays: `cases/D*.json`

The broader change-of-traffic-state/congestion experiments are deferred as
requested. No manuscript claim or figure has been changed.
