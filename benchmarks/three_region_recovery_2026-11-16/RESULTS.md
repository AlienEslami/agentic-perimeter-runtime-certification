# Three-region recovery study — 16 November 2026

## Outcome

A new one-hour recovery case is verified for the three-region directed ring.
Each region starts at an estimated 4500 vehicles, above the MFD critical
accumulation of approximately 3392 vehicles, and each exterior queue starts
at 500 vehicles. Every region must have at most 3300 vehicles at minute 60.
Regional storage is 10000 vehicles and each queue has storage for 2000.

The prescribed proposal prioritizes Region 1 by reducing its entrance input.
Under an allowed arrival/error condition its queue reaches 2265 vehicles.
The repaired plan increases that entrance input from 0.495 to 0.51975 after
minute 5. This admits another 297 vehicles/hour and prevents the overflow.
The verified full-hour maximum of that queue is below 1992.751 vehicles.
The verified endpoint upper bounds for Regions 1–3 are 2435.696, 2364.601 and
2327.264 vehicles, respectively. These are upper bounds on vehicle counts,
not certification scores. All three targets of 3300 vehicles are therefore met.
Queues 2 and 3 have full-hour verified upper bounds below 1310.001 vehicles.

The backup also passes. Its first-queue maximum is below 1735.001 vehicles,
and its regional endpoint upper bounds are 2758.012, 2512.933 and 2403.067.

## Meaning of the comparison figure

The four panels show the three regional counts and the entry queue of Region 1.
The three plans use the same adverse conditions: every initial destination
count is at its upper bound, each queue starts at 510 vehicles, and every
queue receives its forecast plus 300 vehicles/hour throughout the hour.
Shading shows a guaranteed range for the repaired plan over the full stated
uncertainty family, not only the illustrated conditions. The regional target
applies at minute 60; the 2000-vehicle queue capacity applies at every time.

All plans start at minute 0. The change at minute 5 is scheduled in each
plan, not a five-minute computation delay. This is a comparison of alternative
plans from the same initial conditions, not a successive-replacement experiment.

## Conditions and method

- Same traffic dynamics, fixed routing, MFD, destination allocation and
  internal demand as the earlier three-region study.
- Base exterior arrivals: 6000 vehicles/hour per queue; Queue 1 increases
  to 7800 over minutes 15–45.
- Independent arrival errors: any measurable signals in ±300 vehicles/hour
  at each queue. This is five times the previous figure's ±60 bound.
- Initial error: ±25/3 vehicles on each of the nine destination-specific
  accumulations. Each regional total can therefore differ by up to 25
  vehicles. Each of the three queue counts has error ±10 vehicles.
- This regional estimation bound is unchanged from the earlier ring case.
  The new case changes initial congestion, arrivals, plans and the service
  target; it is not a controlled comparison isolating the checker improvement.
- All ring inputs equal one. Entry inputs are zero for minutes 0–5. For
  minutes 5–60, the proposal uses (0.495,0.50,0.50), the backup
  (0.55,0.50,0.50), and the repaired plan (0.51975,0.50,0.50).
- These are explicitly prescribed plans, not newly optimized MPC outputs.
- Verification uses destination coordinates, signed generators, independent
  queue bounds, and 15-second intervals split at every input/arrival change.

The Jacobian evaluation was tightened without changing the model or the
containment argument. The speed quadratic is bounded using its extrema.
For a Jacobian entry, when both partial derivatives have fixed signs over
the entire interval box, its extrema are evaluated at the appropriate corners.
Otherwise ordinary outward interval evaluation is retained. This reduces
dependency overestimation. The original bound evaluation was inconclusive
for the selected backup; the improved evaluation certifies it.

The repaired plan is the first feasible candidate on the 0.05 interpolation
grid. For every lower grid weight, the queue balance supplies an allowed
overflow witness. With upper initial queue and positive arrival error,
z_1(1) >= 7710 - 11000(0.495 + 0.055 lambda). At lambda = 0.40 this is
2023 vehicles. Lambda = 0.45 passes the complete verification, including
the regional constraints and deadlines. No global optimality is claimed.

## Traffic cost and validation

On the common illustrated adverse scenario, the repaired plan retains 59.26%
of the proposal's reduction in the Region-1-weighted cost relative to the
backup. Its total time spent is 0.22% lower than the backup. These are
simulation results for that scenario, not worst-case performance guarantees.

The comparison includes 64 matched scenarios per plan (192 simulations):
four combinations of upper/lower initial states and constant positive/negative
arrival errors, one nominal scenario, and 59 seeded random initial states with
independent five-minute arrival-error blocks. The verification additionally
covers all allowed measurable error signals, beyond this finite sample.

Every interval of the two successful certificates was recomputed from its
recorded predecessor (240 intervals per plan). Sampled regional counts and
queues stayed within the recorded bounds. Independent balance implementations
agreed to 1.5e-13 scaled units. Numerical differentiation agreed with the
Jacobian to 7.1e-9, and high-precision corner evaluations and random interior
checks agreed with the tightened interval bounds. The old three-region
certificate was reproduced before the new study.

The successful initial checks each took about 12.4 seconds during these
runs. This is an observed time, not a worst-case latency guarantee.

## Limits and preserved attempts

The larger regional error tests (50 and 75 vehicles per regional total) were
inconclusive for the selected backup. They remain in results/ along with
earlier unsuccessful evaluations and alternative 4000-vehicles/hour cases.
Thus this study increases the tested arrival uncertainty and introduces
congested recovery, but does not establish a wider regional-estimation range.

The deadline has substantial slack. Admission remains capacity-limited under
all allowed conditions in the successful case, so arrival error changes the
queues without directly changing regional admissions. The guarantees remain
conditional on the stated MFD model, allocation and sufficient receiving
capacity assumptions. No plant mismatch or spillback model has been added.

The study is included in Section 6.2 and Figure 5 of the 16 November 2026
manuscript. Figure 5 shows all three regional accumulations and all three
entry queues. Earlier manuscript versions remain preserved.

## Reproduction

The local vendor/ directory and immutable source_snapshots/ preserve the code
used by each saved certificate. The successful certificate files are:

- results/monotone_backup_class_d25_w300_proof.json.gz
- results/monotone_repair045_class_d25_w300_proof.json.gz

Use candidate(1) and candidate(0.45) from study.py, set net.refined = True,
and call verify_new(net, 'class', 25, 300) to regenerate them. Run evaluate.py
for the matched simulations and validation, then plot_all_queues.py for the six-panel Figure 5. The earlier four-panel
plot.py is retained as a historical visualization.
The numerical result summary is results/validated_results.json. Previous
inconclusive checks are separate files and are not overwritten by these runs.
