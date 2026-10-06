# Deadline-specific certification pilot

1 October 2026. This pilot leaves the 80 original planning problems, all
compiled MPC commands, the seven-state plant, uncertainty contracts and
manuscript unchanged. It studies the 36 plans for which the existing checker
proved all continuous physical limits but could not prove the Region-1
deadline. A passing replay or sensitivity sample is never counted as a
whole-family certificate.

## Result

The pilot found a useful exact structural reduction, but **no new complete
deadline certificates**. The unchanged-plan certified count remains 44/80.
The 36 unresolved plans remain eligible for the already checked repair or
reserve procedure. No new unsafe trajectory was found in the earlier
225,792-trajectory and 288-local-search falsification run.

| Test | Outcome |
|---|---|
| Existing deadline bound, 36 plans | Nonlinear-remainder support spans 73.20–1,144.74 vehicles, far larger than the approximately one-vehicle sampled deadline margin. |
| Terminal sensitivities, 36 plans × 22 starting uncertainty vectors | Same sampled sign pattern in every case; no sampled sign reversals. Later arrival-residual blocks have numerically zero effect on Region-1 deadline accumulation. |
| Exact queue-branch inequalities, all 36 contracts | Supply-limited throughout minutes 20–25; gate-limited throughout minutes 25–60, for every permitted initial queue and measurable arrival residual. |
| Exact five-state quotient, all 36 plans | Because both blocks set `u20=1`, the Region-1 deadline depends on `n1,n2,n11,n22+n20,z`; initial `n10` has no effect. |
| Outward interval adjoint on two representative certified physical corridors | All required initial-sensitivity sign intervals contain zero. Even the five-state quotient does not resolve them. |
| New full-family deadline certificates | **0/36**. |

The branch result follows from scalar queue inequalities. During minutes
20–25, the queue stays below the first gate threshold by at least
0.004099 queue units (about 0.41 vehicle) across all 36 contracts. The
verified lower queue value at minute 25 exceeds the second gate threshold by
at least 0.429059 queue units; its derivative thereafter is at least
25.949314 queue units/hour. These strict inequalities prove that regional
trajectories after minute 25 are independent of the arrival residual during
minutes 25–60. The exterior queue itself still responds to that residual.

The state reduction is algebraic. Put `b=n22+n20`, `p=n1-n11`,
`q=n2-b`, `h_i=G_i(n_i)/n_i` in the manuscript's normalized units and
`a=min(60z,120u02)`. For `u20=1`, the closed equations are

```
n1'  = -(n11+u12*p)*h1 + u21*q*h2 + d11+d12+d10
n2'  = -(b+u21*q)*h2 + u12*p*h1 + a + d22+d21+d20
n11' = -n11*h1 + u21*q*h2 + d11
b'   = -b*h2 + u12*p*h1 + a/2 + d22+d20
z'   = offered_arrival + residual - a.
```

Thus the projection of the original seven-state field is exactly the
five-state field for all physical states and arrivals while `u20=1`. An
independent implementation audit over 1,437 state-command pairs found maximum
field discrepancy 8.53e-14 and Jacobian-identity discrepancy 3.55e-15.

## Why the deadline proof remains open

The sampled derivative signs suggest the maximum is reached at one initial
corner and a high residual during the first five minutes. This cannot be
installed as a proof because it must cover the entire continuous initial box
and all measurable residual histories. The outward interval adjoint built
over the present certified physical corridor encloses both derivative signs.
For example, in D027 its first initial-coordinate interval is approximately
[-694.55, 695.50]; the exact five-state reduction narrows it to
[-343.88, 344.88], still containing zero. These are conservative bound
intervals, not observed sensitivities. The reduction removes irrelevant
dimensions but does not cure the wide inherited physical corridor.

The D027 bound ablation isolates the source. The installed 1-vehicle sensing
half-width plus 300-vehicle/hour residual gives a 3,436.93-vehicle deadline
upper bound against a 2,850-vehicle limit. In a **diagnostic with residual
set to zero**, the same checker gives 2,847.09 vehicles and passes. With
initial width set to zero while retaining the residual, the bound is
3,433.22 vehicles. These altered-contract checks explain conservatism;
neither proves the original contract.

## Decision

Keep the current manuscript claims and the repair/reserve results unchanged.
The queue-branch and five-state reductions are promising ingredients for a
new deadline-specific proof, not yet a replacement certificate. The next
algorithmic step is a narrow validated reachable tube for the **first five
minutes** that preserves correlations through the five-state quotient,
followed by a directional service bound over the fixed-gate continuation.
Its success criterion is an outward upper bound below the original deadline
in the unchanged contract. A new certificate should be independently
recomputed before any manuscript claim is revised.

## Reproduction

- `../../run_directional_service_pilot.py` and `cases.json`,
  `sensitivity_summary.json`: sampled terminal derivatives.
- `../../run_branch_reduction_audit.py` and `queue_branch_audit.json`:
  outward queue-branch inequalities over all 36 contracts.
- `../../run_quotient_identity_audit.py` and `quotient_identity_audit.json`:
  independent five-state implementation check.
- `../../run_interval_adjoint_probe.py`,
  `interval_adjoint_probe_D027.json`, `interval_adjoint_probe_D071.json`:
  full-state interval sensitivity attempt.
- `../../run_reduced_adjoint_probe.py`,
  `reduced_adjoint_probe_D027.json`: five-state interval sensitivity attempt.
- `../../run_service_bound_ablation.py` and
  `service_bound_ablation_D027.json`: diagnostic source-isolation ablation.
- `../../DIRECTIONAL_SERVICE_PROTOCOL.md`: prospectively recorded test
  scope and success criterion.
