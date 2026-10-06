# Deadline falsification of the 36 unresolved full-scenario MPC plans

Specified on 27 September 2026 before running the new search.

## Scope

Keep the existing 80 benchmark contracts, seven-state plant, minute-20
central state, compiled MPC commands, arrival forecast and uncertainty
bounds unchanged. Search only the 36 full-scenario MPC plans for which the
physical certificate passes but the scheduled regional-service certificate
does not pass. This is a diagnostic of those plans, not a new traffic scenario,
planner, certificate or manuscript revision. No paid model calls are needed.

Each initial coordinate may independently vary anywhere within its original
box. The exterior-arrival residual may vary measurably within the original
amplitude bound. Search piecewise-constant residuals on one-minute intervals;
five-minute residuals are a subset. All candidate initial states must satisfy
the original physical rows. No inventory clipping or uncertainty inflation
is permitted.

## Search fixed in advance

For each plan:

1. Reproduce all 128 initial corners under constant positive residual, with
   the existing ten-second RK4 simulator. Compare these results with the
   archived constant-positive replays.
2. Exhaust all 256 five-minute bang-bang residual sequences at each of the
   eight initial corners with largest archived deadline accumulation (2,048
   candidates).
3. Evaluate 2,048 random interior initial states with random one-minute
   residual levels, and 2,048 random initial corners with random one-minute
   bang-bang residuals. Seeds are fixed by case identifier and recorded.
4. Maximize Region-1 accumulation at the original deadline using bounded
   L-BFGS-B and exact forward sensitivities of the discrete RK4 equations.
   Use both five-minute and one-minute residual parameterizations, with four
   starts per parameterization: the worst original constant-positive replay,
   the best structured five-minute candidate, and the best candidate from
   each random family. The five-minute starts compress one-minute values by
   block averaging; the one-minute starts preserve them. Retain every local
   solver status. A local optimum is not a global safety proof.
5. Independently replay the strongest candidate with adaptive DOP853 at
   relative tolerance 1e-11, absolute tolerance 1e-12, and a maximum two-second
   step, splitting every residual or command discontinuity exactly. Compare
   terminal accumulation with ten-second, two-second and one-second RK4.
   Save the candidate's complete initial state and residual sequence, commands,
   constraint margins, costs and solver diagnostics.

Check the vectorized plant against the original scalar field, the trajectory
sensitivities against centered finite differences at an interior point, and
constant-positive replay agreement before reporting outcomes. Recheck all
physical rows during the search as a numerical implementation diagnostic;
their existing mathematical certificates remain the relevant guarantee.

## Interpretation

A missed deadline is a numerical counterexample only after independent
adaptive replay confirms it and the initial state/residual satisfy the
unchanged uncertainty contract. Differences within numerical integration
disagreement are reported as borderline, not a confirmed violation.

If no counterexample is found, report the tested families, counts, best
remaining deadline margin and integration checks. This provides evidence
about checker conservatism, not a proof for all measurable arrival residuals.
The whole uncertainty-family deadline statement still requires a certificate.
Preserve all previous results and manuscript files unchanged. Broader initial
traffic distributions and congestion scenarios are expressly deferred.
