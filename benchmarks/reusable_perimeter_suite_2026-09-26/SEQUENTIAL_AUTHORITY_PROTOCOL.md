# Sequential authority pilot

Specified 1 October 2026 before running the pilot. Preserve all archived
plans, certificates, model, uncertainty contracts and manuscript text.

Use three pre-existing matched Task D pairs: D015→D014, D027→D026 and
D071→D070. Within each pair, the MFD plant, initial information box,
arrival-residual bound and absolute minute-60 Region-1 deadline are identical.
Only the soft Region-2 weight changes from 4 to 2 and back to 4.

At minute 20 install the already certified ordinary-MPC repair for the
weight-4 contract. At minute 25, the agent requests the *remaining constant
block* of the already computed 641-scenario MPC for the weight-2 contract.
At minute 30, it requests the remaining constant block of the original
weight-4 **nominal** MPC proposal. These precomputed source plans are used
as proposal tools; this pilot does not represent fresh reoptimization at
minutes 25 and 30. Keep their exact compiled gate vectors and absolute
minute-60 deadline.

Before each request, propagate the entire current certified information set
through the actual installed prefix, with no new sensor observation. Convert
the verified terminal metric shape to an outward physical box. Check the
candidate over the full remaining interval, including physical rows and the
same pending service deadline. Install it only if the certificate passes;
otherwise retain the active plan's original commands, correlated proof and
expiry. Do not use a checkpoint box as an authorization to restart the
retained plan. Record every proof, source plan, command and reason for
retention. An independent sampled replay checks all 128 initial corners and
five residual profiles under the *executed* multi-decision schedule, with
queue-inclusive delay and Region-1 deadline outcomes.

The desired mechanism result is at least one accepted cross-tool replacement
and one rejected later request with a certified retained continuation. All
three pairs count, including failures. This is a sequential authority pilot,
not a matched head-to-head with robust MPC that reoptimizes at each decision.
That comparison requires a separate prospective protocol and equal
information, planning opportunity and computation budget.

## Counterfactual accounting, specified after the three admission results

For every pair, use the same 640 initial-corner/residual realizations and the
same changing soft-priority weights to replay: (i) the original certified
plan held to minute 60; (ii) every requested replacement executed without a
checker; (iii) the certified sequential schedule above. Keep all outcomes,
including cases where the unchecked execution happens to satisfy sampled
constraints. Report both physical and minute-60 service outcomes, queue-
inclusive TTS, and the changing-priority objective. These are matched
counterfactuals, not formal guarantees for unchecked policies.

For D027/D026, inspect the rejected minute-30 proposal separately: replay
the accepted minute-25 prefix followed by that rejected command. A sampled
violation, if any, illustrates the use of retention; absence of sampled
violations must be reported as such, since a failed proof alone is not a
physical counterexample. No candidate, source plan or hard contract may be
retuned for this analysis.

## Full-grid discovery scan, specified after the pilot result

The three preselected cases did not show sampled violations from unchecked
tool switching. Screen all 40 existing weight-4/weight-2 Task D pairs,
without modifying any source plan or hard contract. A pair is eligible when
the weight-4 ordinary repair and weight-2 stronger MPC source both have
archived accepted initial certificates. For each eligible pair, replay the
same unchecked stage-0/stage-1/stage-2 request sequence at minutes 20/25/30
over all 640 corner/profile realizations. Report the full scan, including
zero-violation outcomes. If a pair shows a sampled violation, run the full
sequential proof procedure on that pair and label it a discovered stress
case, separate from the three preselected pilot pairs.

The first scan implementation mistakenly paired the eight held-out cases by
adjacent ID, although those cases vary deadline and disturbance rather than
soft priority. They are **not matched pairs** and must be excluded. The
corrected scan requires equality of all hard-contract fields and soft weights
4/2 before eligibility. The invalid preliminary `full_grid_scan.json` was
overwritten by the corrected record; the script history records the error.
The corrected scan has 30 eligible development pairs, three with sampled
physical violations under unchecked switching. Run the full proof procedure
for the first violating pair by numeric ID, D025/D024. There is no held-out
matched pair under this protocol; hold-out generalization would require a
new prospectively designed contract set.

After validating the first witness independently, also run the full proof
procedure for the other two violating pairs, D043/D042 and D061/D060, so the
report covers all discovered physical-violation cases rather than only the
first. Keep all rejected-proof diagnostics and sampled outcomes.

## Repair extension for the first stress case

At minute 30 in D025/D024, interpolate the rejected nominal command toward
the then-active accepted scenario-MPC command on the fixed grid
theta = 0.05, 0.10, ..., 1.00. First verify the active command as a **new**
endpoint from the propagated minute-30 box. If that endpoint fails, do not
claim grid completeness; stop the new-proof repair search and retain the
original correlated proof. If it passes, check each compiled interpolated
command in ascending theta, and install the first whose physical and service
certificate passes. Compare its 640 matched replays with the already reported
retention and unchecked schedules. All failed attempts and proof times count.
This extension is exploratory and separate from the preselected pilot.
