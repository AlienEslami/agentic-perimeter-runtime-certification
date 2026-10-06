# Structure-preserving reachable-set extension

Frozen design, 10 September 2026 local time, before new numerical outcomes.

Keep the existing plant, commands, uncertainty frontier, manuscript, previous
live calls and all results unchanged. New code and proofs have separate names.
No live calls, manuscript release, commit or push belongs to this experiment.

## Questions and comparisons

Can structural nonnegativity, autonomous queue bounds, and retained signed
correlations increase the finite-plan certification range? Compare against the
same archived first MPC plan, exact one-hour deadline and 2850-vehicle service
limit. Never replace an unavailable certificate with a sampled safety claim.

Test the baseline widths 1, 5, 10, 25, 50, 100 vehicles at residual60 vehicles/h,
and width1 at residual120,300,600,1200 vehicles/h. Queue half-width is capped at
10 vehicles exactly as in the preceding frontier. Joint tests follow the largest
passing and first failing axis values. Retain every failure. If all axis tests
pass, test initial widths150 and200 with the same queue cap; invalid initial
sets are classified explicitly. Extensions beyond this menu require a declared
follow-up design and preserve these results.

Separate variants are (1) unchanged interval enclosure plus independently proved
class/queue positivity for lower physical constraints; (2) endpoint intersection
with the positive class cone followed by outward reboxing; (3) independent
scalar queue comparison added to that intersection; (4) signed affine-error
propagation to preserve initial-state correlations. Comparisons hold the
commands and physical contract fixed. A combination is run only after its
individual inclusion arguments and interfaces pass review.

## Proof obligations

The installed class completion multiplier q(n)=a*n^2+b*n+c must be proved
nonnegative on the required domain using exact or outward arithmetic. Verify
the complete initial information set has nonnegative classes and queue,
installed class demands and arrivals are nonnegative, and every installed
constant command has the signs required for the invariance proof. Do not infer
positivity merely by clipping a simulated trajectory. Storage and regional jam
limits remain independently checked and are never assumed to prove themselves.

Every tightened set must contain the prior rigorously established reachable
set intersected with independently valid structural information. Record the
uncontracted box, structural information, contracted box and outward center/
radius conversion. No nominal-radius reset or observation update is allowed.
An affine propagation method must validate its matrix exponential, nonlinear
remainder, continuous-step self-inclusion and affine output supports.

An autonomous scalar queue comparison may use explicit branch solutions or
comparison with interval arrival extrema. Arbitrary measurable disturbances
remain in the same declared bound. Crossing a branch boundary requires a
validated enclosure, not tolerance-based rootfinding presented as proof.

## Checks and reporting

Run exact algebra checks, outward interval inclusion tests, independent
class-balance replays, and same-plan disturbance/corner observations. Numerical
replays test implementation and can reveal violations; they do not prove all
trajectories safe. Verify original source/result hashes, bind new proof IDs to
variant, contract, source version and exact plan. Report service/physical proof
failure separately. Describe standard interval/positive-systems ingredients as
such; assess the transportation-specific composition on its actual evidence.
