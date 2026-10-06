# Frozen bounded multi-region transfer protocol

This protocol precedes numerical trajectory or certificate outcomes. The new
experiment is a transfer test, not a replacement or recalibration of the existing
seven-state incident benchmark. No cross-network TTS ranking is intended.

## Model and generalization

For R=2 and R=3, connect all regions in one directed clockwise ring. State c_ij
counts vehicles currently in region i with final destination j; z_i is an exterior
queue entering region i. A local-destination class completes at c_ii v(N_i).
Every other class advances to the next ring region at u_i c_ij v(N_i), preserving
its final destination. N_i=sum_j c_ij. Use the installed literature polynomial
v(N)=0.0014877 N^2-0.29815 N+15.0912, with 100 vehicles per state unit and hours.
The dimensions are R^2+R=6 and12, not the primary benchmark's state convention.

Each exterior admission is a_i=min(60 z_i,120 g_i). Half the admitted vehicles
have the entering region as destination; the other half is split equally among
the other destinations (exact dyadic shares for these two networks). Thus fixed
class shares B obey c'=f(c,u)+Ba and z'=Lambda-a. In the alternative coordinates
y=(c+Bz,z), admission cancels from the first block: y_c'=f(c,u)+B Lambda.
The ordinary class coordinates and these combined-inventory coordinates verify
IDENTICAL physical plans and uncertainty sets within each network.

Primitive internal demand is declared directly: 6000 vehicles/h to the local
destination and a total500 vehicles/h split across other destinations, per
region. These values are not reconstructed from an equilibrium. Initially every
region contains3000 vehicles:1500 local-destination and1500 split equally across
other destinations. Every exterior queue starts with200 vehicles. External mean
arrival is4000 vehicles/h per queue. Queue0 alone receives a30% rectangular
increase from15 through45 minutes; its exact jump times split every proof/replay.

## Fixed plan, contract, and grid

The common horizon is exactly one hour. For the first30 minutes, all ring gates
are0.65 and all admission gates0.50. From30 minutes through the deadline, ring
gate0 is0.80, gate1 is0.55, remaining ring gates are0.65; admission gate0 is0.60
and the others0.50. This is a predeclared connected-network plan, not an optimized
or outcome-selected plan. Commands are exact installed binary coefficients.

Physical rows require all classes and queues nonnegative, every regional total
at most10000 vehicles, every queue at most2000 vehicles, and all gates in[0,1].
The predeclared scheduled task asks all regions to have at most2700 vehicles at
one hour. Physical closure and deadline satisfaction are reported separately;
a failed2700-vehicle task will not be relabeled as an accepted certificate.
All regional terminal upper supports will be recorded even when this task fails.

The frozen grid is R in{2,3}, coordinate method in{class,inventory}, and four
(d,wbar) pairs: (1,60),(10,60),(25,60),(25,300), where d is the REGIONAL total
half-width in vehicles and wbar is a measurable exterior-arrival error bound in
vehicles/h PER queue. The independent physical class half-width is d/R vehicles;
the queue half-width is min(d,10) vehicles. Consequently regional half-widths
sum the R class contributions, and total arrival-error allowance scales with R.
There are16 proof attempts. All outcomes, failures, source hashes and runtimes
are retained. No enlargement of this grid is authorized by this protocol.

## Validated computation

Use a dimension-generic signed-generator/Metzler-remainder enclosure, reusing
only dimension-independent outward interval arithmetic I/J/imat and the existing
independently validated scalar queue helper. Do not mutate installed DIM=7
globals. The polynomial predictor has degree2 and its exact field residual is
evaluated by interval polynomial coefficients. Matrix exponential/integral and
continuous signed-flow series use order18 with outward norm-tail bounds.
The nominal step is1/120 hour; failed self-inclusion may halve to1/1920 hour,
with all failed attempts retained. Exact clock cuts include both arrival jumps,
the command change and the absolute deadline. Each step requires strict
self-inclusion before any constraint/support claim.

Class and queue nonnegativity are proved independently by boundary invariance:
the installed v is globally positive (4ac-b^2>0), demands and incoming flow are
nonnegative, and the full arrival lower bound is positive. Total mass satisfies
M'=sum d+sum Lambda-sum_i c_ii v(N_i), implying bounded inventories on a finite
horizon and preventing finite escape. This argument assumes neither regional
capacity nor queue storage; those upper rows must be checked separately.
The scalar queue bounds restrict the mean-value Jacobian corridor only through
the hull of predictor and actual-queue bounds. Generators/remainders are never
clipped or restarted from displayed hulls. Checkpoints retain their correlation.

## Independent diagnostics and record

Before the grid, check exact rational share/coordinate/incidence identities,
nonzero inter-region coupling, direct class balances against an independently
implemented incoming-flow formula, conservation, analytic interval Jacobians
against central differences and complex-step derivatives away from switches,
and the exponential/integral primitive against high-precision calculations.
Independent adaptive nominal and selected bounded-adverse replays are diagnostics,
not proof bounds. Report full physical rows and deadline proof results separately
from sampled traffic metrics. Save full step evidence and immutable source copies;
an audit checks descriptor hashes, original initial generators, checkpoint chains,
strict self-inclusion, outward row supports and independent replay containment.
Any unresolved proof remains unresolved; no physical trajectory is clipped.
