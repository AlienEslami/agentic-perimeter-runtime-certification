# Deadline-specific certificate pilot

Set on 1 October 2026, before new directional sensitivity results.

## Objective and unchanged contract

Try to replace the overly broad Region-1 deadline support in the existing
checker, without changing any compiled full-scenario MPC plan, seven-state
MFD plant, initial information set, arrival-residual bound, physical row,
absolute deadline, or previously reported result. Only the 36 plans for which
continuous physical safety was already certified but the deadline proof was
unresolved are in scope. Each prospective proof must cover the entire
continuous initial box and all measurable bounded arrival residuals, including
states and residuals between the previously sampled trajectories.

The proposed technical hypothesis is that, for each fixed two-block plan,
Region-1 accumulation at the deadline is coordinate-monotone over the
admissible initial box and nondecreasing in the uncertain exterior-arrival
signal. If those properties hold, the exact maximizer is one initial box
corner and the constant upper residual. Numerical searches have repeatedly
found such a candidate, but are not a proof of the sign properties.

## Ordered tests

1. Reproduce the existing signed-generator deadline upper bounds and isolate
   the generator and nonlinear-remainder contributions for all 36 plans.
2. Numerically evaluate exact forward sensitivities of the terminal Region-1
   accumulation with respect to each initial coordinate and bounded arrival
   perturbations. Test the discovered worst corners and independent interior
   samples. Record the smallest absolute signed derivative and every sign
   change. This is diagnostic only.
3. Attempt to enclose sensitivities over the *entire* certified state corridor
   using outward interval arithmetic, including both sides of the nonsmooth
   admission branch. Every sign needed for corner/waveform reduction must be
   bounded away from zero, or its zero case handled analytically. A proved
   monotonicity reduction must be recorded independently of sampled outcomes.
4. If the sign proof succeeds, integrate the extremal trajectory with a
   validated outer bound on its deadline value and compare with the fixed
   limit. A numerical ODE tolerance or sampling margin is not a certificate.
   Confirm the original physical certificate remains applicable and record
   runtime against the existing decision clock.
5. If a sign is unresolved, explore a targeted interval partition or a
   direct service-row bound, preserving outward arithmetic. Stop and report
   inconclusive cases if a full-family inequality cannot be proved.

No threshold will be relaxed and no uncertain range will be narrowed after
seeing results. All attempted cases and failures stay in the pilot record.
The target is more unchanged-plan certificates; zero is a valid outcome.
Do not modify the manuscript or replace the archived certificates during
this pilot.
