# Reusable perimeter-control benchmark: protocol v1

Protocol established before new sweep results, 26 September 2026.

## Questions and tasks

**R — regional regulation.** The four destination-class, two-gate MFD model of Zhong et al. (2018), using the existing native reproduction, supplies a regulation task. Compare the reproduced almost-smooth CLF against a matched tracking MPC implementation. Both receive the same demand and gate limits. Regulation error, settling, gate constraints and computational time matter here. This is a reproduction of the controller equations, not an author-supplied implementation.

**Q — queue-aware economic planning.** The manuscript's seven-state nonlinear MFD includes exterior storage. At a common minute-20 state, propose a two-block remaining-horizon economic plan, with no regional deadline. Compare direct MPC execution, the published cross-gate CLF extended by the manuscript's queue/exit feedback, certificate-and-retain, and certificate-plus-repair. This is a finite-plan replacement task, not a claim about full receding-horizon optimality.

**D — deadline-constrained replanning.** Use the same state, horizon and physical constraints, adding a regional deadline and a changing regional priority. Compare the economic nominal proposal, physical CBF-QP, verified continuation, certified grid repair, and a task-aware finite-scenario predictive planner. The last comparator receives the same deadline and uncertainty budget; numerical scenario constraints do not constitute an uncertainty-family proof. Its output also passes through the unchanged checker, allowing a like-for-like comparison between planning tools.

## Fixed grids

R: three initial regional totals `(1800,3100)`, `(2400,3600)`, `(3500,2800)` vehicles; equilibrium-proportional and symmetric destination splits; gate intervals `[0.1,0.9]` and `[0.4,0.6]`; nominal and 10% sinusoidal demand. 24 cases per policy. Five-minute replanning, thirty-minute MPC prediction and ninety-minute evaluation.

Q: incident amplitudes 20%, 30%, 40%, 50%; duration 30, 45, 60 minutes; common sensing half-width 1 vehicle and residual 300 vehicles/hour. 12 contracts. All 128 initial-box corners and five residual profiles are evaluated for constant plans. Feedback baselines use a fixed common subset of 16 corners and the same five profiles, reported separately.

D development: sensing half-width 1, 10, 25, 50 vehicles; residual 60, 150, 300 vehicles/hour; deadline 2750, 2850, 2950 vehicles; Region-2 weights 2 and 4. 72 contracts/proposals. D held-out combinations: half-width 5 and 20; residual 100 and 225; deadline 2800 and 2900; weight 3. 8 combinations. These intermediate settings are excluded from development summaries and tuning. The model and algorithm settings remain identical.

Repair: test interpolation `theta=0,0.05,...,1` in that order, between proposed commands and the independently checked constant continuation. Choose the first passing grid point. This is minimum modification on this finite ordered grid; it requires no assumption that admissibility is monotone. A failed continuation certificate is recorded as an unavailable certified backup for that contract. No unverified fallback is relabeled certified.

Scenario planner: constrain the center and +/-all-coordinate initial corners under both constant residual extrema. Include the nominal center rollout in the objective. Same two control blocks, gate constraints, physical rows and deadline. Optimize with SLSQP using analytic forward sensitivities. This is a local numerical comparator, with all solver statuses recorded.

## Realizations and accounting

Use zero, constant-high, constant-low, five-minute alternating and ten-minute sinusoidal residuals. All remain inside the stated residual budget. Each constant plan is evaluated from all 128 corners (640 realizations per contract); the CLF feedback baseline uses 16 predetermined corners (80 realizations). The computationally heavier CBF diagnostic uses one common all-high corner and constant-high residual per deadline contract. Full, reduced and single-trajectory ensembles are reported with their own denominators. Independently audit selected trajectories with adaptive DOP853, including conservation and cost integration. Do not clip vehicle inventories or remove waiting arrivals.

Report queue-inclusive TTS, completed trips, peak queue, regional deadline, weighted priority cost, all sampled physical rows, initialization/certification availability, rejection/repair counts, and solver/proof wall time. Traffic costs of unsafe runs are not ranked as feasible optima. Formal uncertainty-family statements come only from the existing interval checker; finite samples are performance evidence.

## AI study and spending

After numerical runs, use the OpenAI API to test interpretation of active, interrupted and expired priorities and tool-routing instructions on benchmark cases. Preserve prompts, schemas, response bodies, model identifiers, token usage and per-call cost. Distinguish semantic success from physical certification. Budget maximum USD 10, enforced before every request by reserving a conservative token-cost upper bound. Timeout/unknown billing retains the reservation. No paid external tool calls. API evidence does not change the mathematical certificates.

## Interpretation

Report every grid case, including unavailable proofs and failed solvers. Results apply to the specified synthetic literature-MFD benchmark and uncertainty family. Keep the previous manuscript intact until results have been reviewed; provide a separate report and portable experiment package. New paper claims should follow this evidence review.
