# Bounded MFD uncertainty — Phase 1 report

## Decision summary

The field, certifier, signed-generator verifier and simulated plant now support

\[
\widetilde q_{rj}=(1+\Delta_r(t))q_{rj},\qquad |\Delta_r(t)|\leq\bar\delta_r<1.
\]

The planners and all archived commands remain nominal. The zero-uncertainty branch reproduces the manuscript's Table-2 bounds to recorded precision, including 1779.41/1728.68/1721.75 vehicles in the first row. The nominal repair bound is reproduced as 2849.093 vehicles. All deterministic tests pass, and 200 within-contract time-varying simulations lie inside the 0.5% signed-generator enclosure at every 15-s checkpoint.

The requested 2%, 5% and 10% experiments do **not** produce a full-hour certificate for any Table-2 cell or method. The best 2% case closes through 31.0 minutes; the best 5% and 10% cases close through 14.25 and 8.0 minutes. The requested 5% online protocol therefore cannot commission its first plan, so its online and retained-set stress experiments have no certified execution to evaluate.

**Recommendation:** do not put the requested 2–10% study into the manuscript as a positive robustness result. The implementation is sound and exposes a concrete research problem: synthesis and/or a tighter structured enclosure is required before MFD uncertainty at those levels can support the paper's one-hour mission. The subpercent frontier can be retained as diagnostic evidence, but it should enter the manuscript only after the uncertainty scale is calibrated and the repair is redesigned to reserve robust deadline margin.

No manuscript file was edited.

## Code changes

- `mfd_incident_model.py`: regional multiplicative flow factors, exact uncertainty channels, uncertain RK4 replay and uncertainty-aware departure accounting.
- `mfd_incident_verifier.py`: interval-Delta Jacobian, polynomial regional-channel forcing, proof binding and an exact zero-uncertainty branch.
- `mfd_incident_affine_reachability.py`: uncertain generalized Jacobian and forcing while retaining the nominal frozen Jacobian.
- `mfd_incident_mpc_certification.py`, `mfd_incident_positive_reachability.py`: uncertainty-bound plumbing; independent positivity and scalar-queue arguments retained.
- Study/test drivers in this directory generate only `results/mfd_uncertainty/` outputs.

The regional factors preserve all transfer cancellations. Total regional inventory still contains nonpositive trip-completion terms; its existing upper comparison remains valid because \(1+\Delta_r>0\) and the upper bound discards those completion terms. The queue equation contains no MFD factor.

## Tests

| Test | Outcome |
|---|---|
| Field decomposition against the implemented uncertain plant (100 random points) | pass |
| Exact zero-uncertainty Table-2 reproduction | pass |
| Nominal 2849.10-vehicle repair reproduction | pass (2849.093) |
| Structural transfer/sign, quasipositivity and queue checks | pass |
| 200 random/corner containment runs at 0.5%, 15-s checkpoints | pass; maximum normalized ratio 0.999999999996 |
| Acceptance monotonicity on the subpercent grid | pass |

The broader δ=0 Table-3 online code path was not rerun as a new live mission; the exact δ=0 arithmetic branch and all plan-level bounds above match the archived implementation. This is the remaining protocol-level regression if development continues.

## E1 — prescribed Table-2 extension

Each cell below aggregates the six unchanged \((d,\bar w)\) rows. “Maximum horizon” is the farthest any cell closed; failure to close is not a simulated counterexample.

| δ | Method | Full-hour accepted | Maximum horizon (min) | certificate wall time min/median/max (s) |
|---:|---|---:|---:|---:|
| 0% | Original | 2/6 | 60.0 | 2.29/7.45/14.48 |
| 0% | Positive/queue | 3/6 | 60.0 | 3.34/9.40/14.96 |
| 0% | Signed generator | 5/6 | 60.0 | 6.45/17.25/22.69 |
| 2% | Original | 0/6 | 23.5 | 3.19/3.75/6.15 |
| 2% | Positive/queue | 0/6 | 28.0 | 2.99/4.46/7.61 |
| 2% | Signed generator | 0/6 | 31.0 | 3.87/7.46/11.55 |
| 5% | Original | 0/6 | 11.2 | 2.12/2.68/972.22 |
| 5% | Positive/queue | 0/6 | 13.5 | 2.62/3.06/4.85 |
| 5% | Signed generator | 0/6 | 14.2 | 3.13/4.74/5.86 |
| 10% | Original | 0/6 | 5.8 | 1.73/2.19/2.45 |
| 10% | Positive/queue | 0/6 | 7.2 | 1.98/2.51/2.86 |
| 10% | Signed generator | 0/6 | 8.0 | 2.51/3.44/4.03 |

One original-method 5% failure required 972.22 s; it is retained in the record. No tolerance, partition, horizon or target was changed.

## E2 — 1700-vehicle binding service

The nominal planner found plans through a 1550-vehicle synthesis target; the lower four targets were infeasible under the unchanged optimizer.

| Nominal synthesis target (veh) | Plan found | Nominal TTS (veh-h) | Nominal deadline n1 (veh) |
|---:|---:|---:|---:|
| 1700 | yes | 4439.659262357894 | 1700.0024072489787 |
| 1600 | yes | 4585.998243946203 | 1600.0012312425902 |
| 1575 | yes | 4631.316084752902 | 1575.0006534360862 |
| 1550 | yes | 4706.303908926576 | 1550.000715871564 |
| 1500 | no | --- | --- |
| 1450 | no | --- | --- |
| 1400 | no | --- | --- |
| 1300 | no | --- | --- |

At both 2% and 5%, none of the original, positive/queue or signed-generator methods certified any feasible target. Consequently there is no “first certified target,” internal certified service margin, or robust TTS comparison to report. Complete per-certificate wall times and stopping causes are in `required_E2_binding_service.csv`.

## E3 — 5% online missions

| Required initial condition | Result |
|---|---|
| Full-mission 5% certificate at epoch 0 | unavailable for every method and Table-2 contract |
| Candidate admissions | 0 |
| Authorized online executions | 0 |
| Realized storage/service outcomes | undefined under the certified protocol because execution authority is never established |

## E4 — out-of-contract stress

E4 presupposes a retained 5% certified set. Since no such set exists, the requested fresh-box noncontainment count is undefined. A clearly separated plant-only diagnostic was still run with 100 stochastic seeds plus all four constant sign corners and a five-minute alternating profile at each true uncertainty level:

| True uncertainty | Realizations | Storage violations | Service violations | Class-negativity violations |
|---:|---:|---:|---:|---:|
| 7.5% | 105 | 0.0% | 0.0% | 0.0% |
| 10.0% | 105 | 0.0% | 0.0% | 0.0% |
| 15.0% | 105 | 0.0% | 0.0% | 0.0% |

These sampled zero-violation outcomes show conservatism of the current enclosure; they do not replace the missing certificate.

## E5 — minute-20 repair (optional protocol item)

| Plan | δ | Accepted | Certified service upper (veh) | First failure |
|---|---:|---:|---:|---|
| nominal selected repair | 0.00% | yes | 2849.0934657666435 | --- |
| nominal selected repair | 0.10% | no | 2930.7576400279363 | scheduled_service_not_certified |
| nominal selected repair | 0.25% | no | 3090.6982884754707 | scheduled_service_not_certified |
| nominal selected repair | 0.50% | no | 3551.8826862880765 | scheduled_service_not_certified |
| nominal selected repair | 1.00% | no | --- | continuous_physical_constraint_not_certified |
| retained backup | 0.00% | yes | 2310.6589692719735 | --- |
| retained backup | 0.10% | yes | 2422.971221886599 | --- |
| retained backup | 0.25% | yes | 2687.3190831443676 | --- |
| retained backup | 0.50% | no | 4195.689901504256 | scheduled_service_not_certified |
| retained backup | 1.00% | no | --- | continuous_physical_constraint_not_certified |

The minimum-intervention nominal repair has only 0.907 vehicles of certified slack and loses the service certificate at 0.1%. The retained backup stays certified through 0.25% on the tested grid, then loses the service row at 0.5%.

## Machine-readable records

- `required_E1_grid.json/.csv`: every prescribed E1 certificate, stopping cause and wall time.
- `required_E2_binding_service.json/.csv`: all feasible target plans and every E2 certificate.
- `required_E3_E4_status.json`: the blocked certified protocol and separate plant-only diagnostic.
- `mfd_uncertainty_results.json`: subpercent frontier, repair study and 200-run containment audit.
- `containment_certificate_delta_0p005.json`: complete retained interval proof.
- `test_mfd_uncertainty.py`: deterministic regression assertions.
