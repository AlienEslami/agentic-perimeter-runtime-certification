# Sequential tool-authority experiment (1 October 2026)

## What this tests

Can an agent change planning tools and soft regional priorities during one
urban-perimeter incident while the same hard physical limits and absolute
regional deadline remain in force? The experiment uses the archived seven-state
queue-aware MFD plant and existing computed plans. A verified incumbent starts
at minute 20. The agent requests a stronger scenario-MPC command at minute 25
and a nominal-MPC command at minute 30. The supervisor installs each request
only after a continuous-time physical and deadline certificate passes; a failed
request leaves the active, previously verified plan and its original expiry
unchanged. The initial state ranges over a whole box, and offered demand has a
bounded residual. There is no new measurement at either later decision.

The [protocol](../../SEQUENTIAL_AUTHORITY_PROTOCOL.md) was written before the
three pilot cases were run. Both later requests use exact second-block
commands from archived plans, not fresh reoptimization. All 640 initial-corner
and residual-profile realizations were replayed for each schedule. Formal
authority comes from the accepted certificates and retention rule; sampled
replays provide an independent implementation check.

## Results

| Matched hard contract | Minute 25 / 30 | Unchecked switching: physical violations / 640 | Certified switching: physical and service passes / 640 | Median TTS, fixed / unchecked / certified (veh-h) |
|---|---|---:|---:|---:|
| D015/D014 | accept / accept | 0 | 640 / 640 | 3195.51 / 3080.63 / 3080.63 |
| D027/D026 | accept / retain | 0 | 640 / 640 | 3037.44 / 3081.13 / 2716.70 |
| D071/D070 | retain / retain | 0 | 640 / 640 | 3168.85 / 3093.97 / 3168.85 |
| D025/D024 (discovered stress case) | accept / retain | **128** | **640 / 640** | 2962.52 / 3069.75 / 2718.84 |
| D043/D042 (discovered stress case) | retain / retain | **128** | **640 / 640** | 2872.60 / 3069.60 / 2872.60 |
| D061/D060 (discovered stress case) | retain / retain | **64** | **640 / 640** | 2658.85 / 3070.53 / 2658.85 |

The fixed comparator holds the original certified two-block plan through
minute 60. The unchecked comparator executes both requested tool changes at
minutes 25 and 30. All three schedules face the same hard contract and the
same time-varying soft-priority weights within each matched pair. "Service"
means the Region-1 accumulation deadline at minute 60. The physical failures
in this table are exterior-queue storage violations; no sampled service
violations occurred in the scan.

The clean flexibility case is D015/D014. Both requested changes pass, the
certified schedule is identical to unrestricted switching, and median
queue-inclusive total time spent (TTS) falls 3.59% relative to holding the
initial certified plan. The strongest continuity case is D025/D024. The
minute-25 scenario-MPC replacement receives a new whole-family proof. The
minute-30 nominal-MPC request fails the exterior-queue proof, so the accepted
scenario-MPC suffix continues under its existing certificate. Its median TTS
is 2718.84 veh-h, versus 2962.52 for holding the original plan and 3069.75
for unchecked switching. The changing-priority objective is higher for the
certified continuation (6205.84 versus 5897.79 for unchecked switching): the
checker preserves the hard contract at the cost of the agent's later soft
preference in this event. This tradeoff belongs beside the TTS result.

The D025/D024 unchecked schedule exceeds the 2000-vehicle queue limit in
128/640 sampled realizations. An independent adaptive DOP853 integration of
the worst displayed corner with a constant-high residual gives a 2005.590
vehicle peak at minute 60, agreeing with the separate 10-second RK4 replay to
0.00033 vehicles. The corresponding queue peaks are 2009.621 and 2017.599
vehicles for D043/D042 and D061/D060; separate adaptive integrations confirm
these as well. The accepted/retained schedules have 640/640 sampled physical
and service passes in all three stress cases, backed by the installed
certificates. Their initial sensing half-widths are 1, 10 and 25 vehicles,
respectively, all with a 300 veh/h demand-residual bound and the same
2750-vehicle Region-1 deadline. D025's accepted new proof is valid through the original
minute-60 deadline; its reported proof time is about 11 seconds, and the
rejected request's check takes about 9 seconds. This experiment places
requests at their activation times and does not model those seconds of online
latency. A deployed timed switch would use future-dated certification.

The exploratory minute-30 repair check could not establish a new proof even
for its incumbent endpoint from the widened minute-30 physical box. The
propagated box has lost correlations that the incumbent's original proof
retains. The grid repair was therefore not run or counted as successful;
retention of the original correlated proof is the valid action. This is a
distinct reason to preserve proof-linked plan authority across decisions.

The preselected D027/D026 case illustrates the price of conservative proof:
the minute-30 request fails the computed service enclosure even though all
640 sampled unchecked trajectories satisfy the deadline and physical rows.
D071/D070 rejects both replacements because the propagated outer box includes
physically inadmissible states; the incumbent's correlated original proof
still authorizes its continuation. These outcomes prevent interpreting every
rejection as a realized danger.

## Full-grid discovery accounting

The corrected fixed-grid scan contains 36 matched weight-4/weight-2
development pairs; 30 have the archived accepted source certificates needed
for this particular sequence. Three of the 30 have sampled physical violations
under unchecked switching: D025/D024, D043/D042 and D061/D060. All three were
run through the full sequential proof procedure and are reported above. The
initial scan mistakenly treated adjacent held-out IDs as matched pairs even
though they differ in hard deadline and have the same soft weight. The
eligibility rule was corrected before interpreting the scan; held-out cases
are excluded. These 30 pair results are a fixed-grid discovery study, not an
estimate of field failure probability or independent held-out validation.

## Interpretation for the paper

The result supports a specific architectural claim: an agent may request a
different planning method as objectives change, receive a new proof when its
command is admissible, and retain a previously certified continuation when a
later request is not. D015 shows admission without traffic-performance loss
from checking. D025 shows both cross-tool admission and a later rejection
that prevents a sampled queue overflow, while preserving the regional
deadline. The experiment also shows the associated soft-priority tradeoff and
instances of proof conservatism.

A planning tool's own constraints apply to the state, uncertainty description
and horizon on which that tool solved its plan. They do not, by themselves,
authorize a later handover from another executed prefix. Here the handover
check uses the inherited reachable state set and the original absolute
deadline, which is the additional function supplied by the architecture.

It does **not** establish that this architecture outperforms a fully robust,
receding-horizon MPC with the same uncertainty information, terminal
feasibility conditions and replanning opportunities. The stronger MPC source
plans here were computed at minute 20 and reused as proposals; they were not
reoptimized at minutes 25 and 30. A matched reoptimizing comparator requires
adapting the 641-scenario solver to the newly propagated information sets and
remaining horizons. Nor does this run include a live LLM; the request sequence
is a deterministic stand-in for agent choices. The current manuscript was
left unchanged pending a decision about which of these results to include.

## Reproduction files

- `run_sequential_authority.py`: full chained certification and executed
  640-trace replay, with all attempted proofs retained.
- `analyze_sequential_authority.py`: matched fixed, unchecked and certified
  schedule accounting.
- `scan_sequential_grid.py`: all-pair discovery scan with hard-contract
  equality checks.
- `validate_sequential_witness.py`: independent high-accuracy overflow
  witnesses.
- This directory: individual case records, scan, counterfactual summaries and
  witness records.
