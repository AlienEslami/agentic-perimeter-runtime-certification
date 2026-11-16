REVIEWER FOLLOW-UP STUDY — 11 NOVEMBER 2026

The publication changes use probe.py's tight12 variant only. The traffic model,
plans, constraint values and formal acceptance inequalities are unchanged.
The structured_probe.py and coordinate_probe.py files, where present in the
working directory, are exploratory rejected alternatives and are not the
implementation reported in the manuscript.

Refinements
1. Integer powers are bounded using parity; a square crossing zero is nonnegative.
2. Batched interval matrix products retain the original product/summation order
   and nextafter outward rounding. 400 test products match bit for bit.
3. Matrix series use order 12 instead of 18, retaining explicit verified tails.
4. Trial-radius search uses 1% inflation and at most 64 attempts rather than
   10% and 24. Acceptance still requires strict containment and every constraint.
The baseline ignores the probe runner's requested padding argument and executes
the archived 10% inflation. Its unchanged implementation is included under vendor.

Main results
- All 160 nominal/full-scenario decisions are retained: 36/80 and 50/80.
- Same four-box partition: regional endpoint widths decrease about 30–34%.
- Both congested plans pass an unsplit destination check after refinement.
  This checks physical constraints and the pending Region-1 deadline. Its bounds
  are wider than the four-box bounds, which separately establish both regions'
  recovery below critical accumulation. Do not conflate these two comparisons.
- Serial paired verification times: baseline 70.4–78.1 s; refined 17.6–19.4 s.
  These are two repetitions per plan. Planning, sensing and actuation are excluded.
  The delayed-activation study was not rerun; its missed deadlines remain recorded.
- At regional state bound 25 vehicles and queue bound 10 vehicles, arrival error
  600 veh/h permits a certified backup and weight-0.65 repair over the full hour.
  Earlier weights through 0.60 violate an exact necessary queue bound.
- The fixed weight-0.35 plan passes with regional component bounds of 30 vehicles
  and arrival error 300 veh/h. Its backup check is inconclusive at that state bound.
- At 900 veh/h the fixed weight-0.95 candidate passes, but its backup check does
  not. This is an individual plan certificate, not repair-availability evidence.
- Additional failed probes and the uncertainty refinements are exploratory,
  not held-out evidence or estimates of the largest possible uncertainty bound.
- Plant sensitivity scales MFD outflow and internal demand by 0.95/1/1.05.
  The backup passes 5760/5760 tested physical/service cases. Repair passes all
  physical cases but only 5480/5760 deadlines. The 280 misses occur under the
  combined -5% outflow/+5% demand plant. These perturbations are outside the
  certificate's model contract; this is not a microscopic simulation study.

Reproduction
Keep the sibling benchmark directories in the delivered archive together.
Install numpy, scipy, matplotlib and mpmath. From this directory:
  python validate.py
  python run_comparison.py --workers 4
  python uncertainty_followup.py
  python four_box_bounds.py
  python arrival_extension.py
  python queue_witnesses.py
  python extension_validation.py
  python plant_sensitivity.py
  python timing.py
Run timing.py alone, after all other numerical workers have finished. Its paired
order is reversed in the second repetition. Existing run_comparison result files
are reused; use a fresh extraction without comparison/*.json for new runs.
For checking saved summary decisions alone, read decision_summary.json.

The compact archive includes frozen inputs, baseline summary decisions, study
sources, selected full certificates, validation results, and all aggregate
outcomes. Full per-case failed and passing traces are retained in the working
directory comparison/ (roughly 0.9 GB) and can be regenerated from the inputs.
No new LLM calls were made. No GitHub upload was performed in this revision.
