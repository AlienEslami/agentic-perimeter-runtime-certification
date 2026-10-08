# Reproducing the reduced paper

This guide connects the experiments retained in *Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control* to their protocols, programs and recorded results. It also identifies the secondary studies preserved outside the reduced manuscript. The reduction changes the presentation; the recorded plans, certificates, traffic realizations and agent responses remain available.

The [long manuscript](../main.tex) supplies the extended explanations. The supporting source extracts preserve the [implementation details from Appendices A–F](implementation_details.tex) and [experimental details from Appendices G–J](experimental_details.tex). The [complete preceding reduced-paper package](../../archive/reduced_manuscript_2026_10_08_before_substantial_reduction/) preserves the version before this substantial reduction, including its original appendix text.

## Start with the saved evidence

Use Python 3.12. From the repository root, install the recorded numerical environment and check the copied research records:

```text
python -m pip install -r benchmarks/reusable_perimeter_suite_2026-09-26/requirements-lock.txt
python -B scripts/audit_repository_records.py
```

The [repository audit](../../scripts/audit_repository_records.py) verifies file hashes and certificate bindings and parses Python sources. It does not solve new planning problems or contact a model API. It establishes the integrity of the stored material within its stated scope.

For model checks and independent numerical replays, change to the [portable suite](../../benchmarks/reusable_perimeter_suite_2026-09-26/):

```text
cd benchmarks/reusable_perimeter_suite_2026-09-26
python -B audit.py
python -B audit.py --replay
```

The first command checks the physical model, derivatives and exact gate constraints. The second checks selected saved policies using an independent adaptive integrator and verifies their command and proof bindings. They write local audit results. They make no API calls and perform no optimization sweep. The saved reports are [initial_audit.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/initial_audit.json) and [independent_replay_audit.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/independent_replay_audit.json). Their coverage is a documented selection of policies, rather than every trajectory in every study.

Certificates establish the stated guarantee for all initial states and measurable disturbances covered by the certified contract. Numerical replays assess implementation and sampled performance. Keep these two sources of evidence distinct when reporting the results.

## Where each retained study is recorded

All paths below refer to this repository. The suite's [summary.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/summary.json) contains the final synchronous-study counts and performance summaries.

| Retained study | Question and study size | Protocol and source | Saved evidence |
| --- | --- | --- | --- |
| Queue-inclusive planning, Task Q | Does checking leave an already feasible economic MPC plan unchanged? Twelve queue-only contracts vary pulse amplitude and duration. | [PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/PROTOCOL.md); [run_contracts.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/run_contracts.py). | [results/contracts/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/contracts/), files `Q000.json` through `Q011.json`; `summary.json` field `queue`. All twelve nominal plans are certified. |
| Conservation and tighter bounds | How do interval, positivity/queue, and signed-correlation methods change certification of the same fixed plan? The main comparison uses six sensing/arrival uncertainty pairs. | [POSITIVE_REACHABILITY_PROTOCOL.md](../../benchmarks/mfd_uncertainty/POSITIVE_REACHABILITY_PROTOCOL.md); [DESTINATION_STRIP_ADDENDUM.md](../../benchmarks/mfd_uncertainty/DESTINATION_STRIP_ADDENDUM.md); [experimental_details.tex](experimental_details.tex). | [required_E1_grid.json](../../benchmarks/mfd_uncertainty/reproducibility/results/mfd_uncertainty/required_E1_grid.json), **only rows with `delta == 0`**. Also [mfd_incident_affine_reachability_v3_results.json](../../benchmarks/mfd_uncertainty/reproducibility/results/mfd_incident_affine_reachability_v3_results.json) and its retained source snapshots. |
| Binding service margin | Does the tighter certificate allow a lower-cost plan when the service deadline is binding? Six nominal planning targets are compared under the same 1,700-vehicle service limit. | Original `REACHABILITY_SERVICE_MARGIN_PROTOCOL.md` inside the [online-structure archive](../../archive/historical_studies/online_structure_2026_09_10.zip); [matching service-study driver](../../benchmarks/reusable_perimeter_suite_2026-09-26/vendor/mfd_reachability_service_margin.py). | Original `mfd_reachability_service_margin_results.json`, all method proof files, and `mfd_affine_service_margin_audit_results.json` inside the same archive. Restoration and checking are described below. |
| Deadline contracts and checked repair, Task D | How much service coverage does checked repair recover, and what is the cost of the requested priority? Eighty contracts comprise 72 development and eight held-out settings. | [PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/PROTOCOL.md), [FULL_SCENARIO_PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/FULL_SCENARIO_PROTOCOL.md), [RESERVE_PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/RESERVE_PROTOCOL.md). | [results/contracts/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/contracts/), `D012.json` through `D091.json`; [results/full_scenario/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/full_scenario/); [results/reserve_refresh/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/reserve_refresh/). |
| Deadline/queue conflict | Why does sampled predictive feasibility still require a certificate? `D027` is the disclosed conflict example. | Same Task D and full-scenario protocols; [run_full_scenario.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/run_full_scenario.py). | [contracts/D027.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/contracts/D027.json), [full_scenario/D027.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/full_scenario/D027.json), and `summary.json` field `primary_case`. |
| Successive planning requests | Can a beneficial replacement be accepted while a later unsafe request is rejected? `D025_D024` is the main stress case, accompanied by the scan of 30 eligible matched pairs. | [SEQUENTIAL_AUTHORITY_PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/SEQUENTIAL_AUTHORITY_PROTOCOL.md); [run_sequential_authority.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/run_sequential_authority.py). | [REPORT.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/REPORT.md), [D025_D024.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/D025_D024.json), its [matched counterfactuals](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/D025_D024_counterfactuals.json), [independent witness](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/D025_D024_independent_witness.json), and [full_grid_scan.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/full_grid_scan.json). |
| Execution during computation | Does execution remain feasible while replacement preparation takes time? Seventeen contracts are tested at three handover delays, giving 51 cases. | [DELAYED_PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/DELAYED_PROTOCOL.md); [run_delayed_handover.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/run_delayed_handover.py). | [results/delayed_handover/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/delayed_handover/), [delayed_handover_summary.json](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/delayed_handover_summary.json), and [DELAYED_HANDOVER_CASES.csv](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/DELAYED_HANDOVER_CASES.csv). |
| Agent interpretation and actual planning | Do interface and context errors affect requested objectives, and can the physical/service checker still control execution? The interpretation pilot has 96 calls; the actual planning experiment has eight traces and 13 calls. | [API_PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/API_PROTOCOL.md), [run_api.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/run_api.py), and [run_end_to_end_api.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/run_end_to_end_api.py). | [results/api/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/api/) and [results/end_to_end_api/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/end_to_end_api/). Both contain summaries; the latter includes raw calls and operational traces. |
| Connected rings | Does the class/inventory method carry over to more regions and entry queues? Two- and three-region rings, two methods and four uncertainty settings give sixteen certificate attempts. | [MULTIREGION_TRANSFER_PROTOCOL.md](../../benchmarks/connected_network_2026-09-11/MULTIREGION_TRANSFER_PROTOCOL.md); [mfd_multiregion_transfer.py](../../benchmarks/connected_network_2026-09-11/reproducibility/mfd_multiregion_transfer.py). | [mfd_multiregion_transfer_results.json](../../benchmarks/connected_network_2026-09-11/reproducibility/results/mfd_multiregion_transfer_results.json), compressed individual proofs, and the [independent audit source](../../benchmarks/connected_network_2026-09-11/reproducibility/audit_mfd_multiregion_transfer_independent.py). |

## Rerun the two-region traffic experiments

The numerical programs are checkpointed and reuse existing result files. For a fresh run, make a separate working copy of the portable suite's source files, `vendor/`, protocols and `anchor.json`, without copying `results/`. Keep the archived results for comparison. The supplied fixture is sufficient: `prepare_anchor.py` expects the authors' older directory tree and is not part of the portable workflow.

From that suite working directory, the following sequence reproduces the retained synchronous studies and required reserve plans:

```text
python run_contracts.py --only Q --workers 2
python run_contracts.py --only D --workers 2
python run_full_scenario.py --audit
python run_full_scenario.py --workers 2
python run_reserve_refresh.py --workers 2
python select_certified_tools.py
python audit.py --replay
```

The queue family uses pulse amplitudes 20%, 30%, 40% and 50%, each lasting 30, 45 or 60 minutes. The deadline development grid varies sensing half-width, bounded arrival residual, the service limit and the requested priority weight; the eight held-out settings use intermediate values. Exact menus and plan blocking are in `PROTOCOL.md`. Planning and service clocks must remain aligned with the supplied contract.

The full-scenario comparator uses all 128 initial corners and five residual profiles for constraints, with a nominal center trajectory for its objective. The repair search tests the disclosed interpolation grid toward a backup certified for the same contract. `run_reserve_refresh.py` creates independently checked fresh reserves where the archived continuation does not certify under the widened contract.

Each constant-plan performance comparison has 640 realizations, formed from 128 initial corners and five residual profiles. The feedback CLF comparison has 80 realizations from its separately defined 16 corners and five profiles. The physical CBF-QP has one matched adverse trajectory per deadline contract. These denominators must be reported separately. The suite's README explains the local implementation and queue-model extension used for each comparator.

### Sequential replacement and the full pair scan

Keep the Task D and full-scenario source results in place. The sequential experiment uses archived plans and applies a fresh certificate to each proposed remaining portion at its actual decision time. It preserves the earlier service deadline. The main `D025_D024` history starts from a certified repair at minute 20, accepts the lower-priority replacement at minute 25, and rejects the unsafe higher-priority nominal request at minute 30.

From the suite working directory:

```text
python run_sequential_authority.py
python run_sequential_authority.py --pair D025
python run_sequential_authority.py --pair D043
python run_sequential_authority.py --pair D061
python analyze_sequential_authority.py
python analyze_sequential_authority.py --pair D025
python analyze_sequential_authority.py --pair D043
python analyze_sequential_authority.py --pair D061
python scan_sequential_grid.py
python validate_sequential_witness.py
```

The analysis programs replay the stored plans to compare a retained plan, unchecked replacement and certified replacement; they make no model calls. The independent witness program recomputes the disclosed storage violation using a separate adaptive integration setting. The saved [repair extension](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/sequential_authority_2026-10-01/D025_D024_repair_extension.json) documents the additional checked-repair comparison for the same main case; its source is [test_sequential_repair.py](../../benchmarks/reusable_perimeter_suite_2026-09-26/test_sequential_repair.py).

The corrected scan compares development pairs with identical hard contracts and priority weights 4 and 2. Of 36 such pairs, 30 have both required source certificates; three of those 30 show a storage violation under unchecked replacement. The held-out priority-3 cases do not form these matched pairs. The archived report explains the earlier pairing correction; use the corrected `full_grid_scan.json` for the paper's denominator.

### Future-dated handover

After the synchronous studies and fresh reserves have completed:

```text
python run_delayed_handover.py --workers 2
```

The three delays are 60, 120 and 300 seconds. The incumbent continues to execute while its complete initial information set is propagated to the proposed activation time; the study does not supply a fresh observation at handover. The service deadline remains minute 60. A replacement that finishes after its slot is discarded for that activation. The recorded outcomes are 30 replacements and 21 retentions across 51 cases, with all 32,640 sampled physical and service checks passing. Preparation times are measurements of the recorded hardware and workload; a rerun can change timing and consequently activation decisions.

## Reconstruct the structure and binding-service comparisons

### Select the paper's uncertainty comparison

The folder `benchmarks/mfd_uncertainty/` also includes a later diagnostic experiment with uncertain MFD parameters. The paper's six-row comparison varies sensing and arrival uncertainty with nominal MFD parameters. In `required_E1_grid.json`, select `delta == 0`; the other rows add 2%, 5% or 10% multiplicative MFD uncertainty and belong to that separate diagnostic study.

The six paper settings are `(state half-width in vehicles, arrival residual in vehicles/hour)`:

```text
(1, 60), (10, 60), (50, 60), (100, 60), (1, 300), (25, 300)
```

Each has three method rows. The stored zero-delta bounds reproduce the paper's displayed service bounds after upward rounding to whole vehicles. A failed certificate remains a failed certificate, even when sampled trajectories are feasible. The earlier coordinate comparison and its definitions are retained in `experimental_details.tex` and the preceding manuscript archive.

For an optional numerical recalculation restricted to the paper's nominal-MFD comparison, use a separate copy of the uncertainty folder and run this from its `reproducibility/` directory:

```text
python -c "import run_mfd_uncertainty_required_protocol as s; s.DELTAS=(0.,); s.OUTPUT=s.OUTDIR/'paper_nominal_structure_grid.json'; s.main()"
```

This uses the [existing driver](../../benchmarks/mfd_uncertainty/reproducibility/run_mfd_uncertainty_required_protocol.py), its archived first MPC plan and the six declared settings, while writing a separate result file. Running the driver with its default menu additionally computes the later MFD-uncertainty diagnostic and should be labeled accordingly. Source versions and snapshots remain attached to each result.

### Restore the original binding-service records

The original binding-service study is preserved in `archive/historical_studies/online_structure_2026_09_10.zip`. Extract that ZIP into a separate work directory, preserving its contained paths. Its study root is:

```text
output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-10_ONLINE-STRUCTURE/
```

Within that root, read `REACHABILITY_SERVICE_MARGIN_PROTOCOL.md` and `reproducibility/results/mfd_reachability_service_margin_results.json`. The record stores all six nominal targets, commands, optimizer outcomes, independent replays and the three certificates per selected plan. The service limit is 1,700 vehicles; targets are 1,700, 1,690, 1,675, 1,650, 1,625 and 1,600. The best certified points in the recorded grid are 1,600 for the original and positivity/queue methods and 1,625 for the signed method. The signed method's recorded nominal TTS improvement over the original method's best certified grid point is about 0.90%.

The archive preserves the driver dependencies and proof files, but the study driver itself is supplied separately as `benchmarks/reusable_perimeter_suite_2026-09-26/vendor/mfd_reachability_service_margin.py`. Copy that file into the extracted study root's `reproducibility/` directory. Its SHA-256 is:

```text
83bd32ce406c2849c1d1b4d9157b02ae62bfaf6cbc7b4a6e97b40a4e160d6b5d
```

This matches the original result's recorded dependency. From the extracted study root, check the existing record and replay each stored selected plan without optimization or API calls:

```text
python reproducibility/mfd_reachability_service_margin.py --audit-only
```

That audit verifies the source/protocol hashes, exact command bindings, method-specific proof descriptors and independently integrated trajectories, then writes a local audit report. The archived `mfd_affine_service_margin_audit_results.json` preserves the additional original checker audit. To recompute the planning grid, use another restored working copy with the study's own service result and corresponding proof outputs omitted, and run:

```text
python reproducibility/mfd_reachability_service_margin.py
```

Preserve the other archived inputs, source snapshots and protocol files. Historical scripts rely on this restored layout; the portable suite's normal entry points cover the later Task Q/D experiments.

## Check or rerun the connected rings

From the repository root:

```text
python -B scripts/unpack_connected_network_records.py
python -B benchmarks/connected_network_2026-09-11/reproducibility/audit_mfd_multiregion_transfer_independent.py
```

The [unpacker](../../scripts/unpack_connected_network_records.py) verifies the compressed proof hashes and reconstructs their original `.json` files. It refuses to overwrite differing records. These generated files are ignored by Git. The independent audit checks the stored certificates and physical class balances and runs adaptive numerical replays; it makes no model calls or optimization sweep.

The two- and three-region ring models have six and twelve class/queue states respectively. Both certification coordinates use the same one-hour fixed plan, its change at minute 30, and the same four uncertainty pairs `(1,60)`, `(10,60)`, `(25,60)`, `(25,300)`. The arrival pulse runs from minutes 15 to 45. The complete model, initial state, allocation rules and service settings are in `MULTIREGION_TRANSFER_PROTOCOL.md` and `experimental_details.tex`.

For a fresh numerical certificate sweep, make a separate copy of the connected-network folder with its source dependencies and required inputs, and omit the previous transfer result/proof outputs. From the repository root of that working copy:

```text
python benchmarks/connected_network_2026-09-11/reproducibility/mfd_multiregion_transfer.py
```

The driver also supports `--audit-only` for its primitive model checks. The ring study uses fixed plans rather than solving a new network optimizer. Its acceptance counts and timings are specific to the declared models, plans and uncertainty settings.

## Inspect agent records offline; make new live calls only deliberately

The 96-call pilot tests interpretation and routing over precomputed plans. The eight actual traces instead use agent requests to parameterize the planning, checking and repair tools. Keep their results separate. Raw prompts, response content, tool schemas, model identifiers and operational results are saved in the two result directories linked above.

The recorded model request is `gpt-5.4-mini`; responses retain their actual model snapshot identifiers. The study uses reasoning setting `none`, a 400-output-token cap and a strict single-tool interface. The historical [budget ledger](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/api/budget_ledger.json) records the shared study charges. Reading these files, checking recorded certificate bindings, and replaying stored traffic plans require no credentials and make no model requests. There is no separate supplied command that guarantees a byte-identical offline reconstruction of every model response; the saved responses are the record of those calls.

For an optional new live experiment, read `API_PROTOCOL.md`, check the selected model's availability and current pricing, and set `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` locally. Keep credentials outside the repository. From a separate suite working copy with the required completed traffic studies:

```text
python run_api.py --limit 96
python run_end_to_end_api.py
```

These two commands can make paid network requests. They reuse existing call files when present; an empty call-output directory in a separate working copy is needed for a genuinely new experiment. Retain its own responses, model identifiers and ledger. New model outputs need not match the recorded calls. The preliminary `results/initial_expiry_alignment/` and `results/initial_quadrature/` records are preserved but excluded from the final reported denominators.

## Secondary studies preserved outside the reduced presentation

| Study | Why it is retained | Location and rerun scope |
| --- | --- | --- |
| Task R: 24 native regional-regulation cases | Supplies the original CLF/tracking-MPC calibration and comparator context. | [results/regulation/](../../benchmarks/reusable_perimeter_suite_2026-09-26/results/regulation/); the Task R specification is in [PROTOCOL.md](../../benchmarks/reusable_perimeter_suite_2026-09-26/PROTOCOL.md). From the portable suite, use `python run_regulation.py` in a working copy for a new numerical run. |
| Eight-event MPC compatibility | Records the earlier comparison showing how certification can sit around existing MPC plans. Its contracts and clock differ from the current reusable suite. | [mpc_compatibility_2026_09_10.zip](../../archive/historical_studies/mpc_compatibility_2026_09_10.zip), including the original certified/nominal grids, commissioned library and audits. The protocol text is preserved in `experimental_details.tex` and the preceding manuscript archive. Read the original layout/dependencies before attempting a historical rerun. |
| Seventeen online certification cases | Preserves the full comparison of certification methods on successive proposals, proof records and the original twelve-request sequence. | [online_structure_2026_09_10.zip](../../archive/historical_studies/online_structure_2026_09_10.zip). Inside the restored study root, read `STRUCTURE_ONLINE_RUNTIME_PROTOCOL.md`, `ONLINE_STRUCTURE_REVISION_REPORT.md`, `reproducibility/results/mfd_structure_online_runtime_results.json`, and the per-case `reproducibility/results/sor/` records. The archive preserves original code and independent validation; it does not imply a portable current-root entry point. |

The [archive README](../../archive/README.md), [standalone materials map](../STANDALONE_MATERIALS_MAP.md) and [preservation manifest](../../archive/PRESERVATION_MANIFEST_2026-10-07.json) identify the additional historical local-agent, routing and fixed-controller/repair material. Their archived denominators and assumptions remain separate from the current experiments.

The substantial manuscript reduction introduces no new optimization runs, traffic simulations or live model calls. Commands in this guide describe the existing supplied interfaces and their scope; the published evidence is the saved, source-bound research record.
