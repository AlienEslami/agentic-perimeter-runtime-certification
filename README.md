# A Formal Guarantee for Trustworthy Agentic Traffic Control: A Multi-region Perimeter Case

## Current manuscript - 15 November 2026

- [Reduced manuscript, 48 pages](paper/reduced_2026-11-15/main.pdf)
- [LaTeX source package](paper/reduced_2026-11-15/PERIMETER_PARTB_SUBMISSION_CLARIFIED_2026-11-15.zip)
- [Current results and reproduction guide](paper/supporting_materials/REPRODUCTION_GUIDE_2026-11-15.md)
- [Dated release](https://github.com/AlienEslami/agentic-perimeter-runtime-certification/releases/tag/paper-2026-11-15)
- [Revision notes](paper/reduced_2026-11-15/revision_notes.txt)

This research repository contains the manuscript, traffic models, planning
and verification implementations, experimental protocols and saved results.
The current version includes traffic trajectories and inputs, ensemble
controller comparisons, model-error sensitivity and recorded agent studies.
Earlier complete and reduced manuscripts are preserved at their existing
paths. Their descriptions below refer to historical snapshots.

## Contents

- `paper/`: the long manuscript (non-IEEE and IEEE reading layouts), a separate reduced version, essential appendices, figures, declarations, compiled PDFs and reduction notes.
- `archive/`: the preceding companion manuscript and losslessly preserved historical studies, with preservation hashes.
- `benchmarks/reusable_perimeter_suite_2026-09-26/`: the portable two-region suite, unchanged model/checker sources in `vendor/`, protocols, exact plans, certificates, trajectories, and recorded live-agent responses.
- `benchmarks/mfd_uncertainty/`: earlier uncertainty-frontier sources, inputs, source snapshots, and results.
- `benchmarks/connected_network_2026-09-11/`: the complete sixteen-attempt ring study, unchanged source dependencies and snapshots, and independent audit. Large proof records are losslessly compressed as `.json.gz`.
- `provenance/anchor_2026-09-11/`: compressed archived run and continuation proof used to extract the portable minute-20 fixture.
- `scripts/`: read-only repository checks, a proof unpacker, and the historical handover finalization script.
- `REPRODUCTION_MANIFEST.json`: original/stored hashes, compression metadata, and omitted cache files.
- `PUBLIC_RELEASE_CHECKLIST.md`: release status and remaining submission decisions.

## Offline checks

Use Python 3.12. The portable suite supplies `requirements.txt` and the study environment in `requirements-lock.txt`. From the repository root:

```text
python -m pip install -r benchmarks/reusable_perimeter_suite_2026-09-26/requirements-lock.txt
python -B scripts/audit_repository_records.py
```

The repository audit reads copied records, verifies hashes and certificate bindings, and parses Python sources. It makes no API calls and runs no optimization sweep. From the portable-suite folder, use `python -B audit.py` for physical-model/derivative checks and `python -B audit.py --replay` for independent adaptive replays of selected saved policies. Those commands write local audit results. Each audit has its own documented scope.

## Numerical reruns

Read each study's README and protocol first. The portable suite covers native regulation, queue-inclusive planning, deadline contracts, all-corner/profile predictive comparison, checked repair and reserve refresh, delayed handover, sequential authority, and retained later diagnostics. Its core sequence, run from the suite folder, is:

```text
python run_regulation.py
python run_contracts.py --workers 2
python run_full_scenario.py --audit
python run_full_scenario.py --workers 2
python run_reserve_refresh.py --workers 2
python select_certified_tools.py
python run_delayed_handover.py --workers 2
```

Runs are checkpointed and reuse existing results. Use a separate copy of the source folders and `anchor.json`, without copied results, for a fresh run. Preserve this repository's archived results for comparison. Additional studies have their own `*_PROTOCOL.md` files, including sequential authority and falsification. Runtime measurements depend on hardware and concurrent load.

The connected-ring records retain the source-relative paths used by the original audit. From the repository root:

```text
python -B scripts/unpack_connected_network_records.py
python -B benchmarks/connected_network_2026-09-11/reproducibility/audit_mfd_multiregion_transfer_independent.py
```

The unpacker verifies original hashes, creates local `.json` proofs, and refuses to overwrite differing records. Generated proofs are ignored by Git. The study source also supports a fresh sweep; consult `MULTIREGION_TRANSFER_PROTOCOL.md` and its command-line help.

## Optional live API experiments

Saved prompts, schemas, responses, model identifiers, interpretation outcomes, and cost ledgers are included. Offline inspection and numerical checks require no credential. `run_api.py` and `run_end_to_end_api.py` make optional paid requests when explicitly run. They read `OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` from the local environment; keep credentials outside the repository. Review the protocol, historical shared budget ledger, model availability, and current pricing before authorizing a new live run. No live calls were made during repository preparation.

## Coverage and historical paths

The complete portable suite includes development records, held-out contracts, plans, certificates, trajectory records, and retained API attempts. The earlier uncertainty study and complete connected-ring study are included separately, along with the fixture's extraction provenance.

Some archived `vendor/` scripts expect older author layouts. `prepare_anchor.py` refers to the omitted original `output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK/` tree; use the supplied `anchor.json` for portable reproduction. `package_release.py` expects historical ZIP archives. Historical manuscript audits and plot generators can reference omitted old manuscripts or result folders; use the suite's own entry points for its portable interface.

The unchanged `scripts/finalize_benchmark_handover_2026_09_26.py` validates delayed-handover records and emits a historical table fragment. Create its local output directory, `output/package/PERIMETER_BENCHMARK_REVISION_2026-09-26/`, before running it. Current submission text is maintained separately in `paper/`.

The 7 October consolidation adds the cited historical local-agent, routing, fixed-controller/repair, eight-event MPC and complete online comparison records in `archive/historical_studies/`. The preceding complete source package and supplement are preserved in `archive/companion_manuscript_2026_10_06/`. See `paper/STANDALONE_MATERIALS_MAP.md` and `archive/README.md` for scope and historical-layout requirements. Duplicate output trees, credentials, temporary files and Python caches are omitted.

## Historical manuscripts and reproduction guide - 19 October 2026

The 19 October proof revision carries the user-supplied reduced paper's
clearer proofs into the complete manuscript. The complete PDF is 87 pages
and the reduced PDF is 53 pages. The reduced source is preserved exactly
as supplied. Theorem 1 and Proposition 1 retain detailed arguments in the
complete paper's appendices. Formal statements, equation numbers, and
numerical results are preserved. Declarations remain embedded in both
main sources.

The latest [combined LaTeX package](paper/PERIMETER_CONTROL_BOTH_VERSIONS_PROOF_CLARITY_2026-10-19.zip)
contains six files: both main sources, the shared bibliography, two required
figures and a README. Select either main source in Overleaf and use pdfLaTeX.

The [revision note](paper/PROOF_SYNC_2026-10-19.txt) and
[QA record](paper/PROOF_SYNC_QA_2026-10-19.json) document the update,
compilation checks, and independent source-package validation.
The preceding 81-page complete and 47-page reduced papers are preserved
in [the archive](archive/manuscripts_2026_10_19_before_proof_sync/).
Additional reproduction material remains in
[the guide](paper/supporting_materials/REPRODUCING_THE_REDUCED_PAPER.md).
Repository visibility remains private.
