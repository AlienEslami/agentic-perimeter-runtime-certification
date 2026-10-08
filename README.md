# Formal Guarantees for Trustworthy Agent-Mediated Urban Perimeter Control

This private preparation repository contains the manuscript and numerical research material for the paper. The manuscript includes the [repository link](https://github.com/AlienEslami/agentic-perimeter-runtime-certification). The repository is private during manuscript preparation and will be made public before submission.

## Contents

- `paper/`: the long manuscript (non-IEEE and IEEE reading layouts), a separate reduced version, essential appendices, figures, declarations, compiled PDFs and reduction notes.
- `archive/`: the preceding companion manuscript and losslessly preserved historical studies, with preservation hashes.
- `benchmarks/reusable_perimeter_suite_2026-09-26/`: the portable two-region suite, unchanged model/checker sources in `vendor/`, protocols, exact plans, certificates, trajectories, and recorded live-agent responses.
- `benchmarks/mfd_uncertainty/`: earlier uncertainty-frontier sources, inputs, source snapshots, and results.
- `benchmarks/connected_network_2026-09-11/`: the complete sixteen-attempt ring study, unchanged source dependencies and snapshots, and independent audit. Large proof records are losslessly compressed as `.json.gz`.
- `provenance/anchor_2026-09-11/`: compressed archived run and continuation proof used to extract the portable minute-20 fixture.
- `scripts/`: read-only repository checks, a proof unpacker, and the historical handover finalization script.
- `REPRODUCTION_MANIFEST.json`: original/stored hashes, compression metadata, and omitted cache files.
- `PUBLIC_RELEASE_CHECKLIST.md`: outstanding public-release decisions.

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

## Revision and release

Copied numerical algorithms and recorded values are preserved. The 8 October 2026 long and reduced manuscripts retain the complete formal argument and essential experimental protocols. Result headings are plain with brief preceding goal explanations. The reduced version is 70 pages, compared with 79 in the same layout for the long version, and refers to it for extended experimental reporting. All statements, proofs and distinct displayed mathematics remain; the abstract and introduction are unchanged from the approved version. Additional historical material is archived with preservation hashes. Public availability, license choice and remaining author/journal checks are tracked in `PUBLIC_RELEASE_CHECKLIST.md`.

The subsequent 8 October author-approved cleanup removes the funding section,
consolidates Ali Eslami's writing roles, adds Jiangbo Yu's review/editing role,
and removes editorial manuscript-preparation notes. Repository visibility
remains private; the manuscript retains its neutral repository link.

The latest 8 October plain-language revision clarifies certificate
guarantees, repair information, terminal continuation and experimental
scope in both versions. Displayed mathematics and numerical data are
unchanged. The latest checks and source hashes are in
`paper/PLAIN_LANGUAGE_QA_2026-10-08.json`. The exact preceding source package
is archived in `archive/long_reduced_2026_10_08_before_language_revision/`.
