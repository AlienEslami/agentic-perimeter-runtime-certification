# Preserved companion and historical studies

This directory preserves material moved out of the standalone manuscript on 7 October 2026. Essential experimental specifications now appear in Appendices G–J of `paper/main.tex`; the existing proof appendices remain in that paper. The repository is private during manuscript preparation.

## Contents

- `standalone_manuscript_2026_10_07/` preserves the complete preceding source package, README, materials map and audit with preservation hashes, before the separate long/reduced revision.

- `companion_manuscript_2026_10_06/` preserves the preceding supplement source, compiled PDF, references, figures and complete manuscript source ZIP. Its S5–S7 contain the earlier local-agent, routing and fixed-controller/repair narratives and tables.
- `historical_studies/local_task_state_2026_09_11.zip` preserves local-agent requests, responses and execution feedback, the commissioning grid, matched runtime episodes, and the task-state protocols from `FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK`. It contains eight development responses and forty later responses, including the memory intervention.
- `historical_studies/LIVE_OPENAI_CERTIFIED_ROUTING_DATA.zip` preserves the original earlier OpenAI routing package, including literal calls, protocol, source and audit.
- `historical_studies/controller_repair_2026_09_22.zip` preserves the historical transport package's `evidence/` and `benchmark_sources/` folders: fixed-controller adaptation, MPC comparisons, single-contract repair, matched ensemble, search audit and final comparison.
- `historical_studies/mpc_compatibility_2026_09_10.zip` preserves the original MPC certified grid and its audit, operational certified/nominal grids, commissioned library and certified-grid audit.
- `historical_studies/online_structure_2026_09_10.zip` preserves all seventeen online cases and proofs, twenty source snapshots, the original twelve-request sequence, supporting code and independent validation. Its matching manifest records file hashes and source bindings.
- `PRESERVATION_MANIFEST_2026-10-07.json` records archive hashes and, for the newly assembled historical archives, original file entries, byte counts and source hashes.

## Provenance and rerun scope

These archives preserve existing research records. The consolidation made no new optimization runs, traffic experiments or live model calls. Newly assembled ZIPs retain the original workspace-relative paths; the routing ZIP is preserved as supplied. Historical numerical values and recorded calls are unchanged.

The current reusable suite and its offline audit are maintained separately in `benchmarks/reusable_perimeter_suite_2026-09-26/`. Its saved current API calls are in `results/api/` and `results/end_to_end_api/`. The uncertainty and connected-network studies have their own benchmark folders and protocols.

Historical scripts retain their original directory and dependency assumptions. Read the included protocol before rerunning an old study and restore its expected layout and source dependencies as needed. Raw saved calls support inspection and reconstruction of recorded decisions without making a new model request; a new live run is a separate experiment whose model identifier and results must be recorded separately.

The standalone paper contains the complete argument and essential experiment settings. This archive supplies provenance, literal prompts and additional historical evidence beyond the material included in the paper.
