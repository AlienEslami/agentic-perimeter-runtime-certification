# Standalone manuscript materials map

Revision date: 7 October 2026.

The integrated manuscript includes the mathematical argument, existing proof appendices A–F, and the experiment specifications needed to assess its main results. Appendices G–J replace the separate supplement as the reader's source for essential experimental documentation. The abstract and introduction are preserved exactly.

## Material included in the paper

| Appendix | Included material | Former supplement location |
| --- | --- | --- |
| G — Experimental Protocols and Validation | Initial conditions, controller and optimizer settings, uncertainty contracts, repair settings, matched traffic realizations, independent validation, future-dated handover, and successive planning requests. Also includes the eight-event compatibility protocol because the main paper reports that comparison. | Essential portions of S1. |
| H — Certification Comparisons and Service Margins | Coordinate comparison, uncertainty-study design, service-margin settings, successive-plan study, complete supporting comparison tables, MPC compatibility, and the scope of numerical verification. | Essential portions of S2. |
| I — Connected-Network Experiment | Allocation and admission coupling, ring-network model, initial state, fixed plan, uncertainty and service settings, complete sixteen-attempt certificate sweep, and computational interpretation. | Essential portions of S3. |
| J — Agent Experiment Protocol | Current model and request settings; 96-call interpretation pilot; eight actual planning traces with thirteen calls; schemas, histories, objective scoring, physical checks, repair and matched replay rules. Main results remain in the existing execution-evidence table. | Essential portions of S4. |

Body references to the separate supplement are replaced with internal appendix references or references to the repository's historical-study archive. The unchanged introduction's reference to supplementary specifications is clarified outside the introduction by identifying the embedded supplementary appendices.

## Material preserved in GitHub

Paths below are relative to the repository root. The repository remains private during manuscript preparation.

| Material | Preserved location |
| --- | --- |
| Complete supplement before integration, including all earlier protocols and tables | `archive/companion_manuscript_2026_10_06/supplement.tex` and `supplement.pdf`, with references, figures and the preceding complete source ZIP. |
| Earlier local Qwen3 task-state development, held-out and memory-intervention calls; matched certified and nominal traffic episodes; commissioning grid and protocols | `archive/historical_studies/local_task_state_2026_09_11.zip`. Includes eight development responses and forty later responses, retaining their original source-relative paths. |
| Earlier OpenAI routing study, literal requests/responses, protocol, source and audit | `archive/historical_studies/LIVE_OPENAI_CERTIFIED_ROUTING_DATA.zip`. |
| Earlier fixed-controller comparisons, common-plant adaptation, single-contract repair, repair ensemble, exhaustive-grid audit and associated sources | `archive/historical_studies/controller_repair_2026_09_22.zip`. Preserves the historical transport package's `evidence/` and `benchmark_sources/` folders. |
| Original eight-event MPC/certified-MPC and commissioned-library records and audits | `archive/historical_studies/mpc_compatibility_2026_09_10.zip`. Includes the MPC certified grid, its audit, operational certified/nominal grids, commissioned library and certified-grid audit. |
| Complete online certification comparison, original twelve-request sequence, proof records, source snapshots and independent checks | `archive/historical_studies/online_structure_2026_09_10.zip`, with its separate preservation manifest. |
| Current numerical algorithms, exact plans, certificates, trajectories, per-realization outputs and current agent raw calls | `benchmarks/reusable_perimeter_suite_2026-09-26/`. |
| Earlier uncertainty-frontier sources, inputs, saved results and source snapshots | `benchmarks/mfd_uncertainty/`. |
| Complete connected-ring records, source dependencies, snapshots and independent audit | `benchmarks/connected_network_2026-09-11/`. |
| Archive identities, source entries and preservation hashes | `archive/PRESERVATION_MANIFEST_2026-10-07.json`. |

The omitted S5–S7 narratives and extra tables remain accessible in the archived supplement. Their historical studies retain their own models, uncertainty contracts, decision clocks and sample sizes. They are background evidence rather than an additional document needed to follow the integrated paper.

This revision consolidates documentation and preserves existing evidence. It introduces no new experiments, optimization runs or live model calls. The archives preserve historical paths; their presence does not imply that every old script runs directly from the current repository root.
