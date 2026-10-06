# Reusable urban perimeter-control benchmark

This study preserves the 25 September manuscript and supplies three separately defined benchmark tasks: native regional regulation, queue-inclusive economic plan replacement, and deadline-constrained replanning. Read `PROTOCOL.md`, `FULL_SCENARIO_PROTOCOL.md` and `API_PROTOCOL.md` before using the results.

## Contents

- `benchmark_formulation.tex`: manuscript-ready mathematical problem definitions, supplied as an input fragment.
- `anchor.json`: portable minute-20 information state and verified continuation, with archive hashes.
- `vendor/`: unchanged copies of the previous physical model and checker sources. `common.py` adapts archived source-hash file lookups to the portable folder; numerical certificate routines are unchanged.
- `results/regulation/`: 24 matched four-state CLF/tracking-MPC cases.
- `results/contracts/`: 12 queue-only plus 72 development and 8 held-out deadline contracts, with exact commands, solver attempts, certificates and all 640 constant-plan realizations.
- `results/full_scenario/`: the stronger all-corner/profile predictive comparator and its checked repairs, on every deadline contract.
- `results/api/`: 96 OpenAI contextual interpretation/tool-routing calls over precomputed benchmark plans.
- `results/end_to_end_api/`: eight live agent traces whose objective requests actually parameterize and execute the planning, checking and repair tools.
- `results/reserve_refresh/`: independently checked fresh reserve plans where the archived continuation proof did not close under a widened contract.
- `results/initial_quadrature/`: preserved initial integration-development results; excluded from every final summary.
- `results/initial_expiry_alignment/`: preliminary live-tool records using 35-minute priorities with a constant-priority plan through minute 60. The final eight traces align priority expiry with the plan's minute-60 expiry; all preliminary API records and charges are retained.
- `results/certified_tool_selection.json`: minimum nominal center-forecast task cost among same-contract certified tool candidates, with no use of realized outcomes in selection.
- `results/*audit*.json`: physical-model, derivative and adaptive-replay checks.
- `DELAYED_PROTOCOL.md` and `run_delayed_handover.py`: future-dated replacement with incumbent execution during preparation.
- `results/delayed_handover/`: all 51 delayed preparation cases, propagated information boxes, exact schedules, timing, certificates, activation membership and 640 traffic realizations per case.
- `results/delayed_handover_summary.json` and `results/DELAYED_HANDOVER_CASES.csv`: complete delay results without outcome selection.

## Reproduction

Use Python 3.12 and install `requirements.txt`. From this folder:

```text
python audit.py
python run_regulation.py
python run_contracts.py --workers 2
python run_full_scenario.py --audit
python run_full_scenario.py --workers 2
python run_reserve_refresh.py --workers 2
python audit.py --replay
python select_certified_tools.py
python make_report.py
python run_delayed_handover.py --workers 2
```

Runs are checkpointed. Existing result files are reused. For a fresh run, copy the source and fixture into a new folder and keep only the supplied `anchor.json`, without copying `results/`. The fixture was extracted from the existing manuscript study; `prepare_anchor.py` is for the authors' archive layout and is unnecessary for portable reproduction.

The delayed extension requires the synchronous results and certified fresh reserves. It keeps the original minute-60 deadline and propagates the full original information set without a new observation. Handover occurs only if preparation finishes before its planned slot. Actual timing depends on processor and concurrent load. The complete research release includes the authors' finalization script and the revised manuscript; the earlier `make_report.py` report describes the synchronous sweep.

API calls are optional. Set `OPENAI_API_KEY_FILE` to a local credential file (or `OPENAI_API_KEY`) and use `run_api.py`, followed by `run_end_to_end_api.py`. Credentials are never packaged or placed in prompts. Both share a USD 10 budget ledger. Repeating the calls tests a versioned API model; the released raw responses permit exact replay of this study's decisions. The model and price identifiers are recorded per call and in the ledger.

## Fair interpretation

The native CLF equations are reproduced locally. On the seven-state model, the cross-gate CLF is extended with queue/exit LQR feedback and a common directional gate constraint; label it as an extension. Economic MPC and scenario MPC are local implementations of predictive-control baselines, with their blocking, solver and information choices disclosed. They are distinct from the original authors' software and published numerical tables.

Every constant plan uses the same 128 initial corners and five residual profiles. CLF feedback uses 16 fixed corners and the same five profiles. The physical CBF-QP is a single matched adverse trajectory per deadline contract. Do not pool these denominators. Proof acceptance comes from the interval checker, while trajectory results are sampled performance tests. Costs of infeasible trajectories do not establish superior constrained performance.

The all-corner comparator was added after early results exposed the limited coverage of five scenarios. It uses fixed one-vehicle prediction clearance, with no parameter retuning between development and held-out settings. Preserve both result sets when reporting it. The model parameters and internal demand are exact within this synthetic experiment; the study varies sensing and arrival residuals.

## Literature anchors

Zhong et al. (2018), *Robust perimeter control for two urban regions with macroscopic fundamental diagrams: A control-Lyapunov function approach*, Transportation Research Part B 117, 687-707, DOI: 10.1016/j.trb.2017.09.008.

Geroliminis, Haddad and Ramezani (2013), *Optimal Perimeter Control for Two Urban Regions With Macroscopic Fundamental Diagrams: A Model Predictive Approach*, IEEE Transactions on Intelligent Transportation Systems 14(1), 348-359, DOI: 10.1109/TITS.2012.2216877.

Haddad (2017), *Optimal perimeter control synthesis for two urban regions with aggregate boundary queue dynamics*, Transportation Research Part B 96, 1-25, DOI: 10.1016/j.trb.2016.10.016. This motivates exterior waiting as a traffic cost; the benchmark supplies its own fully specified seven-state queue model.
