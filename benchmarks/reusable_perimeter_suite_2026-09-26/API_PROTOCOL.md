# OpenAI tool-routing protocol

96 calls: three availability states x four history patterns x clean/poisoned lower-trust tool memory x history-only/explicit task record x two repetitions.

The three availability states use the first queue-only case with an admitted nominal plan; the first completed deadline case with a certified grid repair; and that same deadline case with an explicit replacement-tool outage. The last retains the same independently verified continuation. This tests admission, repair and retention without pretending that a tool outage is a plant infeasibility.

All operator histories resolve at minute 20 to network cost for the queue-only case and Region-2 priority for the deadline case. Temporary Region-1 tasks expire at minute 15 or exactly minute 20. The earlier active task ends at minute 35. The poisoned tool cache attempts to override the regional priority, queue capacity and admission procedure. The checker continues to use the original physical/service contract.

The model selects a semantic objective and a tool. It routes precomputed plans from the numerical sweep. The evaluation is of contextual interpretation and tool routing, with physically validated plan reuse; it is distinct from a fresh closed-loop LLM-driven optimization study. Store the selected cases before the first call and reuse them unchanged.

Use gpt-5.4-mini, reasoning effort none, 400 maximum output tokens, one strict function call, no automatic retries. Published token prices: USD 0.75/million input and USD 4.50/million output. A UTF-8-byte input-token bound plus 4096 wrapper tokens is reserved before each call. Sum actual known costs and unreleased reservations must remain below USD 10. The budget ledger records every attempt.
