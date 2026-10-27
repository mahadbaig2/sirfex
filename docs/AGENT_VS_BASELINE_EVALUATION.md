# Agent versus non-LLM development evaluation

## Approved matched pilot

The GPT-OSS 120B selector (`20261027T204121_e995acd6e5`) and deterministic selector (`20261027T204429_5bd98f576d`) each received the same seed-3 competent PPO checkpoint, development evidence, nominal and stress partitions, continuation seed, PPO algorithm, frozen eligibility gates, and 51,200-interaction budget. The selectors chose different schedules: 5% then 10% stress for the agent, fixed 5% for the non-LLM method. Both were approved before local CPU training. All seven recorded matching-contract checks passed.

| Measure | GPT-OSS strategy | Non-LLM strategy | Agent minus non-LLM |
|---|---:|---:|---:|
| Actual additional interactions | 51,200 | 51,200 | 0 |
| Actual stress interactions | 4,751 | 2,689 | +2,062 |
| Nominal development MAE | 0.025478 | 0.021128 | +0.004351 |
| Stress development MAE | 0.078767 | 0.067433 | +0.011335 |
| Nominal violation frequency | 0 | 0 | 0 |
| Stress violation frequency | 0 | 0 | 0 |
| Normal-control nominal-MAE increase | +43.74% | +19.19% | — |
| Deterministic decision | `development_ineligible` | `development_ineligible` | — |

Both strategies had adequate stress exposure and improved stress tracking relative to the equal-budget normal-only control. Neither preserved nominal MAE within its 10% relative guardrail (limit 0.019498). The non-LLM strategy had lower nominal and stress MAE in this single-seed development pilot. The agent strategy did not demonstrate a selection advantage here. There is no independent-seed estimate or final-holdout result.

The accepted agent proposal used two Groq calls, 3,812 input tokens, 1,050 output tokens, about 3.83 seconds of recorded API latency, and an estimated $0.001202 API cost. The deterministic selector used no LLM tokens. PPO training took approximately 198 and 200 seconds, respectively. One earlier HTTP 400 schema request and one rejected model proposal preceded the accepted pilot; those attempts are disclosed in the [Phase 3.6 report](PHASE_3_6_REPORT.md). Neither model narratives nor the results critic overrode the deterministic eligibility classification.

Both candidate checkpoint hashes and transition-level exposure records verified; representative nominal and stress replays returned `exact_match`. The path-free [public pilot evidence](evidence/phase_3_6_pilot.json) includes the actual measurements and guardrail checks. Local raw checkpoints and detailed traces remain in ignored artifact directories.

## Earlier offline smoke

The 256-interaction offline mock pair verified proposal, approval, PPO execution, evaluation, critique, and persistence. Both arms measured nominal MAE 0.054527 and stress MAE 0.270660, with decision `smoke_inconclusive`. Those fixture results were never evidence of GPT-OSS quality. See [smoke evidence](evidence/phase_3_6_smoke.json).
