# Phase 3.6 report: agent-guided robustness research

## Outcome

Phase 3.6A numerical diagnostics are complete and published in [the diagnostic report](POLICY_REGRESSION_DIAGNOSTICS.md). Phase 3.6B implements a bounded optional research workflow, passed an offline software smoke, and completed an approved, matched development pilot. **Neither the GPT-OSS strategy nor the deterministic non-LLM strategy passed the nominal-preservation guardrail.** The non-LLM strategy had lower nominal and stress mean tracking errors in this one-seed pilot. This is a development result, not a held-out robustness claim or a general ranking of selection methods.

## Measured diagnostic evidence

Across fixed development observations for seed 3, Phase 3.5 normal-only continuation had mean physical pump command 0.7362, compared with 0.8069 under 5% stress and 0.8676 under 10% stress. Upper-action saturation was 59.4%, 71.6%, and 75.9%, respectively. The stress policies also had higher signed action shifts than normal-only continuation. Seed 17's shifts were smaller. The observed policy shifts and tracking changes are associations on development states; they do not identify a causal root mechanism. Independent matched rollouts and source checkpoint hashes are preserved in the diagnostic artifacts. Intermediate historical checkpoints were unavailable, so no learning-trajectory claim is made.

## Implemented workflow

Strict Pydantic proposals, semantic evidence validation, deduplication, bounded LangGraph proposal and execution graphs, human approval, CPU PPO continuation, transition-level exposure accounting, local research memory, checkpoint verification, development replay, deterministic guardrails, and matched comparison are available through `sirfex research`. Groq is optional; the core and diagnostics remain offline. The initial registry has three supported intervention families. See [architecture](AGENTIC_RESEARCH_ARCHITECTURE.md), [proposal contract](AGENT_IMPROVEMENT_PROPOSALS.md), [protocol](AGENT_EXPERIMENT_PROTOCOL.md), and [security limits](AGENT_SECURITY_AND_LIMITS.md).

## Software smoke and limitation

The offline mock and non-LLM selector each completed a 256-interaction continuation from the same seed-3 checkpoint. Each recorded 15 stress interactions, nominal development MAE 0.054527, and stress development MAE 0.270660. Both decisions were `smoke_inconclusive`, and their numerical differences were zero for this tiny run. The [comparison report](AGENT_VS_BASELINE_EVALUATION.md) explains why these are pipeline checks only. The earlier Groq-path no-key probe ended `provider_failed` before proposal validation or training. Final holdouts were not accessed.

## Approved 51,200-interaction development pilot

Both proposals were approved and executed sequentially from the same seed-3 PPO checkpoint, optimizer state, continuation seed, development evidence, and nominal/stress evaluation sets. Each completed exactly 51,200 additional interactions. The agent selected a 5% then 10% stress schedule; the deterministic selector used 5% throughout. The frozen Phase 3.5 normal-only control had nominal MAE 0.017725588 and stress MAE 0.116625410. The nominal gate permits at most 0.019498147 MAE for seed 3.

| Selector | Actual stress interactions | Completed stress episodes | Nominal MAE | Stress MAE | Nominal change vs normal control | Development decision |
|---|---:|---:|---:|---:|---:|---|
| GPT-OSS 120B | 4,751 | 67 | 0.025478 | 0.078767 | +43.74% | `development_ineligible` |
| Deterministic non-LLM | 2,689 | 53 | 0.021128 | 0.067433 | +19.19% | `development_ineligible` |

Both arms met exposure adequacy, covered all seven discoveries, had zero development nominal and stress violations, passed execution checks, and improved stress MAE over the normal-only control. Both failed the nominal-MAE guardrail. The agent arm's nominal MAE exceeded the non-LLM arm by 0.004351 and its stress MAE by 0.011335. These are descriptive differences for one development checkpoint; no independent seed replication or final-holdout evaluation was performed. Both checkpoints passed SHA-256 verification and both representative nominal/stress replays exactly matched. The full compact [pilot evidence](evidence/phase_3_6_pilot.json) records exposures, decisions, hashes, replay status, and model usage.

The accepted agent run used two Groq calls, 3,812 input tokens and 1,050 output tokens, with an estimated API cost of $0.001202. Local PPO training took about 198 seconds for the agent arm and 200 seconds for the comparator; this excludes proposal and evaluation time. Before freezing the accepted proposal, one schema-rejected API request and one additional proposal response were also attempted. The latter cost estimate was $0.000506. Those preparation calls are disclosed rather than charged to the matched training comparison.

## Scientific status and next gate

The numerical evidence suggests testable hypotheses about stress-induced command shifts but does not prove a cause or robustness gain. An earlier model proposal was rejected because its citations claimed curriculum results absent from the cited discoveries; exact-claim validation now prevents that error. The accepted run made two model calls; the first rejected call's details were not preserved by the earlier attempt-history implementation. Future calls persist each attempt. Neither pilot candidate is eligible for confirmation under the frozen guardrail, so final holdouts remain sealed. Automatic PPO resume and a live model-based results critic are not yet validated; existing deterministic records remain authoritative.

The accepted proposal's prose refers to a “0.1 MAE guardrail,” which is inaccurate. The frozen seed-3 gate is a 10% *relative* increase over the equal-budget normal-only MAE of 0.01772558784423355, yielding a limit of 0.019498146628656905. The deterministic gate uses that rule regardless of the model's wording. The schedule ends at 10% stress exposure, a level that previously breached seed 3's nominal guardrail under a different fixed curriculum; the new schedule remains a hypothesis, not a predicted success.

A matched non-LLM plan was frozen before execution: run `20261027T204429_5bd98f576d`, fixed 5% stress throughout 51,200 interactions, from the same checkpoint and development partitions. Its generated hypothesis inaccurately says “warmup”; the stored schedule was authoritative and the selector wording has been corrected for future proposals.
