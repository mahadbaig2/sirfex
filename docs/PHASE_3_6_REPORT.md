# Phase 3.6 report: agent-guided robustness research

## Outcome

Phase 3.6A numerical diagnostics are complete and published in [the diagnostic report](POLICY_REGRESSION_DIAGNOSTICS.md). Phase 3.6B implements a bounded optional research workflow and has passed a matched offline software smoke. **The agent-versus-non-LLM scientific question remains unanswered.** No live Groq request or meaningful 51,200-step agent pilot was performed; no API key was configured and a concrete pilot proposal has not been approved for training.

## Measured diagnostic evidence

Across fixed development observations for seed 3, Phase 3.5 normal-only continuation had mean physical pump command 0.7362, compared with 0.8069 under 5% stress and 0.8676 under 10% stress. Upper-action saturation was 59.4%, 71.6%, and 75.9%, respectively. The stress policies also had higher signed action shifts than normal-only continuation. Seed 17's shifts were smaller. The observed policy shifts and tracking changes are associations on development states; they do not identify a causal root mechanism. Independent matched rollouts and source checkpoint hashes are preserved in the diagnostic artifacts. Intermediate historical checkpoints were unavailable, so no learning-trajectory claim is made.

## Implemented workflow

Strict Pydantic proposals, semantic evidence validation, deduplication, bounded LangGraph proposal and execution graphs, human approval, CPU PPO continuation, transition-level exposure accounting, local research memory, checkpoint verification, development replay, deterministic guardrails, and matched comparison are available through `sirfex research`. Groq is optional; the core and diagnostics remain offline. The initial registry has three supported intervention families. See [architecture](AGENTIC_RESEARCH_ARCHITECTURE.md), [proposal contract](AGENT_IMPROVEMENT_PROPOSALS.md), [protocol](AGENT_EXPERIMENT_PROTOCOL.md), and [security limits](AGENT_SECURITY_AND_LIMITS.md).

## Software smoke and limitation

The offline mock and non-LLM selector each completed a 256-interaction continuation from the same seed-3 checkpoint. Each recorded 15 stress interactions, nominal development MAE 0.054527, and stress development MAE 0.270660. Both decisions were `smoke_inconclusive`, and their numerical differences were zero for this tiny run. The [comparison report](AGENT_VS_BASELINE_EVALUATION.md) explains why these are pipeline checks only. The Groq-path no-key probe ended `provider_failed` before proposal validation or training. Final holdouts were not accessed.

## Scientific status and next gate

The numerical evidence suggests testable hypotheses about stress-induced command shifts but does not prove a cause or robustness gain. Meaningful comparison needs a live, validated GPT-OSS proposal, explicit approval for its specified 51,200-step continuation, a matched non-LLM arm, and development-only evaluation. Any confirmatory final evaluation would need a separately frozen protocol. Automatic PPO resume and a live model-based results critic are not yet validated; existing deterministic records remain authoritative.
