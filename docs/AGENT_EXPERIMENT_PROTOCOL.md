# Phase 3.6 development experiment protocol

The frozen development context uses the Phase 3.2 competent checkpoint, Phase 3.4's seven verified avoidable discoveries, Phase 3.5's nominal and screening-stress sets, and Phase 3.6A numerical diagnostics. Phase 3.4 and 3.5 final partitions stay sealed. The proposal records checkpoint and dataset hashes before approval. An explicit `--approve-training` flag is required to execute it.

The initial software smoke profile is capped at 256–512 PPO interactions and is never classified as an improvement. The pilot profile permits up to 51,200 interactions, at most two proposal calls, 8,000 total tokens, and a configured wall-clock cap. One starting seed is selected before proposals. The deterministic selector and agent must receive the same evidence context, start checkpoint, training seed, PPO budget, development sets, and guardrails. A later matched study must report every rejected proposal, model call, and actual PPO attempt. A real Groq proposal is pending approval; its PPO pilot remains unrun.

For 51,200-step candidates, the existing Phase 3.5 equal-budget normal-only arm is the control. Exposure adequacy, 10% nominal-MAE guardrail, nominal and stress violation frequencies, execution failures, and a stress-gain threshold determine development eligibility. These checks run from SIRFEX measurements. A development pass would still require a separately frozen confirmatory protocol; it is not a held-out robustness result.

Example commands from the repository root after `pip install -e ".[agents]"`:

```powershell
sirfex research diagnose --config configs/research/pump_diagnostics.json
sirfex research propose --config configs/research/agent_smoke.json
sirfex research validate <research-run-id>
sirfex research execute <research-run-id> --approve-training
sirfex research verify <research-run-id>
sirfex research compare <agent-run-id> <non-llm-run-id>
```

Inspect `sirfex research show` and the local `plan.json` before approving. `configs/research/agent_pilot.json` selects Groq; `configs/research/non_llm_pilot.json` selects its fixed 5% comparator. The [frozen public plans](evidence/phase_3_6_groq_proposal.json) record both exact interventions awaiting separate approval for meaningful PPO continuation.
