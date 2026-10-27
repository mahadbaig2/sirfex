"""Export the reviewed Groq proposal without credentials or local filesystem paths."""

import json
from pathlib import Path

from sirfex.research.store import ResearchStore
from sirfex.training.checkpoints import write_json

RUN_ID = "20261027T204121_e995acd6e5"
REJECTED_RUN_ID = "20261027T204002_b08085ec00"
BASELINE_RUN_ID = "20261027T204429_5bd98f576d"


def main() -> None:
    store = ResearchStore("artifacts/research")
    manifest = store.read(RUN_ID, "manifest.json")
    if manifest["status"] != "awaiting_approval":
        raise ValueError("Expected a pending proposal")
    usage = store.read(RUN_ID, "model_usage.json")
    plan = store.read(RUN_ID, "plan.json")
    baseline_plan = store.read(BASELINE_RUN_ID, "plan.json")
    if (
        plan["starting_checkpoint_sha256"] != baseline_plan["starting_checkpoint_sha256"]
        or plan["development_hashes"] != baseline_plan["development_hashes"]
        or plan["continuation_seed"] != baseline_plan["continuation_seed"]
        or plan["training_timesteps"] != baseline_plan["training_timesteps"]
    ):
        raise ValueError("Pilot plans do not match on the required contract")
    payload = {
        "schema_version": "sirfex-groq-proposal-public:1",
        "run_id": RUN_ID,
        "status": manifest["status"],
        "provider_model": manifest["provider_model"],
        "proposal": store.read(RUN_ID, "proposals.json"),
        "validation": store.read(RUN_ID, "validation.json"),
        "plan": plan,
        "matched_non_llm_plan": {
            "run_id": BASELINE_RUN_ID,
            "status": store.read(BASELINE_RUN_ID, "manifest.json")["status"],
            "proposal": store.read(BASELINE_RUN_ID, "proposals.json"),
            "validation": store.read(BASELINE_RUN_ID, "validation.json"),
            "plan": baseline_plan,
            "review_note": (
                "The selector's original hypothesis says warmup, but its actual "
                "schedule is fixed 5% stress throughout. The code wording is corrected "
                "for future proposals; this frozen artifact is unchanged."
            ),
        },
        "model_usage": {
            "provider_calls": usage["provider_calls"],
            "llm_calls": usage["llm_calls"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "estimated_usd": usage["estimated_usd"],
            "request_latencies_seconds": [r["latency_seconds"] for r in usage["requests"]],
        },
        "earlier_rejected_proposal": {
            "run_id": REJECTED_RUN_ID,
            "status": store.read(REJECTED_RUN_ID, "manifest.json")["status"],
            "reason": "unsupported_evidence_claim",
            "training_executed": False,
        },
        "attempt_accounting_limit": (
            "The accepted run used two model calls. The first rejected call was not "
            "preserved in the original attempt-history format; later runs retain it."
        ),
        "training_executed": False,
        "final_holdouts_accessed": False,
        "review_note": (
            "The model's '0.1 MAE guardrail' wording is inaccurate. The frozen rule "
            "allows at most 10% relative nominal MAE increase over the equal-budget "
            "normal control: seed-3 limit 0.019498146628656905. The model cannot alter it."
        ),
    }
    path = Path("docs/evidence/phase_3_6_groq_proposal.json")
    write_json(path, payload)
    print(json.dumps({"path": str(path), "status": payload["status"]}))


if __name__ == "__main__":
    main()
