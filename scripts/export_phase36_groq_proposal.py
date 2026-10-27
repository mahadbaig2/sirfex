"""Export the reviewed Groq proposal without credentials or local filesystem paths."""

import json
from pathlib import Path

from sirfex.research.store import ResearchStore
from sirfex.training.checkpoints import write_json

RUN_ID = "20261027T204121_e995acd6e5"
REJECTED_RUN_ID = "20261027T204002_b08085ec00"


def main() -> None:
    store = ResearchStore("artifacts/research")
    manifest = store.read(RUN_ID, "manifest.json")
    if manifest["status"] != "awaiting_approval":
        raise ValueError("Expected a pending proposal")
    usage = store.read(RUN_ID, "model_usage.json")
    plan = store.read(RUN_ID, "plan.json")
    payload = {
        "schema_version": "sirfex-groq-proposal-public:1",
        "run_id": RUN_ID,
        "status": manifest["status"],
        "provider_model": manifest["provider_model"],
        "proposal": store.read(RUN_ID, "proposals.json"),
        "validation": store.read(RUN_ID, "validation.json"),
        "plan": plan,
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
