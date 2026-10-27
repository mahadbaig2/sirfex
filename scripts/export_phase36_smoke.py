"""Export a small, path-free public record of the completed offline software smoke."""

import json
from pathlib import Path

from sirfex.research.comparison import compare
from sirfex.research.store import ResearchStore
from sirfex.training.checkpoints import write_json

AGENT = "20261027T192619_707a8c02ae"
BASELINE = "20261027T192827_51f292034e"
FAILED_GROQ = "20261027T193144_fa2e1724a4"


def main() -> None:
    store = ResearchStore("artifacts/research")
    matched = compare(store, AGENT, BASELINE)
    results = {}
    for key, item in matched["results"].items():
        results[key] = {
            name: value
            for name, value in item.items()
            if name not in {"model_usage", "training_seconds"}
        }
        results[key]["provider_calls"] = item["model_usage"]["provider_calls"]
        results[key]["llm_calls"] = item["model_usage"]["llm_calls"]
        results[key]["training_seconds"] = item["training_seconds"]
        experiment = store.read(item["run_id"], "experiments.json")
        results[key]["starting_checkpoint_sha256"] = experiment["starting_checkpoint_sha256"]
        results[key]["candidate_checkpoint_sha256"] = experiment["checkpoint"]["sha256"]
        results[key]["exposure_reconciled"] = experiment["exposure"]["reconciliation_valid"]
    failure = store.read(FAILED_GROQ, "manifest.json")
    payload = {
        "schema_version": "sirfex-phase-3-6-public-smoke:1",
        "classification": "offline_software_smoke",
        "matched_contract_checks": matched["matched_contract_checks"],
        "agent_minus_baseline_nominal_mae": matched["agent_minus_baseline_nominal_mae"],
        "agent_minus_baseline_stress_mae": matched["agent_minus_baseline_stress_mae"],
        "results": results,
        "groq_no_key_probe": {
            "run_id": FAILED_GROQ,
            "status": failure["status"],
            "error_type": failure.get("error_type"),
            "ppo_executed": False,
        },
        "final_holdouts_accessed": False,
        "interpretation": (
            "Offline mock checks software wiring only; "
            "no GPT-OSS or robustness claim."
        ),
    }
    target = Path("docs/evidence/phase_3_6_smoke.json")
    write_json(target, payload)
    print(json.dumps({"path": str(target), "classification": payload["classification"]}))


if __name__ == "__main__":
    main()
