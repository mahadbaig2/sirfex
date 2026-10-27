"""Export the matched development pilot after both approved runs finish."""

import json
from pathlib import Path

from sirfex.research.comparison import compare
from sirfex.research.store import ResearchStore
from sirfex.research.verification import verify_run
from sirfex.training.checkpoints import write_json

AGENT = "20261027T204121_e995acd6e5"
BASELINE = "20261027T204429_5bd98f576d"
PREP_RUNS = ["20261027T203910_52c00f2e4b", "20261027T204002_b08085ec00"]


def main() -> None:
    store = ResearchStore("artifacts/research")
    comparison = compare(store, AGENT, BASELINE)
    for run_id in (AGENT, BASELINE):
        verify_run(run_id, store)
    public = {
        "schema_version": "sirfex-phase-3-6-pilot-public:1",
        "classification": comparison["status"],
        "agent_run_id": AGENT,
        "non_llm_run_id": BASELINE,
        "matched_contract_checks": comparison["matched_contract_checks"],
        "agent_minus_baseline_nominal_mae": comparison["agent_minus_baseline_nominal_mae"],
        "agent_minus_baseline_stress_mae": comparison["agent_minus_baseline_stress_mae"],
        "results": {},
        "preparation_requests": [
            {
                "run_id": run_id,
                "status": store.read(run_id, "manifest.json")["status"],
                "provider_calls": store.read(run_id, "model_usage.json")["provider_calls"],
                "input_tokens": store.read(run_id, "model_usage.json")["input_tokens"],
                "output_tokens": store.read(run_id, "model_usage.json")["output_tokens"],
                "estimated_usd": store.read(run_id, "model_usage.json")["estimated_usd"],
            }
            for run_id in PREP_RUNS
        ],
        "final_holdouts_accessed": False,
        "interpretation_limit": (
            "Single-seed development pilot only. Eligibility is deterministic. "
            "No independent replication or final-holdout generalization claim."
        ),
    }
    for key, item in comparison["results"].items():
        run_id = item["run_id"]
        experiment = store.read(run_id, "experiments.json")
        public["results"][key] = {
            **{name: value for name, value in item.items() if name != "model_usage"},
            "starting_checkpoint_sha256": experiment["starting_checkpoint_sha256"],
            "candidate_checkpoint_sha256": experiment["checkpoint"]["sha256"],
            "exposure": experiment["exposure"],
            "guardrails": store.read(run_id, "decision.json"),
            "replay_verification": store.read(run_id, "replay_verification.json"),
            "model_usage": {
                name: value for name, value in item["model_usage"].items() if name != "requests"
            },
            "model_requests": [
                {
                    "provider": request["provider"],
                    "model": request["model"],
                    "latency_seconds": request["latency_seconds"],
                    "input_tokens": request["input_tokens"],
                    "output_tokens": request["output_tokens"],
                    "estimated_usd": request["estimated_usd"],
                }
                for request in item["model_usage"]["requests"]
            ],
        }
    target = Path("docs/evidence/phase_3_6_pilot.json")
    write_json(target, public)
    print(json.dumps({"path": str(target), "classification": public["classification"]}))


if __name__ == "__main__":
    main()
