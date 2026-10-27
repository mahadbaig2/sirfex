"""Matched development comparison without granting the model decision authority."""

from __future__ import annotations

from sirfex.research.store import ResearchStore
from sirfex.training.protocol import content_hash


def compare(store: ResearchStore, agent_id: str, baseline_id: str) -> dict:
    if agent_id == baseline_id:
        raise ValueError("Comparison requires distinct research runs")
    items = {}
    for key, run_id in (("agent", agent_id), ("non_llm", baseline_id)):
        manifest = store.read(run_id, "manifest.json")
        protocol = store.read(run_id, "protocol.json")
        if (
            manifest.get("kind") != "agent-improvement-research"
            or content_hash(protocol) != manifest.get("protocol_sha256")
            or manifest.get("status") != "completed"
        ):
            raise ValueError(f"{key} research run is incomplete or changed")
        items[key] = {
            "run_id": run_id,
            "manifest": manifest,
            "protocol": protocol,
            "summary": store.read(run_id, "summary.json"),
            "experiment": store.read(run_id, "experiments.json"),
            "evaluation": store.read(run_id, "development_evaluation.json"),
            "decision": store.read(run_id, "decision.json"),
            "usage": store.read(run_id, "model_usage.json"),
        }
    agent, baseline = items["agent"], items["non_llm"]
    if (
        agent["protocol"]["config"]["method"] not in ("groq_agent", "mock_agent")
        or baseline["protocol"]["config"]["method"] != "non_llm"
    ):
        raise ValueError("Expected agent selector and non-LLM baseline")
    same = {
        "starting_checkpoint": agent["protocol"]["source_checkpoint_sha256"]
        == baseline["protocol"]["source_checkpoint_sha256"],
        "development_sets": agent["protocol"]["development_hashes"]
        == baseline["protocol"]["development_hashes"],
        "starting_seed": agent["protocol"]["config"]["starting_seed"]
        == baseline["protocol"]["config"]["starting_seed"],
        "requested_steps": agent["protocol"]["config"]["training_timesteps"]
        == baseline["protocol"]["config"]["training_timesteps"],
        "actual_steps": agent["experiment"]["actual_additional_steps"]
        == baseline["experiment"]["actual_additional_steps"],
        "evidence_context": agent["protocol"]["evidence_context_sha256"]
        == baseline["protocol"]["evidence_context_sha256"],
        "continuation_seed": agent["experiment"]["checkpoint"]["training_seed"]
        == baseline["experiment"]["checkpoint"]["training_seed"],
    }
    if not all(same.values()):
        raise ValueError(f"Unmatched strategy comparison: {same}")
    results = {}
    for key, item in items.items():
        results[key] = {
            "run_id": item["run_id"],
            "method": item["protocol"]["config"]["method"],
            "provider_model": item["manifest"]["provider_model"],
            "actual_training_interactions": item["experiment"]["actual_additional_steps"],
            "actual_stress_interactions": item["experiment"]["exposure"]["stress_interactions"],
            "stress_completed_episodes": item["experiment"]["exposure"][
                "stress_completed_episodes"
            ],
            "decision": item["decision"]["status"],
            "nominal_mae": item["evaluation"]["development_nominal"]["mean_tracking_error"],
            "stress_mae": item["evaluation"]["development_stress"]["mean_tracking_error"],
            "nominal_violation_frequency": item["evaluation"]["development_nominal"][
                "constraint_violation_frequency"
            ],
            "stress_violation_frequency": item["evaluation"]["development_stress"][
                "constraint_violation_frequency"
            ],
            "model_usage": item["usage"],
            "training_seconds": item["experiment"]["training_seconds"],
        }
    real = agent["protocol"]["config"]["method"] == "groq_agent"
    status = (
        "development_comparison"
        if real and results["agent"]["decision"] != "smoke_inconclusive"
        else "offline_software_smoke"
    )
    return {
        "schema_version": "sirfex-agent-vs-baseline:1",
        "status": status,
        "matched_contract_checks": same,
        "results": results,
        "agent_minus_baseline_nominal_mae": results["agent"]["nominal_mae"]
        - results["non_llm"]["nominal_mae"],
        "agent_minus_baseline_stress_mae": results["agent"]["stress_mae"]
        - results["non_llm"]["stress_mae"],
        "interpretation": (
            "Offline mock comparison verifies matching and persistence only; "
            "it is not evidence of GPT-OSS strategy superiority."
            if not real
            else "Development comparison only; no held-out superiority claim."
        ),
    }
