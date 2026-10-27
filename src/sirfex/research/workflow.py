"""Bounded LangGraph proposal flow; scientific decisions remain deterministic."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TypedDict
from uuid import uuid4

from sirfex import __version__
from sirfex.research.context import evidence_context
from sirfex.research.proposals import Proposal, proposal_hash, validate_proposal
from sirfex.research.providers import (
    DeterministicBaselineProvider,
    GroqProvider,
    MockProvider,
)
from sirfex.research.store import ResearchStore
from sirfex.training.protocol import content_hash
from sirfex.training.runner import _git_commit

PROMPT_VERSION = "sirfex-research-prompt:1"
ALLOWED_METHODS = {"groq_agent", "mock_agent", "non_llm"}


class ProposalState(TypedDict, total=False):
    context: dict
    analysis: dict
    proposal_data: dict
    validation: dict
    model_usage: list[dict]
    provider_calls: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    status: str
    error_type: str
    proposal_hash: str
    plan: dict


def validate_config(value: dict) -> dict:
    allowed = {
        "profile",
        "method",
        "diagnostic_run_id",
        "starting_seed",
        "training_timesteps",
        "max_provider_calls",
        "max_total_tokens",
        "max_completion_tokens",
        "wall_seconds",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown research configuration")
    config = dict(value)
    if (
        config.get("profile") not in ("smoke", "pilot")
        or config.get("method") not in ALLOWED_METHODS
    ):
        raise ValueError("Research profile/method must be smoke|pilot and an allowed selector")
    steps = config.get("training_timesteps")
    if (
        type(steps) is not int
        or steps % 128
        or not 256 <= steps <= 51200
        or (config["profile"] == "smoke" and steps > 512)
    ):
        raise ValueError("Invalid research training interaction cap")
    if config.get("starting_seed") not in (3, 17):
        raise ValueError("Research seed must be a frozen development seed")
    if type(config.get("diagnostic_run_id")) is not str:
        raise ValueError("A verified diagnostic run ID is required")
    for name, lower, upper in (
        ("max_provider_calls", 1, 2),
        ("max_total_tokens", 512, 8000),
        ("max_completion_tokens", 128, 1536),
        ("wall_seconds", 30, 3600),
    ):
        item = config.get(name)
        if type(item) is not int or not lower <= item <= upper:
            raise ValueError(f"Invalid {name}")
    if config["max_completion_tokens"] > config["max_total_tokens"]:
        raise ValueError("Completion token allowance exceeds total")
    return config


def _provider(method):
    return (
        GroqProvider()
        if method == "groq_agent"
        else MockProvider()
        if method == "mock_agent"
        else DeterministicBaselineProvider()
    )


def _previous_hashes(store: ResearchStore) -> set[str]:
    hashes = set()
    for manifest in store.list():
        if manifest.get("kind") != "agent-improvement-research":
            continue
        try:
            result = store.read(manifest["run_id"], "validation.json")
        except (OSError, ValueError):
            continue
        if result.get("status") == "valid" and result.get("configuration_hash"):
            hashes.add(result["configuration_hash"])
    return hashes


def _graph(provider, config, local, previous):
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError("Install the agents extra for the LangGraph research workflow") from exc

    def analyst(state: ProposalState) -> dict:
        context = state["context"]
        return {
            "analysis": {
                "measured_action_shift": context["normal_to_stress_action_shift"],
                "measured_development_metrics": context["development_metrics"],
                "uncertainty": "Fixed-observation differences do not establish causal mechanism",
            }
        }

    def strategist(state: ProposalState) -> dict:
        calls = state.get("provider_calls", 0) + 1
        if calls > config["max_provider_calls"]:
            return {"status": "provider_budget_exhausted"}
        remaining = (
            config["max_total_tokens"]
            - state.get("input_tokens", 0)
            - state.get("output_tokens", 0)
        )
        if remaining < 128:
            return {"status": "token_budget_exhausted"}
        try:
            response = provider.propose(
                state["context"],
                max_completion_tokens=min(config["max_completion_tokens"], remaining),
            )
        except Exception as exc:
            return {
                "status": "provider_failed",
                "error_type": type(exc).__name__,
                "provider_calls": calls,
                "llm_calls": state.get("llm_calls", 0)
                + (1 if config["method"] == "groq_agent" else 0),
            }
        in_total = state.get("input_tokens", 0) + response.input_tokens
        out_total = state.get("output_tokens", 0) + response.output_tokens
        usage = state.get("model_usage", []) + [response.usage()]
        if in_total + out_total > config["max_total_tokens"]:
            return {
                "status": "token_budget_exhausted",
                "provider_calls": calls,
                "model_usage": usage,
                "input_tokens": in_total,
                "output_tokens": out_total,
            }
        return {
            "proposal_data": response.data,
            "model_usage": usage,
            "provider_calls": calls,
            "llm_calls": state.get("llm_calls", 0) + (1 if response.provider == "groq" else 0),
            "input_tokens": in_total,
            "output_tokens": out_total,
            "status": "proposed",
        }

    def validate(state: ProposalState) -> dict:
        try:
            proposal = Proposal.model_validate(state["proposal_data"])
            result = validate_proposal(
                proposal,
                evidence_ids=local["evidence_ids"],
                discovery_ids=set(local["discoveries"]),
                max_steps=config["training_timesteps"],
                allowed_seeds={config["starting_seed"]},
            )
            if result["status"] == "valid":
                digest = proposal_hash(
                    proposal,
                    checkpoint_sha256=local["baseline_metadata"]["sha256"],
                    development_hashes=local["development_hashes"],
                    continuation_seed=5000 + proposal.starting_seed,
                )
                if digest in previous:
                    result = {"status": "rejected", "reasons": ["duplicate_configuration"]}
                else:
                    result["configuration_hash"] = digest
            return {
                "proposal_data": proposal.model_dump(mode="json"),
                "validation": result,
                "status": "validated" if result["status"] == "valid" else "rejected",
            }
        except Exception as exc:
            return {
                "validation": {
                    "status": "rejected",
                    "reasons": ["invalid_structured_proposal"],
                    "error_type": type(exc).__name__,
                },
                "status": "rejected",
            }

    def planner(state: ProposalState) -> dict:
        proposal = Proposal.model_validate(state["proposal_data"])
        plan = {
            "schema_version": "sirfex-research-plan:1",
            "starting_checkpoint_sha256": local["baseline_metadata"]["sha256"],
            "development_hashes": local["development_hashes"],
            "training_timesteps": proposal.training_timesteps,
            "continuation_seed": 5000 + proposal.starting_seed,
            "estimated_ppo_seconds_at_480_sps": proposal.training_timesteps / 480,
            "human_approval_required": True,
            "final_holdouts_allowed": False,
            "configuration_hash": state["validation"]["configuration_hash"],
        }
        return {"plan": plan, "status": "awaiting_approval"}

    builder = StateGraph(ProposalState)
    builder.add_node("failure_analyst", analyst)
    builder.add_node("improvement_strategist", strategist)
    builder.add_node("validate_proposal", validate)
    builder.add_node("experiment_planner", planner)
    builder.add_edge(START, "failure_analyst")
    builder.add_edge("failure_analyst", "improvement_strategist")
    builder.add_conditional_edges(
        "improvement_strategist",
        lambda state: "validate_proposal" if state.get("status") == "proposed" else END,
        {"validate_proposal": "validate_proposal", END: END},
    )
    builder.add_conditional_edges(
        "validate_proposal",
        lambda state: (
            "experiment_planner"
            if state.get("status") == "validated"
            else "improvement_strategist"
            if state.get("status") == "rejected"
            and state.get("provider_calls", 0) < config["max_provider_calls"]
            and state.get("validation", {}).get("reasons") != ["duplicate_configuration"]
            else END
        ),
        {
            "experiment_planner": "experiment_planner",
            "improvement_strategist": "improvement_strategist",
            END: END,
        },
    )
    builder.add_edge("experiment_planner", END)
    return builder.compile(checkpointer=InMemorySaver())


def propose(config_value: dict, training, research: ResearchStore) -> dict:
    config = validate_config(config_value)
    context, local = evidence_context(
        training,
        research,
        diagnostic_run_id=config["diagnostic_run_id"],
        starting_seed=config["starting_seed"],
        training_timesteps=config["training_timesteps"],
    )
    previous = _previous_hashes(research)
    provider = _provider(config["method"])
    protocol = {
        "schema_version": "sirfex-agent-research-protocol:1",
        "config": config,
        "prompt_template_version": PROMPT_VERSION,
        "evidence_context_sha256": content_hash(context),
        "development_hashes": local["development_hashes"],
        "source_checkpoint_sha256": local["baseline_metadata"]["sha256"],
        "final_holdout_access": False,
        "decision_authority": "sirfex-deterministic-validation",
    }
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:10]}"
    manifest = {
        "schema_version": "sirfex-research:1",
        "run_id": run_id,
        "kind": "agent-improvement-research",
        "status": "proposing",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "sirfex_version": __version__,
        "method": config["method"],
        "provider_model": provider.model,
        "protocol_sha256": content_hash(protocol),
    }
    research.create(
        run_id,
        {"manifest.json": manifest, "protocol.json": protocol, "evidence_context.json": context},
    )
    started = time.perf_counter()
    try:
        graph = _graph(provider, config, local, previous)
        state: ProposalState = graph.invoke(
            {
                "context": context,
                "model_usage": [],
                "provider_calls": 0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            {"configurable": {"thread_id": run_id}, "recursion_limit": 12},
        )
        research.write(run_id, "analysis.json", state.get("analysis", {}))
        if "proposal_data" in state:
            research.write(run_id, "proposals.json", state["proposal_data"])
        if "validation" in state:
            research.write(run_id, "validation.json", state["validation"])
        if "plan" in state:
            research.write(run_id, "plan.json", state["plan"])
            research.write(
                run_id,
                "approvals.json",
                {"status": "pending", "configuration_hash": state["plan"]["configuration_hash"]},
            )
        usage = {
            "provider_calls": state.get("provider_calls", 0),
            "llm_calls": state.get("llm_calls", 0),
            "input_tokens": state.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0),
            "requests": state.get("model_usage", []),
            "estimated_usd": sum(
                row.get("estimated_usd") or 0 for row in state.get("model_usage", [])
            ),
        }
        research.write(run_id, "model_usage.json", usage)
        research.write(
            run_id,
            "research_memory.json",
            {
                "schema_version": "sirfex-research-memory:1",
                "attempt_id": run_id,
                "parent_attempt_id": None,
                "model_identifier": provider.model,
                "proposal_schema_version": state.get("proposal_data", {}).get("schema_version"),
                "hypothesis": state.get("proposal_data", {}).get("hypothesis"),
                "evidence_references": state.get("proposal_data", {}).get(
                    "supporting_evidence", []
                ),
                "intervention": state.get("proposal_data", {}).get("intervention"),
                "validation_decision": state.get("validation", {}),
                "configuration_hash": state.get("validation", {}).get("configuration_hash"),
                "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
                "dataset_hashes": protocol["development_hashes"],
                "training_budget": config["training_timesteps"],
                "actual_training_interactions": 0,
                "actual_stress_exposure": 0,
                "nominal_development_results": None,
                "stress_development_results": None,
                "deterministic_eligibility_decision": None,
                "model_generated_interpretation": None,
                "model_usage": usage,
                "final_attempt_status": state["status"],
            },
        )
        manifest["status"] = state["status"]
        if "error_type" in state:
            manifest["error_type"] = state["error_type"]
        summary = {
            "run_id": run_id,
            "status": manifest["status"],
            "method": config["method"],
            "provider_model": provider.model,
            "proposal_valid": state.get("validation", {}).get("status") == "valid",
            "configuration_hash": state.get("validation", {}).get("configuration_hash"),
            "model_usage": usage,
            "runtime_seconds": time.perf_counter() - started,
            "final_holdouts_accessed": False,
        }
        research.write(run_id, "summary.json", summary)
        return summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        research.write(run_id, "manifest.json", manifest)
