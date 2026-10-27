"""Approved, development-only PPO execution through existing SIRFEX primitives."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TypedDict

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.research.context import evidence_context, verified_discoveries
from sirfex.research.proposals import (
    DiscoveryWeighting,
    Proposal,
    executable_schedule,
    proposal_hash,
    validate_proposal,
)
from sirfex.research.store import ResearchStore
from sirfex.training.checkpoints import verify_checkpoint
from sirfex.training.config import PPOConfig
from sirfex.training.exposure import exposure_adequacy
from sirfex.training.matched_study import (
    ACTION,
    OBSERVATION,
    REWARD,
    TERMINATION,
    _evaluate,
    model_fingerprint,
)
from sirfex.training.phase35 import SOURCE_32, SOURCE_34
from sirfex.training.ppo import train_ppo
from sirfex.training.protocol import content_hash, scenario_record
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.remediation_diagnostics import metadata_from_dict
from sirfex.training.validation import _checkpoint


class ExecutionState(TypedDict, total=False):
    run_id: str
    approved: bool
    status: str
    experiment: dict
    development_evaluation: dict
    decision: dict
    critique: dict


def _development(training, seed):
    source32 = training.load(SOURCE_32)
    old34 = training.load(SOURCE_34)
    normal = [
        Scenario.from_dict(row["scenario"])
        for row in source32["dataset_partitions"]["groups"]["training"]
    ]
    nominal = [
        Scenario.from_dict(row["scenario"])
        for row in source32["dataset_partitions"]["groups"]["validation"]
    ]
    stress_rows = old34["stress_benchmark"]["groups"]["screening_stress"]
    stress = [Scenario.from_dict(row["scenario"]) for row in stress_rows]
    source35 = training.load("20261027T143749_d88d2dd37f")
    if (
        content_hash([scenario_record(s) for s in normal]) != source35["protocol"]["training_hash"]
        or content_hash([scenario_record(s) for s in nominal])
        != source35["protocol"]["development_nominal_hash"]
        or content_hash(stress_rows) != source35["protocol"]["development_stress_hash"]
    ):
        raise ValueError("Development partition hash changed")
    baseline = metadata_from_dict(source32["baseline_selection"]["selected"][str(seed)])
    verify_checkpoint(baseline.path, source35["protocol"]["baseline_sha256"][str(seed)])
    return normal, nominal, stress, baseline, source35


def execute(
    run_id: str, *, approved: bool, training, research: ResearchStore, experiments_store
) -> dict:
    manifest = research.read(run_id, "manifest.json")
    protocol = research.read(run_id, "protocol.json")
    if manifest.get("kind") != "agent-improvement-research":
        raise ValueError("Expected agent improvement study")
    if content_hash(protocol) != manifest.get("protocol_sha256"):
        raise ValueError("Research protocol hash changed")
    if manifest.get("status") != "awaiting_approval":
        raise ValueError("Research proposal is not awaiting approval")
    if not approved:
        raise ValueError("Explicit --approve-training is required")
    config = protocol["config"]
    proposal = Proposal.model_validate(research.read(run_id, "proposals.json"))
    context, local = evidence_context(
        training,
        research,
        diagnostic_run_id=config["diagnostic_run_id"],
        starting_seed=config["starting_seed"],
        training_timesteps=config["training_timesteps"],
    )
    if content_hash(context) != protocol["evidence_context_sha256"]:
        raise ValueError("Proposal evidence context changed")
    validation = validate_proposal(
        proposal,
        evidence_ids=local["evidence_ids"],
        discovery_ids=set(local["discoveries"]),
        max_steps=config["training_timesteps"],
        allowed_seeds={config["starting_seed"]},
    )
    saved_validation = research.read(run_id, "validation.json")
    expected_hash = proposal_hash(
        proposal,
        checkpoint_sha256=local["baseline_metadata"]["sha256"],
        development_hashes=local["development_hashes"],
        continuation_seed=5000 + proposal.starting_seed,
    )
    if (
        validation["status"] != "valid"
        or saved_validation.get("configuration_hash") != expected_hash
        or research.read(run_id, "plan.json")["configuration_hash"] != expected_hash
    ):
        raise ValueError("Proposal no longer matches validated and frozen plan")
    if proposal.training_timesteps != config["training_timesteps"]:
        raise ValueError("Approved interaction cap changed")
    normal_training, nominal, stress, baseline, source35 = _development(
        training, proposal.starting_seed
    )
    if baseline.sha256 != protocol["source_checkpoint_sha256"]:
        raise ValueError("Starting checkpoint identity changed")
    discovery = verified_discoveries(training)
    pool = [discovery[key]["scenario"] for key in sorted(discovery)]
    discovery_map = {discovery[key]["scenario_id"]: key for key in discovery}
    schedule = executable_schedule(proposal)
    weights = None
    target = None
    if isinstance(proposal.intervention, DiscoveryWeighting):
        weights = {
            discovery[key]["scenario_id"]: value
            for key, value in proposal.intervention.weights.items()
        }
        target = proposal.intervention.stress_fraction
    manifest["status"] = "training"
    research.write(run_id, "manifest.json", manifest)
    research.write(
        run_id,
        "approvals.json",
        {
            "status": "approved",
            "approved_by": "explicit_cli_flag",
            "approved_at_utc": datetime.now(UTC).isoformat(),
            "configuration_hash": expected_hash,
            "training_timesteps": proposal.training_timesteps,
        },
    )

    def build_graph():
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise ImportError("Install the agents extra for execution graph") from exc

        def approval_gate(state: ExecutionState) -> dict:
            return {"status": "approved" if state["approved"] else "approval_required"}

        def training_node(state: ExecutionState) -> dict:
            del state
            env = PumpGymEnv(
                normal_training,
                seed=5000 + proposal.starting_seed,
                observation_schema=OBSERVATION,
                stress_scenarios=pool,
                discovery_ids=discovery_map,
                stress_schedule=schedule,
                stress_target_fraction=target,
                stress_weights=weights,
                policy_action_schema=ACTION,
                reward_schema=REWARD,
                termination_schema=TERMINATION,
            )
            ppo = PPOConfig.from_dict(baseline.ppo_config)
            fingerprint = {}

            def check_start(model):
                fingerprint.update(model_fingerprint(model))
                if (
                    fingerprint["sha256"]
                    != source35["protocol"]["baseline_fingerprint"][str(proposal.starting_seed)]
                ):
                    raise ValueError("Starting PPO policy or optimizer state differs")

            started = time.perf_counter()
            try:
                model = train_ppo(
                    env,
                    ppo,
                    seed=5000 + proposal.starting_seed,
                    timesteps=proposal.training_timesteps,
                    checkpoint=baseline.path,
                    expected_sha256=baseline.sha256,
                    wall_seconds=config["wall_seconds"],
                    on_model=check_start,
                )
                actual = model.num_timesteps - baseline.training_timesteps
                exposure = env.exposure_summary(
                    scheduled_interactions=proposal.training_timesteps,
                    model_interactions=actual,
                )
                meta = _checkpoint(
                    model,
                    research.directory(run_id),
                    run_id,
                    "research_candidate",
                    {
                        "ppo": ppo.to_dict(),
                        "policy_action_schema": ACTION,
                        "reward_schema": REWARD,
                        "termination_schema": TERMINATION,
                    },
                    5000 + proposal.starting_seed,
                    OBSERVATION,
                )
                verify_checkpoint(meta.path, meta.sha256)
                record = {
                    "requested_additional_steps": proposal.training_timesteps,
                    "actual_additional_steps": actual,
                    "starting_checkpoint_sha256": baseline.sha256,
                    "starting_fingerprint": fingerprint,
                    "checkpoint": meta.to_dict(),
                    "training_seconds": time.perf_counter() - started,
                    "exposure": exposure,
                    "configuration_hash": expected_hash,
                }
                research.write(run_id, "experiments.json", record)
                if not exposure["reconciliation_valid"]:
                    raise ValueError("Training exposure did not reconcile")
                if actual != proposal.training_timesteps:
                    return {"experiment": record, "status": "interrupted"}
                return {"experiment": record, "status": "trained"}
            finally:
                env.close()

        def evaluate_node(state: ExecutionState) -> dict:
            meta = metadata_from_dict(state["experiment"]["checkpoint"])
            from sirfex.training.controller import PPOController

            model = PPOController.load(meta.path, meta)._model
            value = {
                "development_nominal": _evaluate(model, meta, nominal, "development_nominal"),
                "development_stress": _evaluate(model, meta, stress, "development_stress"),
            }
            research.write(run_id, "development_evaluation.json", value)
            replays = []
            for partition, scenarios in (
                ("development_nominal", nominal),
                ("development_stress", stress),
            ):
                record = run_and_save_experiment(
                    experiments_store,
                    scenarios[0],
                    "ppo",
                    max_steps=200,
                    controller_config={"checkpoint": meta.to_dict()},
                )
                replays.append(
                    {
                        "partition": partition,
                        "experiment_id": record.experiment_id,
                        "result": replay_experiment(
                            experiments_store, record.experiment_id
                        ).to_dict(),
                    }
                )
            research.write(run_id, "replay_verification.json", {"records": replays})
            return {"development_evaluation": value, "status": "evaluated"}

        def guardrails_node(state: ExecutionState) -> dict:
            actual = state["experiment"]["actual_additional_steps"]
            exposure = state["experiment"]["exposure"]
            gate = exposure_adequacy(
                exposure, len(pool), trial_complete=actual == proposal.training_timesteps
            )
            if config["profile"] == "smoke":
                decision = {
                    "status": "smoke_inconclusive",
                    "exposure_gate": gate,
                    "reason": "Smoke interactions lack an equal-budget normal control",
                }
            else:
                control = next(
                    row
                    for row in source35["screening_runs"]["records"]
                    if row["seed"] == proposal.starting_seed and row["candidate_id"] == "normal"
                )
                if control["actual_additional_steps"] != proposal.training_timesteps:
                    raise ValueError("No equal-budget historical normal control")
                n = state["development_evaluation"]["development_nominal"]
                s = state["development_evaluation"]["development_stress"]
                cn, cs = control["development_nominal"], control["development_stress"]
                checks = {
                    "exposure": gate["status"] == "adequate_exposure",
                    "nominal_mae": n["mean_tracking_error"] <= cn["mean_tracking_error"] * 1.10,
                    "nominal_violations": n["constraint_violation_frequency"]
                    <= cn["constraint_violation_frequency"],
                    "stress_violations": s["constraint_violation_frequency"]
                    <= cs["constraint_violation_frequency"],
                    "execution": all(
                        state["development_evaluation"][part]["execution_failures"]
                        <= control[part]["execution_failures"]
                        for part in ("development_nominal", "development_stress")
                    ),
                    "stress_gain": (
                        s["constraint_violation_frequency"] < cs["constraint_violation_frequency"]
                        or s["mean_tracking_error"] <= cs["mean_tracking_error"] - 0.005
                    ),
                }
                decision = {
                    "status": "development_eligible"
                    if all(checks.values())
                    else "development_ineligible",
                    "checks": checks,
                    "exposure_gate": gate,
                    "control_checkpoint_sha256": control["checkpoint"]["sha256"],
                    "normal_mae_relative_change": n["mean_tracking_error"]
                    / cn["mean_tracking_error"]
                    - 1,
                    "stress_mae_difference": s["mean_tracking_error"] - cs["mean_tracking_error"],
                }
            research.write(run_id, "decision.json", decision)
            return {"decision": decision, "status": "classified"}

        def critic_node(state: ExecutionState) -> dict:
            decision = state["decision"]
            critique = {
                "source": "deterministic-template"
                if config["method"] == "non_llm"
                else "offline-mock-critic"
                if config["method"] == "mock_agent"
                else "deterministic-fallback-after-training",
                "interpretation": (
                    "The smoke run verifies the execution pipeline; it is not evidence of "
                    "controller improvement."
                    if decision["status"] == "smoke_inconclusive"
                    else "The frozen development checks decide eligibility; "
                    "this narrative cannot override it."
                ),
                "authoritative_decision": decision["status"],
            }
            research.write(run_id, "agent_critiques.json", critique)
            return {"critique": critique, "status": "completed"}

        graph = StateGraph(ExecutionState)
        graph.add_node("approval_gate", approval_gate)
        graph.add_node("bounded_training", training_node)
        graph.add_node("development_evaluation", evaluate_node)
        graph.add_node("deterministic_guardrails", guardrails_node)
        graph.add_node("results_critic", critic_node)
        graph.add_edge(START, "approval_gate")
        graph.add_conditional_edges(
            "approval_gate",
            lambda s: "bounded_training" if s["status"] == "approved" else END,
            {"bounded_training": "bounded_training", END: END},
        )
        graph.add_conditional_edges(
            "bounded_training",
            lambda s: "development_evaluation" if s["status"] == "trained" else END,
            {"development_evaluation": "development_evaluation", END: END},
        )
        graph.add_edge("development_evaluation", "deterministic_guardrails")
        graph.add_edge("deterministic_guardrails", "results_critic")
        graph.add_edge("results_critic", END)
        return graph.compile(checkpointer=InMemorySaver())

    try:
        graph = build_graph()
        state: ExecutionState = graph.invoke(
            {"run_id": run_id, "approved": approved},
            {"configurable": {"thread_id": run_id}, "recursion_limit": 12},
        )
        manifest["status"] = state["status"]
        summary = {
            "run_id": run_id,
            "status": state["status"],
            "method": config["method"],
            "training_interactions": state.get("experiment", {}).get("actual_additional_steps"),
            "actual_stress_interactions": state.get("experiment", {})
            .get("exposure", {})
            .get("stress_interactions"),
            "development_decision": state.get("decision", {}).get("status"),
            "final_holdouts_accessed": False,
        }
        try:
            memory = research.read(run_id, "research_memory.json")
        except OSError:
            memory = {
                "schema_version": "sirfex-research-memory:1",
                "attempt_id": run_id,
                "parent_attempt_id": None,
            }
        memory.update(
            {
                "actual_training_interactions": state.get("experiment", {}).get(
                    "actual_additional_steps", 0
                ),
                "actual_stress_exposure": state.get("experiment", {})
                .get("exposure", {})
                .get("stress_interactions", 0),
                "nominal_development_results": state.get("development_evaluation", {}).get(
                    "development_nominal"
                ),
                "stress_development_results": state.get("development_evaluation", {}).get(
                    "development_stress"
                ),
                "deterministic_eligibility_decision": state.get("decision", {}).get("status"),
                "model_generated_interpretation": None,
                "research_interpretation": state.get("critique", {}).get("interpretation"),
                "final_attempt_status": state["status"],
            }
        )
        research.write(run_id, "research_memory.json", memory)
        research.write(run_id, "summary.json", summary)
        return summary
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        research.write(run_id, "manifest.json", manifest)
