"""Bounded proposal and comparison rules; no network or PPO training."""

import copy

import pytest

from sirfex.research.comparison import compare
from sirfex.research.proposals import Proposal, validate_proposal
from sirfex.research.providers import MockProvider
from sirfex.research.store import ResearchStore
from sirfex.research.workflow import _graph, validate_config


def context():
    return {
        "diagnostic_run_id": "diagnostic_123",
        "starting_seed": 3,
        "training_timesteps": 256,
        "normal_to_stress_action_shift": {},
        "development_metrics": {},
    }


def config():
    return {
        "profile": "smoke",
        "method": "mock_agent",
        "diagnostic_run_id": "diagnostic_123",
        "starting_seed": 3,
        "training_timesteps": 256,
        "max_provider_calls": 2,
        "max_total_tokens": 2048,
        "max_completion_tokens": 512,
        "wall_seconds": 120,
    }


def test_registry_rejects_unknown_fields_holdout_and_budget():
    proposal = MockProvider().propose(context(), max_completion_tokens=512).data
    parsed = Proposal.model_validate(proposal)
    assert (
        validate_proposal(
            parsed, evidence_ids={"diagnostic_123"}, discovery_ids=set(), max_steps=256
        )["status"]
        == "valid"
    )
    bad = copy.deepcopy(proposal)
    bad["evaluation_policy"] = "final_holdout"
    with pytest.raises(ValueError):
        Proposal.model_validate(bad)
    bad = copy.deepcopy(proposal)
    bad["reward_schema"] = "rewrite"
    with pytest.raises(ValueError):
        Proposal.model_validate(bad)
    bad = copy.deepcopy(proposal)
    bad["training_timesteps"] = 512
    assert (
        "training_budget_exceeded"
        in validate_proposal(
            Proposal.model_validate(bad),
            evidence_ids={"diagnostic_123"},
            discovery_ids=set(),
            max_steps=256,
        )["reasons"]
    )
    bad = copy.deepcopy(proposal)
    bad["supporting_evidence"][0]["artifact_id"] = "unknown_evidence"
    assert (
        validate_proposal(
            Proposal.model_validate(bad),
            evidence_ids={"diagnostic_123"},
            discovery_ids=set(),
            max_steps=256,
        )["status"]
        == "rejected"
    )


def test_graph_terminates_at_approval_and_deduplicates():
    cfg = validate_config(config())
    local = {
        "evidence_ids": {"diagnostic_123"},
        "discoveries": {},
        "baseline_metadata": {"sha256": "a" * 64},
        "development_hashes": {"nominal": "b", "stress": "c"},
    }
    state = _graph(MockProvider(), cfg, local, set()).invoke(
        {
            "context": context(),
            "provider_calls": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_usage": [],
        },
        {"configurable": {"thread_id": "test1"}},
    )
    assert state["status"] == "awaiting_approval"
    assert state["provider_calls"] == 1
    duplicate = _graph(
        MockProvider(), cfg, local, {state["validation"]["configuration_hash"]}
    ).invoke(
        {
            "context": context(),
            "provider_calls": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_usage": [],
        },
        {"configurable": {"thread_id": "test2"}},
    )
    assert duplicate["status"] == "rejected"
    assert duplicate["provider_calls"] == 1


def test_provider_failure_is_recorded_without_training():
    class Failing:
        def propose(self, *args, **kwargs):
            raise TimeoutError("no network")

    state = _graph(Failing(), validate_config(config()), {}, set()).invoke(
        {
            "context": context(),
            "provider_calls": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_usage": [],
        },
        {"configurable": {"thread_id": "failure"}},
    )
    assert state["status"] == "provider_failed"
    assert state["error_type"] == "TimeoutError"
    assert state["provider_calls"] == 1


def test_invalid_output_retries_at_most_twice():
    class Invalid:
        def propose(self, *args, **kwargs):
            result = MockProvider().propose(context(), max_completion_tokens=512)
            return type(result)(**{**result.__dict__, "data": {"unexpected": True}})

    state = _graph(Invalid(), validate_config(config()), {}, set()).invoke(
        {
            "context": context(),
            "provider_calls": 0,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_usage": [],
        },
        {"configurable": {"thread_id": "invalid"}},
    )
    assert state["status"] == "rejected"
    assert state["provider_calls"] == 2
    assert state["validation"]["reasons"] == ["invalid_structured_proposal"]


def test_compare_rejects_incomplete_run(tmp_path):
    store = ResearchStore(tmp_path)
    store.create("agent_123", {"manifest.json": {"status": "pending"}})
    store.create("baseline_123", {"manifest.json": {"status": "pending"}})
    with pytest.raises((OSError, ValueError)):
        compare(store, "agent_123", "baseline_123")
