"""Strict, versioned development intervention proposals and deterministic validation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sirfex.training.protocol import content_hash


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceRef(StrictModel):
    type: Literal["diagnostic", "historical", "discovery"]
    artifact_id: str = Field(min_length=5, max_length=100)
    claim: str = Field(min_length=8, max_length=300)


class ScheduleSegment(StrictModel):
    until_step: int = Field(gt=0, le=51200)
    stress_fraction: float = Field(ge=0, le=0.15)


class ScheduledCurriculum(StrictModel):
    type: Literal["scheduled_curriculum"]
    stress_pool: Literal["expanded_7"]
    schedule: list[ScheduleSegment] = Field(min_length=2, max_length=3)


class NominalRehearsal(StrictModel):
    type: Literal["nominal_rehearsal"]
    stress_pool: Literal["expanded_7"]
    stress_fraction: float = Field(gt=0, le=0.15)
    final_nominal_fraction: float = Field(ge=0.2, le=0.5)


class DiscoveryWeighting(StrictModel):
    type: Literal["discovery_weighting"]
    stress_pool: Literal["expanded_7"]
    stress_fraction: float = Field(gt=0, le=0.15)
    weights: dict[str, float] = Field(min_length=7, max_length=7)


Intervention = Annotated[
    ScheduledCurriculum | NominalRehearsal | DiscoveryWeighting,
    Field(discriminator="type"),
]


class Proposal(StrictModel):
    schema_version: Literal["sirfex-agent-proposal:1"]
    hypothesis: str = Field(min_length=20, max_length=500)
    supporting_evidence: list[EvidenceRef] = Field(min_length=1, max_length=5)
    starting_seed: Literal[3, 17]
    training_timesteps: int = Field(ge=256, le=51200)
    intervention: Intervention
    evaluation_policy: Literal["development_only"]
    expected_effect: str = Field(min_length=12, max_length=300)
    possible_failure_mode: str = Field(min_length=12, max_length=300)

    @field_validator("training_timesteps")
    @classmethod
    def rollout_multiple(cls, value: int) -> int:
        if value % 128:
            raise ValueError("Training steps must be a multiple of 128")
        return value


def proposal_hash(
    proposal: Proposal, *, checkpoint_sha256: str, development_hashes: dict, continuation_seed: int
) -> str:
    """Deduplicate by intervention and experiment contract, not prose."""
    return content_hash(
        {
            "seed": proposal.starting_seed,
            "timesteps": proposal.training_timesteps,
            "intervention": proposal.intervention.model_dump(mode="json"),
            "checkpoint_sha256": checkpoint_sha256,
            "development_hashes": development_hashes,
            "continuation_seed": continuation_seed,
        }
    )


def validate_proposal(
    proposal: Proposal,
    *,
    evidence_ids: set[str],
    discovery_ids: set[str],
    max_steps: int,
    allowed_seeds: set[int] = frozenset({3, 17}),
    claim_catalog: dict[str, dict] | None = None,
) -> dict:
    """Independent allowlist validation; an LLM cannot approve its own output."""
    failures = []
    if proposal.starting_seed not in allowed_seeds:
        failures.append("starting_seed_not_allowed")
    if proposal.training_timesteps > max_steps:
        failures.append("training_budget_exceeded")
    for ref in proposal.supporting_evidence:
        allowed = discovery_ids if ref.type == "discovery" else evidence_ids
        if ref.artifact_id not in allowed:
            failures.append(f"unknown_{ref.type}_evidence:{ref.artifact_id}")
        if claim_catalog is not None:
            canonical = claim_catalog.get(ref.artifact_id)
            if (
                canonical is None
                or ref.type != canonical["type"]
                or ref.claim != canonical["claim"]
            ):
                failures.append(f"unsupported_evidence_claim:{ref.artifact_id}")
    intervention = proposal.intervention
    if isinstance(intervention, ScheduledCurriculum):
        endpoints = [segment.until_step for segment in intervention.schedule]
        if (
            endpoints[-1] != proposal.training_timesteps
            or endpoints != sorted(set(endpoints))
            or any(step % 128 for step in endpoints)
        ):
            failures.append("invalid_schedule_boundaries")
        if all(segment.stress_fraction == 0 for segment in intervention.schedule):
            failures.append("schedule_has_no_stress_intervention")
    elif isinstance(intervention, DiscoveryWeighting):
        if set(intervention.weights) != discovery_ids:
            failures.append("unknown_or_missing_discovery_weight")
        if (
            any(
                type(v) not in (int, float) or not 0.05 <= v <= 0.4
                for v in intervention.weights.values()
            )
            or abs(sum(intervention.weights.values()) - 1) > 1e-6
        ):
            failures.append("invalid_discovery_weights")
    elif isinstance(intervention, NominalRehearsal):
        tail_steps = round(proposal.training_timesteps * intervention.final_nominal_fraction / 128)
        if tail_steps < 1 or tail_steps >= proposal.training_timesteps // 128:
            failures.append("invalid_nominal_rehearsal_boundary")
    return {
        "status": "valid" if not failures else "rejected",
        "reasons": failures,
        "schema_version": proposal.schema_version,
    }


def executable_schedule(proposal: Proposal) -> list[dict] | None:
    intervention = proposal.intervention
    if isinstance(intervention, ScheduledCurriculum):
        return [item.model_dump() for item in intervention.schedule]
    if isinstance(intervention, NominalRehearsal):
        tail = round(proposal.training_timesteps * intervention.final_nominal_fraction / 128) * 128
        return [
            {
                "until_step": proposal.training_timesteps - tail,
                "stress_fraction": intervention.stress_fraction,
            },
            {"until_step": proposal.training_timesteps, "stress_fraction": 0.0},
        ]
    return None
