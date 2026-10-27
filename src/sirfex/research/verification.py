"""Recompute research integrity checks from trusted local evidence."""

from __future__ import annotations

from sirfex.research.context import evidence_context
from sirfex.research.proposals import Proposal, proposal_hash, validate_proposal
from sirfex.training.checkpoints import verify_checkpoint
from sirfex.training.protocol import content_hash


def validate_saved(run_id, training, research) -> dict:
    manifest = research.read(run_id, "manifest.json")
    protocol = research.read(run_id, "protocol.json")
    if manifest.get("kind") != "agent-improvement-research" or content_hash(
        protocol
    ) != manifest.get("protocol_sha256"):
        raise ValueError("Research protocol is invalid")
    config = protocol["config"]
    context, local = evidence_context(
        training,
        research,
        diagnostic_run_id=config["diagnostic_run_id"],
        starting_seed=config["starting_seed"],
        training_timesteps=config["training_timesteps"],
        include_claims=protocol.get("evidence_context_schema_version")
        == "sirfex-research-context:2",
    )
    if content_hash(context) != protocol["evidence_context_sha256"]:
        raise ValueError("Research evidence context changed")
    proposal = Proposal.model_validate(research.read(run_id, "proposals.json"))
    result = validate_proposal(
        proposal,
        evidence_ids=local["evidence_ids"],
        discovery_ids=set(local["discoveries"]),
        max_steps=config["training_timesteps"],
        allowed_seeds={config["starting_seed"]},
        claim_catalog=local.get("claim_catalog"),
    )
    expected = proposal_hash(
        proposal,
        checkpoint_sha256=local["baseline_metadata"]["sha256"],
        development_hashes=local["development_hashes"],
        continuation_seed=5000 + proposal.starting_seed,
    )
    saved = research.read(run_id, "validation.json")
    if saved.get("status") == "valid":
        if result["status"] != "valid" or saved.get("configuration_hash") != expected:
            raise ValueError("Saved proposal validation changed")
    elif saved.get("status") != "rejected":
        raise ValueError("Saved proposal validation is invalid")
    return {
        "run_id": run_id,
        "current_validation": result,
        "saved_validation": saved,
        "configuration_hash": expected,
        "evidence_context_verified": True,
    }


def verify_run(run_id, research) -> dict:
    manifest = research.read(run_id, "manifest.json")
    protocol = research.read(run_id, "protocol.json")
    if content_hash(protocol) != manifest.get("protocol_sha256"):
        raise ValueError("Research protocol hash mismatch")
    result = {
        "run_id": run_id,
        "kind": manifest.get("kind"),
        "status": manifest.get("status"),
        "protocol_verified": True,
    }
    if (
        manifest.get("kind") == "agent-improvement-research"
        and manifest.get("status") == "completed"
    ):
        experiment = research.read(run_id, "experiments.json")
        checkpoint = experiment["checkpoint"]
        verify_checkpoint(checkpoint["path"], checkpoint["sha256"])
        if (
            experiment["configuration_hash"]
            != research.read(run_id, "validation.json")["configuration_hash"]
            or not experiment["exposure"]["reconciliation_valid"]
            or experiment["actual_additional_steps"] != protocol["config"]["training_timesteps"]
        ):
            raise ValueError("Research execution evidence did not reconcile")
        result["checkpoint_verified"] = True
        result["exposure_verified"] = True
    return result
