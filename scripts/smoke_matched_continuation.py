"""Exercise both continuation arms on known stress without opening final holdouts."""

import argparse
import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from sirfex.core import Scenario
from sirfex.training.checkpoints import verify_checkpoint, write_json
from sirfex.training.config import PPOConfig
from sirfex.training.controller import PPOController
from sirfex.training.evaluation import evaluate_controller
from sirfex.training.matched_study import model_fingerprint
from sirfex.training.persistence import TrainingStore
from sirfex.training.ppo import train_ppo
from sirfex.training.pump_gym import PumpGymEnv
from sirfex.training.remediation_diagnostics import metadata_from_dict
from sirfex.training.validation import _checkpoint


def smoke(prepared_id, store):
    prepared = store.load(prepared_id, allow_partial=True)
    if prepared["manifest"]["status"] != "prepared":
        raise ValueError("Expected frozen prepared study")
    baseline = prepared["baseline_verification"]["3"]
    meta = metadata_from_dict(baseline["checkpoint"])
    groups = prepared["partitions"]["groups"]
    normal = [Scenario.from_dict(row["scenario"]) for row in groups["training"]]
    known = [Scenario.from_dict(row["scenario"]) for row in groups["known_stress"]]
    run_id = f"smoke_{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory = store.directory(run_id)
    directory.mkdir()
    ppo = PPOConfig.from_dict(meta.ppo_config)
    records, checkpoints = [], {}
    for arm, ratio in (("normal", 0.0), ("robust", 0.3)):
        env = PumpGymEnv(
            normal,
            seed=5003,
            observation_schema=meta.observation_schema,
            policy_action_schema=meta.policy_action_schema,
            reward_schema=meta.reward_schema,
            termination_schema=meta.termination_schema,
            stress_scenarios=known,
            stress_ratio=ratio,
        )
        starting = {}

        def remember(model, *, state=starting):
            state.update(model_fingerprint(model))
            if state["sha256"] != baseline["fingerprint"]["sha256"]:
                raise ValueError("Smoke branch state differs from selected baseline")

        started = time.perf_counter()
        model = train_ppo(
            env,
            ppo,
            seed=5003,
            timesteps=128,
            checkpoint=meta.path,
            expected_sha256=meta.sha256,
            wall_seconds=90,
            on_model=remember,
        )
        saved = _checkpoint(
            model,
            directory,
            run_id,
            f"{arm}_3",
            {
                "ppo": ppo.to_dict(),
                "policy_action_schema": meta.policy_action_schema,
                "reward_schema": meta.reward_schema,
                "termination_schema": meta.termination_schema,
            },
            5003,
            meta.observation_schema,
        )
        verify_checkpoint(saved.path, saved.sha256)
        evaluation = evaluate_controller(
            lambda m=model, d=saved: PPOController(m, d),
            known,
            "smoke_known_stress",
            max_steps=200,
            observation_schema=meta.observation_schema,
            reward_schema=meta.reward_schema,
        ).to_dict()
        records.append(
            {
                "arm": arm,
                "starting_sha256": meta.sha256,
                "starting_fingerprint": starting,
                "additional_timesteps": model.num_timesteps - meta.training_timesteps,
                "checkpoint": saved.to_dict(),
                "evaluation": evaluation,
                "sampled_episodes": dict(env.sample_counts),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        checkpoints[arm] = saved.to_dict()
        env.close()
    if any(r["additional_timesteps"] != 128 for r in records):
        raise ValueError("Smoke continuation budget mismatch")
    if records[0]["starting_fingerprint"] != records[1]["starting_fingerprint"]:
        raise ValueError("Smoke branches did not restore identical optimizer state")
    report = {
        "schema_version": "matched-smoke:1",
        "training_run_id": run_id,
        "source_study": prepared_id,
        "status": "completed",
        "note": "Software/throughput check only; no final holdout outcomes or research claim",
        "records": records,
    }
    write_json(
        directory / "manifest.json",
        {
            "schema_version": "2",
            "phase": "3.3",
            "kind": "matched-smoke",
            "training_run_id": run_id,
            "status": "completed",
            "checkpoints": checkpoints,
        },
    )
    write_json(directory / "evaluation.json", report)
    return {
        "training_run_id": run_id,
        "status": "completed",
        "additional_steps_per_arm": 128,
        "starting_state_equal": True,
        "checkpoint_hashes": {arm: value["sha256"] for arm, value in checkpoints.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_id")
    parser.add_argument("--store", default="artifacts/training")
    args = parser.parse_args()
    print(json.dumps(smoke(args.prepared_id, TrainingStore(args.store)), indent=2))
