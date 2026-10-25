"""Feasibility-filtered Phase 2 search with replay-verified control witnesses."""

import time

from sirfex.core import Scenario
from sirfex.experiments.replay import replay_experiment
from sirfex.experiments.service import run_and_save_experiment
from sirfex.search.objectives import PumpConstraintObjective
from sirfex.search.runner import SearchRunner
from sirfex.search.space import SearchSpace
from sirfex.training.checkpoints import write_json
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.protocol import competence, leakage_check, physical_key


def validate_avoidable_evidence(experiments, failing_id, reference_id):
    """No label is awarded from an equilibrium test or unverified metadata alone."""
    replay = []
    try:
        failing, reference = experiments.load(failing_id), experiments.load(reference_id)
        if physical_key(failing.scenario) != physical_key(reference.scenario):
            raise ValueError("Failure and reference physical scenarios differ")
        if failing.scenario.seed != reference.scenario.seed:
            raise ValueError("Failure and reference seeds differ")
        analysis = analyze_feasibility(failing.scenario)
        if analysis["finite_horizon_viable"] is not True:
            raise ValueError("Finite-horizon viability is not established")
        if failing.result.get("execution_status") != "completed" or not any(
            f["category"] == "constraint_violation" for f in failing.result.get("failures", [])
        ):
            raise ValueError("Tested controller has no observed physical failure")
        if (
            reference.result.get("execution_status") != "completed"
            or reference.result["episode_length"] != analysis["horizon"]
            or reference.result["terminated"]
            or reference.result.get("failures")
            or any(t["info"]["constraint_violation"] for t in reference.trajectory)
        ):
            raise ValueError("Reference did not complete the full horizon without violation")
        replay = [replay_experiment(experiments, i).to_dict() for i in (failing_id, reference_id)]
        if any(r["status"] not in ("exact_match", "tolerance_match") for r in replay):
            raise ValueError("Failure or reference replay is unverified or mismatched")
        commands = [r["action"][0] for r in reference.trajectory]
        if len(commands) != analysis["horizon"] or not all(0 <= u <= 1 for u in commands):
            raise ValueError("Reference trajectory is not an admissible finite-horizon witness")
        return {
            "avoidable": True,
            "evidence_strength": ["reference-demonstrated", "feasibility-proven"],
            "feasibility": analysis,
            "failure_event": failing.result["failures"],
            "reference_controller": reference.manifest.controller_metadata,
            "failure_experiment_id": failing_id,
            "reference_experiment_id": reference_id,
            "replay": replay,
            "witness": {
                "type": "replayed physical control sequence",
                "commands": commands,
                "horizon": len(commands),
                "reference_experiment_id": reference_id,
            },
        }
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        return {
            "avoidable": False,
            "reason": str(exc),
            "replay": replay,
            "failure_experiment_id": failing_id,
            "reference_experiment_id": reference_id,
        }


def run_avoidable(config, experiment_store, search_store, training_store, *, baseline=None):
    started = time.perf_counter()
    trials, wall = config.get("trials", 24), config.get("wall_seconds", 120)
    if type(trials) is not int or not 1 <= trials <= 100 or not 0 < wall <= 300:
        raise ValueError("Avoidable search requires 1..100 trials and <=300 finite seconds")
    deadline = started + wall
    space = SearchSpace.from_dict(config)
    if space.environment_id != "pump" or config.get("objective") != "pump.constraint_violation":
        raise ValueError("Avoidability requires the unchanged pump physical violation objective")
    # A hard horizon bound keeps a single analytical check or evaluation bounded.
    horizon = space.fixed_parameters.get("horizon")
    if type(horizon) is not int or not 1 <= horizon <= 1000:
        raise ValueError("Avoidable search must declare a fixed horizon in 1..1000")
    controller_id = config.get("controller_id", "ppo")
    controller_config = config.get("controller_config", {})
    excluded = []
    competence_verification = None
    competence_evaluations = 0
    if controller_id == "ppo":
        from sirfex.training.controller import PPOController
        from sirfex.training.evaluation import evaluate_controller
        from sirfex.training.remediation_diagnostics import metadata_from_dict

        if not baseline:
            raise ValueError("PPO avoidable search requires a completed competent --baseline study")
        study = training_store.load(baseline)
        seed = str(config.get("baseline_seed", 3))
        meta = metadata_from_dict(study.get("baseline_selection", {}).get("selected", {})[seed])
        if study["manifest"].get("status") != "completed":
            raise ValueError("Baseline study is incomplete")
        groups = study["dataset_partitions"]["groups"]
        validation = [Scenario.from_dict(row["scenario"]) for row in groups["validation"]]
        model = PPOController.load(meta.path, meta)
        metrics = evaluate_controller(
            lambda: model,
            validation,
            "competence-recheck",
            max_steps=200,
            observation_schema=meta.observation_schema,
            reward_schema=meta.reward_schema,
        ).to_dict()
        competence_verification = competence(metrics)
        competence_evaluations = len(validation)
        if not competence_verification["passed"]:
            raise ValueError("Selected PPO does not currently pass the frozen competence gate")
        excluded = [
            Scenario.from_dict(row["scenario"]) for group in groups.values() for row in group
        ]
        controller_config = {"checkpoint": meta.to_dict()}

    def candidate_policy(scenario):
        analysis = analyze_feasibility(scenario, max_horizon=1000)
        separate = leakage_check({"prior_partitions": excluded, "search": [scenario]})["passed"]
        return {
            "accepted": analysis["finite_horizon_viable"] is True and separate,
            "feasibility": analysis,
            "disjoint_from_prior_partitions": separate,
        }

    runner = SearchRunner(
        space,
        PumpConstraintObjective(),
        controller_id,
        experiment_store=experiment_store,
        search_store=search_store,
        max_steps=horizon,
        controller_config=controller_config,
        candidate_policy=candidate_policy,
    )
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Deadline exhausted during competence recheck")
    found = runner.run(
        config.get("strategy", "random"),
        trials=trials,
        strategy_seed=config.get("seed", 32),
        scenario_seeds=[config.get("scenario_seed", 320000)],
        wall_seconds=remaining,
    )
    records, references = [], 0
    search_id = found["manifest"]["search_id"]
    report = {
        "schema_version": "avoidable-failures:1",
        "search_id": search_id,
        "baseline": baseline,
        "competence_verification": competence_verification,
        "records": records,
        "status": "running",
        "configuration": config,
    }
    path = search_store._directory(search_id) / "avoidable.json"
    write_json(path, report)
    for discovery in found["discoveries"][:8]:
        if time.perf_counter() >= deadline:
            break
        scenario = Scenario.from_dict(discovery["scenario"])
        for reference_id in ("pump-analytical", "pump-rule"):
            if time.perf_counter() >= deadline:
                break
            references += 1
            try:
                reference = run_and_save_experiment(
                    experiment_store, scenario, reference_id, max_steps=horizon
                )
            except (ValueError, RuntimeError) as exc:
                records.append(
                    {
                        "discovery_id": discovery["discovery_id"],
                        "avoidable": False,
                        "reference_id": reference_id,
                        "reason": f"Reference execution failed: {exc}",
                        "experiment_id": getattr(exc, "experiment_id", None),
                    }
                )
                write_json(path, report)
                continue
            proof = validate_avoidable_evidence(
                experiment_store, discovery["experiment_ids"][0], reference.experiment_id
            )
            records.append(
                {
                    "discovery_id": discovery["discovery_id"],
                    "scenario": discovery["scenario"],
                    "controller_id": controller_id,
                    "controller_config": controller_config,
                    "search_id": search_id,
                    "reference_id": reference_id,
                    **proof,
                }
            )
            write_json(path, report)
    loaded = search_store.load(search_id)
    qualifying = {r["discovery_id"] for r in records if r["avoidable"]}
    report.update(
        status="completed" if time.perf_counter() < deadline else "interrupted",
        summary={
            **found["summary"],
            "feasible_candidates": sum(
                t.get("candidate_analysis", {}).get("feasibility", {}).get("finite_horizon_viable")
                is True
                for t in loaded["trials"]
            ),
            "observed_controller_violations": sum(
                any(r["is_failure"] for r in t["risk_scores"]) for t in loaded["trials"]
            ),
            "reference_evaluations": references,
            "competence_recheck_evaluations": competence_evaluations,
            "search_replay_executions": sum(
                len(d["experiment_ids"])
                for d in found["discoveries"]
                if d["replay_status"] != "unverified"
            ),
            "witness_replay_executions": sum(len(r.get("replay", [])) for r in records),
            "avoidable_discoveries": len(qualifying),
            "replay_verified_avoidable_discoveries": len(qualifying),
            "reference_success_discoveries": len(qualifying),
            "wall_seconds_including_references": time.perf_counter() - started,
            "reference_candidate_limit": 8,
            "note": "Search evaluations, reference evaluations and replay executions are separate",
        },
    )
    write_json(path, report)
    return {"search_id": search_id, "status": report["status"], "summary": report["summary"]}
