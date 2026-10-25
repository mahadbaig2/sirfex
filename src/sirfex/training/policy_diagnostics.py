"""Isolated, version-sensitive SB3 Gaussian inspection; never invented trace data."""

import numpy as np

from sirfex.environments.pump_observations import PumpObservationEnvironment
from sirfex.training.controller import PPOController
from sirfex.training.ppo import dependency_versions
from sirfex.training.protocol import scenario_record


def action_statistics(raw, bounded, physical, low=0.0, high=1.0):
    raw, bounded, physical = (
        np.asarray(x, dtype=float).reshape(-1) for x in (raw, bounded, physical)
    )
    if not len(raw) or not (len(raw) == len(bounded) == len(physical)):
        raise ValueError("Action diagnostics require equal nonempty arrays")
    if not all(np.isfinite(x).all() for x in (raw, bounded, physical)):
        raise ValueError("Action diagnostics require finite values")
    return {
        "count": len(raw),
        "lower_bound_fraction": float(np.mean(physical == 0)),
        "upper_bound_fraction": float(np.mean(physical == 1)),
        "interior_fraction": float(np.mean((physical > 0) & (physical < 1))),
        "mean_applied_action": float(physical.mean()),
        "std_applied_action": float(physical.std()),
        "raw_outside_fraction": float(np.mean((raw < low) | (raw > high))),
        "mean_clipping_magnitude": float(np.mean(np.abs(raw - bounded))),
    }


def diagnose_policy(metadata, scenarios, *, seed=32, samples=32):
    import torch

    if not 1 <= samples <= 1000 or not 1 <= len(scenarios) <= 100:
        raise ValueError("Diagnostic batch or sample budget exceeded")
    controller = PPOController.load(metadata.path, metadata)
    model = controller._model  # isolated inspection, not a training dependency in the engine
    if model.policy.squash_output:
        raise ValueError("This diagnostic supports unsquashed Gaussian PPO only")
    env = PumpObservationEnvironment(metadata.observation_schema)
    observations = np.stack([env.reset(s)[0] for s in scenarios])
    tensor, _ = model.policy.obs_to_tensor(observations)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        distribution = model.policy.get_distribution(tensor)
        mean = distribution.distribution.mean.cpu().numpy()
        std = distribution.distribution.stddev.cpu().numpy()
        deterministic = distribution.get_actions(deterministic=True).cpu().numpy()
        sampled = np.stack([distribution.get_actions().cpu().numpy() for _ in range(samples)])
        predicted, _ = model.predict(observations, deterministic=True)
    low, high = float(model.action_space.low[0]), float(model.action_space.high[0])
    bounded = np.clip(deterministic, low, high)
    if not np.array_equal(bounded, predicted):
        raise ValueError("SB3 predict disagrees with the inspected Gaussian clipping path")
    from sirfex.training.actions import to_physical

    schema = metadata.policy_action_schema
    physical = np.stack([to_physical(x, schema) for x in bounded])
    sampled_bounded = np.clip(sampled, low, high)
    sampled_physical = np.stack([to_physical(x, schema) for x in sampled_bounded.reshape(-1, 1)])
    rows = []
    for i, scenario in enumerate(scenarios):
        env.reset(scenario)
        _, _, _, _, info = env.step(physical[i])
        rows.append(
            {
                **scenario_record(scenario),
                "observation": observations[i].tolist(),
                "policy_mean": mean[i].tolist(),
                "policy_stddev": std[i].tolist(),
                "raw_deterministic": deterministic[i].tolist(),
                "raw_samples": sampled[:, i, :].tolist(),
                "postprocessed_deterministic": predicted[i].tolist(),
                "physical_command": physical[i].tolist(),
                "applied_inflow": info["inflow"],
            }
        )
    env.close()
    return {
        "schema_version": "ppo-actions:1",
        "checkpoint": metadata.to_dict(),
        "dependencies": dependency_versions(),
        "diagnostic_seed": seed,
        "diagnostic_scope": "New inference on saved model; not historical rollout raw actions",
        "interface_note": "SB3 policy.get_distribution Normal mean/stddev is version-sensitive",
        "rows": rows,
        "deterministic": action_statistics(deterministic, bounded, physical, low, high),
        "stochastic": action_statistics(sampled, sampled_bounded, sampled_physical, low, high),
    }
