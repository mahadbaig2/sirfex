# PPO action diagnostics

The Phase 3.2 inspection establishes a concrete path to the zero physical
commands. It does not claim that the policy's raw action was zero.

```text
Gaussian policy mean/stddev → sampled action (training/stochastic inference)
                         or mean action (deterministic inference)
→ SB3 Box clipping → versioned policy coordinates → physical command
→ original pump validation → inflow → recorded experiment action
```

The implementation uses SB3's `policy.get_distribution`, inspecting its Normal
mean and standard deviation only inside `training/policy_diagnostics.py`.
This access is version-sensitive and records installed library versions.
Deterministic predictions are checked against clipping of the inspected raw
means. Newly drawn raw samples are labeled as new inference on saved models;
historical trajectories that recorded only physical commands are not backfilled.

## Historical checkpoint measurements

On 12 fixed nominal validation initial observations, with diagnostic seed 32 and
32 stochastic samples per observation:

| Saved checkpoint | Raw mean range | Standard deviation | Deterministic commands at zero | Sampled commands at zero |
|---|---|---:|---:|---:|
| Minimal, 10,240 | -2.270 to -2.073 | .9623 | 100% | 98.4375% |
| Minimal, 25,600 | -3.083 to -2.851 | .8920 | 100% | 100% |
| Calibrated, 10,240 | -3.242 to -3.139 | .9444 | 100% | 100% |
| Calibrated, 25,600 | -5.045 to -4.992 | .9251 | 100% | 100% |

Six of the 384 samples from the earliest minimal checkpoint were positive; the
other sampled batches contained no positive commands. This finite sample does
not prove the Gaussian has zero probability of positive actions. The final two
policies also failed all 12 recorded stochastic episodes under seed 32; their
representative stochastic replays matched exactly.

SB3's negative Gaussian means are clipped to zero before the adapter sees them.
Physical inflow measurements agree with the mapped command. There is no evidence
of an adapter conversion or replay bug causing zero output. The cause of learning
such means involves reward, exploration and optimization; raw diagnostics alone
cannot uniquely identify it.

## Versioned action coordinates

`pump-direct:1` retains policy and physical bounds `[0,1]`.
`pump-normalized:1` uses policy bounds `[-1,1]` and maps the bounded policy output
to physical `(a+1)/2`. This makes an initial zero Gaussian mean correspond to
half pump command. Both versions preserve the physical command domain and use
the unchanged pump transition function. Invalid/nonfinite adapter actions are
rejected; SB3 performs its documented Gaussian clipping before the adapter.

Checkpoint metadata and model ZIPs record the policy action schema. Loading
checks model-space bounds; continuation checks action, reward and termination
contracts. Old checkpoints default to direct coordinates. They are never silently
treated as normalized models. Reward/schema metadata tampering is rejected for
new model contracts.

Diagnostics report physical lower/upper/interior fractions, mean and standard
deviation, raw out-of-range fraction, and clipping magnitude. Clipping magnitudes
are in policy-coordinate units; they should not be numerically compared across
coordinate schemas without accounting for the mapping.

The normalized-only pilot still failed every nominal case at 10,240 and 25,600
steps: all 169 recorded actions at each checkpoint were full pump command.
Normalization alone moved boundary saturation; it did not establish competence.
See the [competence study](PPO_COMPETENCE_STUDY.md) for the controlled reward result.

For the three selected competent checkpoints, a separate fixed-initial-observation
inspection found deterministic physical lower/upper saturation of 0/0, 0/0 and
1/12/0 for seeds 3, 17 and 41. With 32 stochastic samples per observation,
raw out-of-range fractions were .2943, .3073 and .3229; sampled saturation can
coexist with mostly interior *deterministic trajectory* actions. These are
different populations: the raw distribution batch samples only initial states,
whereas validation trajectories cover 2,400 time steps per policy. The latter
had lower-bound fractions 0, 0 and .000417. Selected raw diagnostics are in
`selected_policy_diagnostics.json`; selected physical traces are in
`policy_trace_diagnostics.json` under the local study and summarized in the
[public evidence](evidence/phase_3_2_remediation.json).

```powershell
sirfex train diagnose-policy --config configs/training/action_diagnostics.json
sirfex train diagnose-policy --config configs/training/action_diagnostics.json --checkpoint pump-calibrated_3_25600
sirfex train report <diagnostic-run-id>
```

The config's source run must exist locally. Inference mode and seed are recorded;
stochastic PPO replay uses a private CPU Torch RNG state, leaving training RNG
state unchanged. Cross-platform bitwise reproducibility is not promised.

Sources: installed SB3 2.9.0 `common/policies.py`, `common/on_policy_algorithm.py`,
and the [official SB3 normalization guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).
