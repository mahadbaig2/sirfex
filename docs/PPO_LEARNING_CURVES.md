# PPO learning curves: Phase 3.1 pilot

Measured on 2026-10-25 Asia/Karachi (2026-10-24 UTC), run
`20261024T225829_4cc56983a8`, implementation commit `981a902`.
The [public JSON evidence](evidence/phase_3_1_pilot.json) contains configuration,
physical scenario partitions, per-episode metrics, checkpoints' hashes, discovery
classification, and subsequent policy-trace diagnostics. Binary checkpoints
remain local under `artifacts/training/<run-id>/checkpoints`.

One independent training seed (3); same 12 nominal validation scenarios for both
schemas; 200-step horizons. The gate was fixed before training: no violations,
no execution failures, all full horizons, and mean absolute tracking error <=.10.
Reward and dynamics were unchanged.

| Observation | Total trained steps | Violations | Mean absolute tracking error | Mean length | Full horizons | Gate |
|---|---:|---:|---:|---:|---:|---|
| Minimal, 3 values | 10,240 | 12/12 | .246537 | 11.75 | 0/12 | Not met |
| Minimal, 3 values | 25,600 | 12/12 | .246537 | 11.75 | 0/12 | Not met |
| Calibrated, 6 values | 10,240 | 12/12 | .246537 | 11.75 | 0/12 | Not met |
| Calibrated, 6 values | 25,600 | 12/12 | .246537 | 11.75 | 0/12 | Not met |
| Rule reference | — | 0/12 | .068919 | 200 | 12/12 | Met |

Every PPO checkpoint had maximum violation magnitude .0422524, mean violation
magnitude .0241457, and worst tracking error .553108. Mean simulated time to
failure was 1.175 seconds. No evaluation execution failed. The rule's worst
tracking error was .234073, but its scenario-weighted mean met the declared gate.

Across the 12 PPO validation episodes, empirical distributions (identical at all
four curve points) were:

| Metric | Minimum | 25th percentile | Median | 75th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Episode mean tracking error | .162758 | .187398 | .238931 | .286573 | .364078 |
| Episode length | 7 | 10 | 12.5 | 13.25 | 15 |
| Maximum violation magnitude | .007652 | .011067 | .025234 | .034067 | .042252 |

These describe the fixed scenario pool; they are not training-seed confidence intervals.

The calibrated schema showed **no measured improvement in this pilot**. This
does not establish that calibration is useless: both learned policies collapsed
to the same observed zero-command behavior, so the comparison does not isolate
observability from optimization and reward design. A feedforward minimal policy
also cannot distinguish equal observations with different capacity/efficiency/leak.

## Reward and policy diagnostics

The original simulation reward is `-(level-target)^2 - .01*command^2`; training
adds `-10*violation_magnitude`. Physical failure remains a separate constraint
event. Per-episode diagnostic sums give these means:

| Controller | Tracking penalty | Actuator penalty | Constraint penalty | Total training-form return |
|---|---:|---:|---:|---:|
| Rule reference | -1.422754 | -1.094167 | 0 | -2.516920 |
| Every measured PPO checkpoint | -.905311 | 0 | -.241457 | -1.146768 |

The failed PPO episodes receive a less-negative undiscounted cumulative return
than the full-horizon rule reference. Episode lengths differ sharply. This is
evidence that raw cumulative return does not measure safe competence; it is
consistent with an early-termination incentive problem. It is **not** a causal
proof of PPO's failure mechanism or a comparison of discounted optimal policies.
Gaussian action clipping, exploration and optimization remain possible factors.
No reward weight was tuned after seeing these results.

Post-pilot inspection persisted 48 validation experiments (12 for each of four
checkpoints) through the ordinary experiment service. Each checkpoint produced
141 actions, all exactly zero after the standard PPO action clipping. A
representative episode from each checkpoint replayed exactly. These trajectories
are visible in the existing Evidence Explorer; no final holdouts were opened.

## CPU cost

Windows 11, Python 3.12.10, NumPy 2.5.4, Gymnasium 1.4.0, SB3 2.9.0 and
Torch 2.14.1; Intel Core i5-8350U (four physical cores, eight logical processors),
CPU device, two Torch threads, one environment/worker.

| Schema | Segment | Training wall seconds | Steps/second | Checkpoint bytes |
|---|---|---:|---:|---:|
| Minimal | 0 → 10,240 | 28.947 | 353.8 | 140,248 |
| Minimal | 10,240 → 25,600 | 30.614 | 501.7 | 140,252 |
| Calibrated | 0 → 10,240 | 17.977 | 569.6 | 144,972 |
| Calibrated | 10,240 → 25,600 | 41.585 | 369.4 | 144,976 |

Total study wall time was 121.923 seconds for 51,200 trained transitions plus
analysis/evaluation, within the 900-second cap. RSS at completion was 289,296,384
bytes (275.9 MiB); process peak working set was 293,244,928 bytes (279.7 MiB).
The recorded artifact-size snapshot was 768,352 bytes before final manifest
writes and the later diagnostics. Startup/import, shared-machine load and small
segments affect throughput; these timings do not rank the schema architectures.

## Selection and reproducibility

No competent baseline was selected. The saved diagnostic checkpoints are:

| Checkpoint | SHA-256 |
|---|---|
| `pump-minimal_3_10240` | `87193a1ac63b2d9c7bf088db6a4cc87e53362b61c892ea52ac3d0c15eee6b4f5` |
| `pump-minimal_3_25600` | `06a36b2907c183fef4851a58d7769f9f89c3be8471d6dd5a19b05edc9e0cda67` |
| `pump-calibrated_3_10240` | `b24daf21ab94f5d4d686c14ddb690ca491de5c070d326e1ec3421218cff18445` |
| `pump-calibrated_3_25600` | `fc9c8d0047bff68eee2b2b8201e503d110dff03c8cfe00bde3f0891d26e6617f` |

Protocol SHA-256:
`403452221c7c414dd06e9ba148edbe178c75a9d852673e742b09979b39bfd79a`.
All four local model hashes were reverified after the pilot. Reproduction across
different dependency versions or machines is best-effort; stochastic training
need not reproduce identical model bytes.

```powershell
sirfex train validate --config configs/training/pump_learning_curve.json
sirfex train report 20261024T225829_4cc56983a8
sirfex train verify 20261024T225829_4cc56983a8
python scripts/inspect_validation_policies.py 20261024T225829_4cc56983a8
python scripts/export_validation_report.py 20261024T225829_4cc56983a8 --output review.json
```

The supplied run ID exists on the development machine. Other users generate a
new ID and need local Phase 2 artifacts to reproduce the discovery audit; the
public evidence export contains the measured definitions and results. No
multi-seed confidence interval or robustness significance claim is supported.
