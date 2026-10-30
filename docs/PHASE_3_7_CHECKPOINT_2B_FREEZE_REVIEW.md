# SIRFEX Phase 3.7 Checkpoint 2B: protocol freeze review

**Decision:** `checkpoint_2b_awaiting_explicit_approval`. This is a development-only, descriptive study proposal. No real controller cohort evaluation, training, sealed-final access, protocol freeze, or approval record occurred in Stage 1.

## A. Git and CI state

Stage 1 started from clean branch `phase-3.7-checkpoint-2a-completion` at `177dc67cc28e33dcced0cc146ce2eb413e670b97`, also the remote branch HEAD. PR #1 was already merged into `main` at `8e2d968a691b9ddfb9bf837ed1f9829585e9bede`; no scientific implementation commits followed Checkpoint 2A.1 before this review. The post-merge [main workflow 40972369659](https://github.com/mahadbaig2/sirfex/actions/runs/40972369659) passed both `test` and `training`. The new Stage 1 PR, final pushed commit and CI run are recorded in the final publication note below.

Local verification on the starting tree: `python -m pytest -q` **243 passed, four dependency warnings**; `ruff check .`, `ruff format --check src tests scripts` (148 files), and `git diff --check` passed. The supported scientific dry-run passed with 17 controllers, 12 anchors, seven research pairs and an 88,168-episode upper bound. A guarded `scientific-execute` attempt with the actual candidate was rejected with `Draft/candidate protocol cannot execute` before any rollout. All 15 PPO checkpoint files were freshly rehashed, metadata matched source runs, loaded through the pump factory, satisfied action/observation contracts, and returned equal finite actions on repeated deterministic calls. No PPO training was invoked.

## B. Exact review identities and proposed frozen bytes

| Input | Identity |
|---|---|
| Candidate protocol semantic SHA-256 | `aaef7614397fadd37f96be31afad1bdde240742528348330acfd15b76b338258` |
| Candidate protocol raw-file SHA-256 | `155cf2872a3288298ba4e7c410e4494dda05a3142da8244aac5204e62de2e532` |
| Candidate controller inventory semantic SHA-256 | `0fdfe5cc1c23c5f6d5592e5fc8be7e53e2f8d6117633798c262344a08336c0ea` |
| Candidate budget semantic SHA-256 | `3552a209f9fd091723118476fca381dc154b84c539b937b0ceb39ff9ec5df71a` |
| Development manifest semantic SHA-256 | `1bc850397b4f608b0cf7c75e63660b3bdfa52923d043b0b876f8720d763f88b0` |
| Development manifest raw-file SHA-256 | `fc58c67f2767912fe58140ad5c9bc73ca7c3b6b6a39537e525cc33b0d5f2de07` |
| Differential search-space raw-file SHA-256 | `d99869ff60bc7025c2eccb545816743d58da79c172e1833621a7f65d7cfe31f6` |
| Candidate deterministic work-plan semantic SHA-256 | `1ee6babcd78ac5e51c09ebfb054c8f6e03f14f7b2ee35dadf1308f51b5c1acef` |

The **proposed future frozen bytes** are defined exactly as follows, without creating an executable file now: load the current candidate JSON as UTF-8; replace only `status` with `FROZEN — AUTHORIZED FOR EXECUTION` and `executable` with JSON `true`; recompute `semantic_sha256` using `sirfex.robustness.identity.semantic_hash` on that object; serialize `json.dumps(value, ensure_ascii=False, indent=2) + '\n'` as UTF-8. Preserve the current candidate schema and every other field. This deterministic 14,909-byte proposal has semantic SHA-256 `51b181eab1a86c3f42bd27f5221b260dad3e4d0d13b9558473b9b8a35741dec4` and raw-byte SHA-256 `269918979a7ab70eed9dd354fbeacae8d7851bb3ceee923ea26a0bad70b38b8b`. These are **proposal identifiers, not approval digests**. The proposed frozen file does not exist. Its work-plan hash must be computed and approved separately because the protocol hash changes when status changes. If the reviewer changes any field or byte convention, recompute and seek approval of the actual resulting bytes.

## C. Fixed controller and scenario inventory

The source-run IDs are `20261025T205339_64b622c146` (Phase 3.2 start), `20261027T143749_d88d2dd37f` (Phase 3.5 screen), and `20261026T222151_846c8296a2` (Phase 3.3 secondary). Every PPO row below uses the `ppo` evaluation adapter with deterministic inference; the source artifact is `artifacts/training/<source-run>/checkpoints/<ID>.zip`. Seed and historical group are fixed by the candidate and source metadata. The Phase 3.5 rows are screening policies, not evidence of an eligible curriculum.

| Historical group | Seed | Controller ID | Checkpoint SHA-256 |
|---|---:|---|---|
| Phase 3.2 start | 3 | `ppo_3_10240` | `859c9244e83fce5cf31600d9f7fae0038e053fc8b3dc1663218ed616257e70f9` |
| Phase 3.2 start | 17 | `ppo_17_10240` | `05b5fe45c1f46b7eb28699bd4a1dc693e9c11d56fba3376211c5985168a9d333` |
| Phase 3.2 start | 41 | `ppo_41_25600` | `48fe432e373aef9f64e55e0092f9b09ba95ef407bcc6d81f76a1f61432fd83f0` |
| Phase 3.5 screen | 3 | `screen_3_normal` | `5d11abfd79be87e11b9576046fd6e7d279ef8ee824f4559c6eebaa1030d238d4` |
| Phase 3.5 screen | 3 | `screen_3_expanded_5` | `1dd1302b6e91d4227246d7eb151ec34676bf2ff2d041db78b3757f1382217557` |
| Phase 3.5 screen | 3 | `screen_3_expanded_10` | `f5db7f7bf937568d23a927b5ac671d270694ea291c364137b65967585167dbcf` |
| Phase 3.5 screen | 17 | `screen_17_normal` | `6dd798247ad77ee718773185b542a6c91b3a43cd9e665983c512c27d2712a133` |
| Phase 3.5 screen | 17 | `screen_17_expanded_5` | `4fa3335b76a97569b298abb1de2f939e788320181f2eafe58b09d02dcb8cbf65` |
| Phase 3.5 screen | 17 | `screen_17_expanded_10` | `de4c2ea70f38b7bb25e20d1472329ea47ea9a97342b5598f005ff14664aba019` |
| Phase 3.3 secondary | 3 | `seed_3_normal` | `cca14de90a83fa3205efb586e9c34f0b461edf12afb8cfa90e74d05659f75f27` |
| Phase 3.3 secondary | 3 | `seed_3_robust` | `4b2dddc8fd2aa76b49fdffa40d3c7a6848d5a49b0a6b2949d82b53e484a0b2d1` |
| Phase 3.3 secondary | 17 | `seed_17_normal` | `e0889617537e2278872303626f176cfad42dc4fbc4f1b6907d7c0764f6e89f54` |
| Phase 3.3 secondary | 17 | `seed_17_robust` | `21f9ae78d4176d3a2817b22d641009a6a3204734c5d5a407b9b20ca59500477c` |
| Phase 3.3 secondary | 41 | `seed_41_normal` | `d46f64ac798d83ff894eec784e3995e99d7ff80e795f860b2ca0eb4c07230fa8` |
| Phase 3.3 secondary | 41 | `seed_41_robust` | `a04b792f77686627775bf5d421b7e4b2b92b70ed66c7e1675f7460644bd3c54f` |

The two non-checkpoint references are `pump-rule` and `pump-analytical`, loaded by their own pump factories. Their fixed controller IDs and adapters are part of the candidate inventory.

The unchanged development manifest contains 12 training exclusions, 12 nominal validation anchors, 24 screening-stress cases and seven public historical discoveries: **55** unique declared cases. Its provenance states generator-prefix extraction cross-checked with Phase 3.5 development hashes; bundled final records were unopened. Its declared leakage check passed with no issues, absolute physical-parameter tolerance `1e-4` except horizon, and disjoint seeds. `validate_manifest` checked scope, partition counts, scenario identities, provenance, anchor order and manifest semantic hash. Source hashes and each scenario ID remain in the manifest. No sealed-final content was loaded or used for overlap checking.

## D. Independent scientific audit

The current implementations in `margins.py`, `disturbance.py`, `differential.py`, `evidence.py`, `study.py`, `study_execute.py`, `study_store.py`, `study_report.py` and `scientific.py` were inspected against the Stage 1 invariants. Controller pool sessions verify checkpoint SHA before loading, use deterministic `act`, reset the environment per episode and retain cumulative timing when sessions close. They neither train nor mutate weights. The signed margin consumes simulator trace evidence, including pre-clipping violation magnitude; a nonviolating partial trace does not establish survival.

The finite-horizon analyzer supplies independent viability status. The report separates proven viable (`target_feasible` or `bounded_operation_feasible`), infeasible and undetermined grid cases; only proven viable failures count as avoidable. Prepared grids are shared and hash-bound across controllers. The 16 direction combinations and 25-point ordered grid are fixed. Parameter transformations do not clamp; invalid/undefined/bound-censored points remain distinguishable. `delta_first` is the first observed failure in grid order, nonmonotonicity is preserved, and restricted physical-ceiling refinement is disabled. The preflight covered 4,800 controller-independent grid points, including 1,598 infeasible configurations.

All seven X/Y research comparisons are fixed. Each pair receives 200 seeded, nonadaptive proposals; exclusions consume slots. Class A safe margin gaps and Class B avoidable failure candidates remain separate. Class B requires deterministic repeat and independently replayed persisted evidence before a selected headline is accepted. Selection order is deterministic. Reports are derived from verified stored cases and replay artifacts. Complete status requires every mandatory work item, planned paired scenario coverage, no missing valid grid evidence and verified selected headlines; otherwise status is `partial_incomplete`. Markdown regeneration checks report semantic identity. No unresolved correctness defect was found in this focused review.

## E. Prespecified measurement and interpretation

The four **primary** directed pairs are `screen_3_expanded_5 → screen_3_normal`, `screen_3_expanded_10 → screen_3_normal`, `screen_17_expanded_5 → screen_17_normal`, and `screen_17_expanded_10 → screen_17_normal`. The three **secondary** historical pairs are `seed_3_robust → seed_3_normal`, `seed_17_robust → seed_17_normal`, and `seed_41_robust → seed_41_normal`. Direction is always X minus Y. Six starting-PPO versus each classical reference comparisons are descriptive contexts, not training-effect estimates. The protocol gives their exact IDs and order.

Primary endpoints: binary modeled physical violation **stratified by proven viable, infeasible and unknown physical status**; `M_post`, the minimum signed margin over recorded post-initial transitions; `delta_first`, the first observed failure on the finite 0–0.600 grid; and explicit threshold censoring. Secondary endpoints: nominal and development-stress tracking MAE, full-episode minimum signed margin, signed change from initial margin, Class A and verified Class B discovery counts, replay verification outcomes, physical-infeasibility threshold, whether failure precedes that threshold, and nonmonotone failure sequences. Raw fixed-scenario rows, grid thresholds and differential findings retain the evidence behind these summaries; any derived count must be computed from those verified rows with coverage stated. No post-result threshold or hypothesis change is allowed.

Report per-scenario paired X-minus-Y differences and the full observed distributions, including minimum, median and 10th-percentile signed margins, coverage, exclusions and censoring. Keep right-censored, parameter-bound-censored, invalid, undefined and unknown-feasibility results separate. Do not set missing thresholds to zero, impute absent cases, or infer rankings from incomplete pairs. Report tied, adverse and null effects. Three seeds and correlated checkpoints do not justify independent-sample p-values; simulator transitions are not independent experimental replicates. The simplified simulator and finite grid give neither global minima nor real-world safety certification.

## F. Resource decision and human boundary

The proposed limits remain **eight cumulative wall hours, 30 evaluation minutes per PPO, and 2 GiB of local artifacts**. Maximum planned work is 81,600 grid episodes, 408 fixed stress episodes, up to 6,160 differential/repeat/archive episodes, and 88,168 episodes overall. The prior warm PPO grid projection is 13,896–18,288 seconds (3.86–5.08 hours); warm differential estimate is 1,189–1,565 seconds, before cold loading, physical preparation, durable replay, persistence and reporting. Cold loads were observed at 58.5–67.8 seconds. The compact-case example projected about 26.3 MB for 3,264 cells, but trajectory and memory examples are not worst-case bounds. The per-PPO cap may truncate the plan, and sequential work can leave paired arms asymmetrically covered. **Recommendation:** retain these exact caps for review and explicitly accept possible incomplete results; a guaranteed-completion requirement instead needs bounded non-scientific calibration and separate approval of revised limits. No cap changed here.

The [execution plan](PHASE_3_7_CHECKPOINT_2B_EXECUTION_PLAN.md) supplies exact CLI syntax, freeze commit, independent approval record, output layout, stop/resume and retention rules. The [risk register](PHASE_3_7_CHECKPOINT_2B_RISK_REGISTER.md) classifies execution blockers, interpretation limits and operational uncertainty. Stage 2 requires your explicit approval of the proposed frozen bytes and resource decision. Until then the candidate remains `executable: false`, no approval digest or record has been issued, and no real scientific run is authorized.

## Final publication note

To be completed after the Stage 1 documentation commit, push, draft PR creation and latest branch CI conclusion. This note reports repository state; it cannot grant scientific authorization.
