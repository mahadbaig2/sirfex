# SIRFEX Phase 3.7 Checkpoint 1.1 — hardening report

## A. Repository state

Initial branch: `main`; initial HEAD: `8e1c6298ab3899436580a925fa2cc7b1c4b0880b`, matching the reviewed Checkpoint 1 implementation. The working tree was clean and no newer commit was present. Checkpoint 1.1 tests and calibration ran locally. The completed changes were then committed and pushed to `main` following the user's explicit follow-up instruction. No unrelated local edits were found; the push does not imply that GitHub independently reran local tests.

## B. Audit findings

| Severity | Finding | Location | Correction and verification |
| --- | --- | --- | --- |
| Critical | Lower-level case writes could replace completed evidence | `robustness/store.py` | Storage-enforced write once, identity checks, explicit recovery and writer locks; conflict/corruption/interruption tests |
| Critical | Caller-supplied prepared grid could substitute physical or feasibility metadata | `robustness/disturbance.py` | Bound versioned grid, exact expected transformation and analyzer validation, protocol hash; deliberate-tamper tests |
| Critical | In-memory differential repeats were labeled confirmed Class B without durable independently replayed evidence | `robustness/differential.py`, `service.py` | Candidate stays unverified until four archived episodes and independent regeneration match; archive and mismatch tests |
| High | Generic report replay status obscured evidence stage | `robustness/report.py`, `service.py` | Per-finding in-memory, persisted and regenerated statuses and artifact IDs; report tests |
| High | Resume-only counts could change a semantic report hash, and preserved first timing could be mislabeled as current invocation | `robustness/service.py` | Keep reusable scientific counts separate from invocation metrics; resume tests compare report hashes and zero new work |
| Medium | Committed Checkpoint 1 report said changes remained uncommitted | Checkpoint 1 report and reproducibility guide | Publication correction with pushed commit ID; original local test claims preserved |
| Medium | Earlier 40-step cold-start extrapolation overstated full-study precision | Checkpoint 1 report | Separate 200-step warm calibration and qualified planning range |

Existing `EpisodeResult` traces provide pre-step observations and post-step simulator info, including authoritative pre-clipping violations. Existing controller factories reconstruct rule, analytical and hash-verified PPO policies; `ExperimentStore` and `replay_experiment` support durable local trajectories. Existing search spaces, duplicate checks and exact finite-horizon feasibility analysis remain in use. The numerical margin/physics contracts were reviewed and left unchanged.

## C. Implementation corrections

`RobustnessStore.write_case` now independently recomputes case identity from protocol, controller and anchor; checks declared membership where present; allows semantically identical retries; rejects conflicting completed evidence and corrupt existing files; and requires `recovery=True` to replace failed, invalid or pending work. Exclusive filesystem locks reject simultaneous protocol, run and case writers. Temporary-file atomic replacement prevents valid-looking partial JSON after interrupted writes. A process crash can leave a lock, requiring manual operator recovery after checking that no writer remains.

`PreparedDisturbanceGrid` binds the exact anchor, direction, ordered grid, transformed scenarios, scenario IDs, environment/analyzer versions, feasibility values and optional protocol SHA-256. Evaluation compares caller input against expected preparation before executing a controller. Exact scenario-keyed analyzer caching avoids repeated physical analysis across controllers. Rehashing fabricated metadata does not make it acceptable.

Differential sampling now labels an in-memory repeat as `class_b_unverified`. Selected ranked candidate findings are archived as four existing-format experiments: X original/repeat and Y original/repeat. The archive checks scenario, controller, protocol and episode hashes, validates margins and physical viability, compares persisted repeats, then regenerates all four through existing replay infrastructure. Only a matching candidate is promoted to Class B. Class A remains a safe near miss with its own durable verification status. The JSON report carries artifact IDs and per-finding statuses; Markdown deterministically renders them. Resume-only reuse counts are excluded from the semantic report; the current invocation records zero new episodes while retaining the first measured calibration separately.

## D. Mathematical verification

The original normalized-level margin definition remains `m_t=min(h_t-lo,hi-h_t)` for safe recorded states and `-violation_magnitude` for a physical crossing before clipping. `M0` uses the scenario's physical initial state; `M_post` uses transitions `t>=1`; `M_min` includes `t=0`. Earliest ties are deterministic. Malformed or partial nonviolating traces cannot establish full-horizon survival. Existing synthetic lower/upper clipped-crossing tests and a locally available Phase 3.6 development trace passed again.

All 16 directions, bounded transforms, undefined zero-demand/leak axis, parameter censoring, ordered outcomes, nonmonotonicity flags, and descriptive first observed grid thresholds remain intact. A safe complete episode at proven finite-horizon infeasibility still stops with diagnostic evidence. Restricted feasibility bisection remains limited to adverse directions with a viable/infeasible bracket and `0<=dt*leak<=1`; it does not estimate a global controller threshold.

## E. Evidence integrity

Case IDs include protocol, controller and anchor identities; changing checkpoint SHA-256, physical scenario or protocol changes identity. Completed-case retry is idempotent and a rejected overwrite preserves its original semantic hash. Failed work requires explicit recovery. Locked writers reject concurrency on the same run/case; no distributed lock or protection from deliberate filesystem modification is claimed. Archived experiments contain scenario, seed, environment, controller config/checkpoint metadata, full original and repeat trajectories, margins, violation indicators, result hashes, replay statuses and stable IDs. An incomplete `ExperimentStore` archive is refused until manually investigated. The regeneration procedure is `sirfex experiments --store <run>/experiments replay <experiment-id>`.

A bounded synthetic CLI example using one proposal and reference controllers produced one verified Class B with four archived experiment IDs. The generated JSON report hash was `50c9d2674d8fbb19295e0a51125428a8b1e50702bcace357d2cd912e8daba8e0`; regenerating its Markdown through `sirfex robustness report` produced identical file bytes. This is an implementation check on a deliberately simple synthetic scenario, not a scientific controller result.

## F. Testing

At the start of this turn, `.venv/Scripts/python.exe -m pytest -q` independently reported **181 passed, four warnings in 41.45 s**. An intermediate full run after core hardening reported **205 passed, four warnings in 37.78 s**. The final complete suite reported **207 passed, four warnings in 35.35 s**. After expanding one import-isolation test to cover all three numerical modules, its targeted rerun passed. The four warnings remain the three LunarLander SWIG deprecation warnings and the existing Stable-Baselines3 action-space recommendation. Targeted tests cover write once/idempotence, corruption, recovery and locks; 12 prepared-grid tamper classes and analyzer cache reuse; four persisted episodes, exact/tolerance/mismatch replay, downgrade, missing or corrupt evidence, interrupted archive and Class A separation. Resume tests cover unchanged semantic report hashes and truthful zero-new-episode timing. `.venv/Scripts/python.exe -m ruff check .`, `ruff format --check`, `python -m json.tool` on the machine draft, and `git diff --check` passed. Tiny tolerance and differential CLI calibrations completed on synthetic input. No test changed historical evidence.

## G. CPU performance

The bounded local calibration used one synthetic pump scenario at horizon 200, three warm measured episodes per controller, one verified Phase 3.2 seed-3 PPO checkpoint and reused controller instances. Both calibration passes completed all three 200-transition episodes per controller. Approximate measured ranges across the two passes:

| Component | Measured result |
| --- | ---: |
| Python process startup | 0.114–0.135 s per fresh minimal process |
| Torch import | 5.50–5.78 s |
| Stable-Baselines3 import after Torch | 0.187–0.200 s |
| SHA-256 checkpoint verification | 0.0018–0.038 s |
| PPO model/controller initialization | 3.90–4.52 s |
| PPO warm episode with trace, 200 transitions | 0.193–0.254 s; 1.06–1.17 ms/transition averaged by pass |
| Analytical warm 200-transition episode | 0.014–0.023 s; 0.086–0.100 ms/transition averaged |
| Rule warm 200-transition episode | 0.008–0.013 s; 0.052–0.055 ms/transition averaged |
| Isolated deterministic PPO action | 0.712–0.764 ms/action over 100 calls |
| Four directions × five grid-point preparation | 0.145 s (latest pass) |
| One 200-transition full experiment archive | 0.028–0.035 s; 74,825 bytes |
| One small JSON report generation + Markdown rendering | below 0.001 s each |

The worst-case proposed grid is `17*12*16*25 = 81,600` episodes, at most 16.32 million transitions: 14.4 million PPO and 1.92 million reference transitions if all reach horizon 200. Multiplying observed warm transition costs gives a rough **3.9–5.2 hours** for grid episode execution, depending on observed PPO timing. Full trajectory persistence for every episode would add roughly **0.6–0.8 hours and about 6.1 GB** at this one-case size, which motivates compact grid summaries and selected full archives. One-time imports/model loads, feasibility preparation and report overhead are smaller in this tiny sample; differential proposals, headline replay, filesystem contention and checkpoint variation are not yet measured at study scale. A provisional **4–6 hour grid-only planning range** is an engineering estimate, not a benchmark or promise. PPO inference is the likely bottleneck. The earlier 24.6-minute and 31.5-hour straight extrapolations from 40-step/cold-start calibration should not be used to budget the study. Raw measurements remain local in `artifacts/robustness/checkpoint1_1_cost_calibration.json`.

## H. Documentation

Updated the Checkpoint 1 report, margin specification, disturbance protocol, differential testing guide and reproducibility guide. Added this report, `docs/PHASE_3_7_CHECKPOINT_2_PROTOCOL_DRAFT.md`, the non-executable `configs/robustness/phase37_checkpoint2_protocol.draft.json`, and a bounded reproducible calibration script. The original test/calibration claims remain identified as local measurements; GitHub publication is not represented as independent CI verification.

Complete changed-file inventory: `README.md`; `docs/DIFFERENTIAL_CONTROLLER_TESTING.md`, `DISTURBANCE_TOLERANCE_PROTOCOL.md`, `PHASE_3_7_CHECKPOINT_1_REPORT.md`, `ROBUSTNESS_MARGIN_SPECIFICATION.md`, `ROBUSTNESS_REPRODUCIBILITY.md`, plus the two new reports named above; `configs/robustness/phase37_checkpoint2_protocol.draft.json`; `scripts/calibrate_robustness.py`; `src/sirfex/robustness/differential.py`, `disturbance.py`, `evidence.py`, `report.py`, `service.py`, `store.py`; and `tests/unit/test_robustness_differential.py`, `test_robustness_disturbance.py`, `test_robustness_evidence.py`, `test_robustness_service.py`, `test_robustness_store_report.py`.

## I. Scientific boundaries

No controller training, physics/reward/action/observation edit, Groq/LangGraph expansion, full controller sweep, full differential search, final-holdout scenario/content/outcome access, or historical study rewrite occurred. Only synthetic and existing development-only inputs were used. No new scientific robustness result is claimed.

## J. Checkpoint 2 proposal

The draft lists three selected Phase 3.2 starting policies, six Phase 3.5 **screening** policies, six Phase 3.3 historical secondary policies, and rule/analytical references. The 15 PPO files are locally present with verified SHA-256; only Phase 3.2 seed 3 was load-tested in this checkpoint. Phase 3.5's status remains `no_eligible_curriculum`, so these are screening comparisons, not an approved successful continuation. Proposed grid is `0..0.60` in `0.025` steps with censoring; Class A gap is provisionally `0.05` normalized level units. Proposed seeded feasibility-filtered differential search is 200 candidates per pair with five archived findings per class. The draft proposes eight hours wall time, 30 minutes per PPO controller and two GB artifact cap, all subject to review. It is explicitly non-executable.

## K. Remaining risks

Before freeze: produce a safe development-only anchor identity manifest without touching sealed partitions; load-test every intended checkpoint; confirm grid/domain compatibility, pairing, seed namespace and duplicate exclusions; establish differential-search and storage overhead at a bounded pilot scale; choose final controller set and budgets. Local file hashes detect accidental corruption but are not authenticated signatures against a malicious operator. A stale writer lock or interrupted experiment archive requires manual inspection. The simplified pump model cannot establish real-world safety.

## L. Final status

`checkpoint_1_1_complete`
