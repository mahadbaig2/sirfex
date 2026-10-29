# SIRFEX Phase 3.7 — Checkpoint 1 implementation report

**Status:** Checkpoint 1 software infrastructure only. Scientific protocol remains draft; no comparative study is declared complete.

## A. Implementation summary

The pump robustness package adds typed signed margins over recorded `EpisodeResult` traces; controller-independent disturbance grids with feasibility analysis, complete outcome vectors and restricted ceiling refinement; same-scenario differential Classes A/B with replay verification; and versioned deterministic reports. CLI orchestration allows only explicitly requested, small software calibration. Existing scenario, runner, controller factory, feasibility, replay, search-space and leakage interfaces are reused. The historical `EpisodeResult` trace has pre-transition observations and post-transition physical info; the latter includes a pre-clipping violation magnitude. PPO is loaded through the existing hashed checkpoint adapter only when selected in calibration configuration.

## B. Files changed

| Purpose | Files |
| --- | --- |
| Metrics, search, evidence | `src/sirfex/robustness/__init__.py`, `identity.py`, `margins.py`, `disturbance.py`, `differential.py`, `report.py`, `store.py`, `service.py` |
| CLI and examples | `src/sirfex/cli.py`, `configs/robustness/pump_calibration.json`, `configs/robustness/pump_differential_calibration.json`, `README.md` |
| Verification | `tests/unit/test_robustness_margins.py`, `test_robustness_disturbance.py`, `test_robustness_differential.py`, `test_robustness_store_report.py` |
| Documentation | `docs/PHASE_3_7_DRAFT_3_IMPLEMENTATION.md`, `ROBUSTNESS_MARGIN_SPECIFICATION.md`, `DISTURBANCE_TOLERANCE_PROTOCOL.md`, `DIFFERENTIAL_CONTROLLER_TESTING.md`, `ROBUSTNESS_REPRODUCIBILITY.md`, this report |

Publication correction (Checkpoint 1.1): the Checkpoint 1 implementation was subsequently committed and pushed as [`8e1c6298ab3899436580a925fa2cc7b1c4b0880b`](https://github.com/mahadbaig2/sirfex/commit/8e1c6298ab3899436580a925fa2cc7b1c4b0880b). The tests and calibration below ran locally; this does not imply GitHub independently reran them. Calibration outputs remained ignored local artifacts. The [Checkpoint 1.1 report](PHASE_3_7_CHECKPOINT_1_1_REPORT.md) documents later hardening and supersedes the original evidence-integrity limitations.

## C. Mathematical verification

For valid pump limits `lo < hi`, `m_t=min(h_t-lo,hi-h_t)` in normalized tank level fraction. A violating transition uses `-violation_magnitude`, measured before the simulator clips its displayed state. `M0` uses the physical scenario initial state, `M_post=min_{t>=1}m_t`, `M_min=min_{t>=0}m_t`, and `delta_M=M_post-M0`. Synthetic tests exercise positive, zero and negative values; both boundaries; clipped crossings; initial minimum and improvement; float32 observation rounding; invalid and partial traces; and deterministic repetition. One locally available Phase 3.6 **development** trajectory was measured without modifying its source record. Differential comparisons use `M_post`; Class B requires a violating X, surviving Y, viable physical scenario and matching deterministic repeats. Disturbance tests cover transformations, zero and undefined axes, censored bounds, full grid, nonmonotone controller/feasibility sequences, descriptive ratios, restricted refinement and the feasibility/simulator discrepancy stop.

## D. Scientific boundary verification

No controller training, reward/physics/observation/action change, neural model, research-agent edit, historical evidence edit, final-holdout access, or reinterpretation of Phases 3.3–3.6 occurred. The two example anchors/search proposals are synthetic; a Phase 3.2 baseline checkpoint was read and hash verified for inference only. This work supports no new scientific performance claim. The draft protocol explicitly states it is not frozen.

## E. Test results

Baseline before editing: `.venv/Scripts/python.exe -m pytest -q` gave **134 passed**, four warnings. After editing, the same command gave **181 passed, four warnings in 150.38 s**. The warnings concern SWIG types in LunarLander and the existing SB3 action-space recommendation. Targeted new checks gave **17 margin**, **17 disturbance**, and **13 differential/store/report** passes. `.venv/Scripts/python.exe -m ruff check .` returned `All checks passed!`; `git diff --check` passed. CLI tolerance, differential, margins, report regeneration and PPO-inference calibration were also run manually with bounded development/synthetic input.

## F. CPU calibration

All calibration used synthetic 40-transition pump scenarios and small grids. Measurements include file writes and process setup within the CLI invocation; per-controller numbers time only episode evaluation. They are engineering measurements, not scientific findings.

| Calibration | Measured episodes / transitions | Measured wall time | Episode rate | Observed RSS | Local summary case bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rule + analytical, 1 anchor × 4 directions × 3 grid points | 26 / 1,040 | 0.468 s (latest uncontended run); other observed runs 0.199–1.560 s | 55.5/s latest | 40.6 MB end | 18,267 |
| Rule + analytical differential, two proposals | 4 / 160 | 0.059 s | 68.1/s | 40.5 MB | no per-case grid files |
| Rule + analytical + verified Phase 3.2 PPO, 1 anchor × 2 directions × 3 grid points | 21 / 840 | 29.001 s, cold PPO import/load included | 0.724/s | 289.9 MB end | 15,663 |

In the PPO run, controller-only measured episode times were 0.538 s / 7 episodes (PPO), 0.135 s / 7 (analytical), and 0.039 s / 7 (rule). A simple 82,000-episode multiplication yields about **1,477 s (24.6 min)** using the latest non-PPO wall rate and **113,243 s (31.5 h)** using the PPO cold-start wall rate. These are **estimates only**, neither a measured full-study cost nor a valid prediction for a different mix, 200-step horizons, load reuse, hardware contention, storage policy or full search. The 29 s cold-start penalty makes the second extrapolation especially misleading. The calibration JSON retains raw timings for review.

## G. Reproducibility

The protocol and each completed controller/anchor case carry canonical semantic SHA-256 identity. Checkpoint paths are local locators; PPO identity uses its verified SHA-256. Atomic case and status writes preserve completed work after interruption. A forced interruption test resumes to the same ordered semantic-result hash as an uninterrupted run; a changed protocol, duplicate case or tampered case is rejected. An actual calibration resume reused two cases and retained semantic result hash `63bf14ad…` and report hash `587b13e4…`; the original measured calibration timing was preserved. JSON is authoritative, Markdown is deterministic and regenerated only after report-hash verification. Seeds, source hashes, Git HEAD, controller and scenario identities and replay statuses are recorded where applicable. Class B uses existing episode replay comparison; no Class B is claimed without matching repeats.

## H. Known limitations

Checkpoint 1.1 clarification: the original Class B repeat check was in-memory and did not itself archive full paired trajectories or independently regenerate them. The original store also did not enforce completed-case immutability at its lowest write method. Both limitations are addressed and verified separately in the Checkpoint 1.1 report; they were not established by the original 181-test run.

- Only the simplified pump simulator has a signed-margin and disturbance implementation; no claim covers other environments or real hardware.
- `delta_first` and infeasibility thresholds are first **observed grid** values; censoring, nonmonotonic outcomes and unknown feasibility remain explicit. Bisection refines only a restricted physical feasibility bracket, never controller failure.
- The calibration search is small, seeded and nonadaptive; an empty differential class does not establish controller parity.
- A saved complete `EpisodeResult` trace is required. Missing/conflicting evidence raises a typed error; a partial nonviolating horizon cannot establish survival.
- The CLI enforces software-calibration bounds. It does not run the full proposed 16-direction, 17-controller, 25-point scientific design.
- Throughput estimates use short synthetic episodes; PPO startup and memory are measured once, and wall time depends strongly on hardware and process contention.
- Local calibration artifacts are ignored by Git and are not a preregistered, published study. Checkpoint 2 needs a separate approved protocol and evidence policy.

## I. Checkpoint 2 readiness

Separate approval is needed for maximum disturbance, grid resolution, differential near-miss threshold, full versus reduced controller set, wall-time cap, development search domain and seed namespace, exclusion policy, exact frozen checkpoint list, and protocol freeze/commit requirements. Full scientific execution and any publication remain outside Checkpoint 1.

## J. Final status

`checkpoint_1_complete`
