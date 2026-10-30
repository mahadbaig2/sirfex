# Phase 3.7 Checkpoint 2B: risk register

| Risk and evidence | Class | Decision or control |
|---|---|---|
| No human approval of exact frozen bytes, limits and cohort; candidate `executable: false` | Execution-blocking | Keep gate closed until explicit review, frozen commit and separate authorization record. |
| Any changed checkpoint, development manifest, search space or frozen work plan | Execution-blocking | Re-run hashes and dry-run after freeze; stop on mismatch. No silent substitution. |
| Missing, failed or invalid mandatory cases, unmatched paired coverage, or unverified Class B durable replay | Execution-blocking for complete report | Preserve partial evidence and mark `partial_incomplete`; inspect before any recovery. |
| 8-hour cumulative wall cap and 30-minute per-PPO cap may cut off the sequential workload | Operational uncertainty | Retain proposed caps for approval with explicit acceptance of incomplete outcomes. Warm PPO grid projection is 3.86–5.08 hours; differential warm estimate is 0.33–0.43 hours, excluding load, physics, replay, archival and reporting overhead. A larger cap needs bounded evidence and a new approval. |
| 2-GiB cap and free disk space | Operational uncertainty | A prior compact-case example projected about 26.3 MB for 3,264 cases and one trajectory example was about 0.7 MB; neither bounds worst-case archives. Check free space and preserve headroom. |
| Native model memory and cold loads | Operational uncertainty | Cold model load observation was 58.5–67.8 seconds. Python-traced synthetic pilot memory excludes native allocations. Watch the execution host; do not infer a full-cohort memory bound. |
| Crash accounting may charge downtime to active controllers | Operational uncertainty | Inspect the durable ledger before resume. The conservative charge is deliberately an upper bound, not precise CPU time. |
| Only three training seeds; Phase 3.5 checkpoints share training history, and Phase 3.3 is a separate context | Scientific-interpretation limitation | Report per-scenario matched effects and distributions; no independent-sample p-values, pooled dose-response or independence claims. Historical Phase 3.5 decision remains `no_eligible_curriculum`. |
| Pump simulator, bounded grid/search and development-only scenarios | Scientific-interpretation limitation | Report observed failure thresholds as finite-grid observations, distinguish physical infeasibility, and make no real-world safety or global-minimum claim. |
| Censored or unknown feasibility outcomes and nonmonotone failures | Scientific-interpretation limitation | Preserve raw statuses and coverage; no zero imputation or artificial ordering of thresholds. |
| Optional PPO throughput calibration or broader simulator validation | Optional future improvement | If guaranteed completion is required, run a separately bounded, explicitly non-scientific calibration and seek new limit approval before freeze. |

**Recommendation:** retain the currently proposed 8-hour, 30-minute and 2-GiB caps for human review, with acceptance that the run may remain incomplete. No budget is changed in Stage 1.
