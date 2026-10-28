# Phase 3.7 Draft 3 implementation copy

**DRAFT — Checkpoint 1 implementation only; scientific protocol not frozen.**

This implementation-facing copy summarizes the user-supplied Phase 3.7 Checkpoint 1 brief. It does not select final study checkpoints, development domains, thresholds, or budgets. No final-holdout contents may be opened during this checkpoint.

The proposed measurement system has four parts: signed pump constraint margins, bounded disturbance-to-failure grids, same-scenario differential controller comparisons, and deterministic JSON/Markdown evidence. The signed transition margin is `min(h_t-lo, hi-h_t)` for safe transitions and the negative of the simulator's pre-clipping `violation_magnitude` for violations. Report `M0`, full-episode `M_min`, post-initial `M_post`, and `M_post-M0` separately. Differential near-miss comparisons use `M_post`; actual avoidable failures require X violation, Y survival, finite-horizon physical viability, and replayed evidence.

The 16 candidate disturbance directions comprise all 15 nonempty subsets of increased demand, reduced capacity, reduced efficiency and increased leak, plus a separate reduced-demand surplus direction. The provisional grid is `delta = 0.025*k` for `k=0..24`; Checkpoint 1 runs only tiny synthetic subsets. Full grid outcome vectors and parameter-domain censoring must be recorded. The feasibility analyzer's exact reachability result is authoritative for this simplified pump; a controller surviving a proven-infeasible scenario halts the affected run. Feasibility refinement is optional and restricted to justified adverse domains with `0 <= dt*leak(delta) <= 1` throughout the bracket.

The proposed later differential search uses a prospectively defined, feasibility-filtered development search space, seeded random proposals, 200 proposals per controller pair, and top five findings per class. Those numbers are **not authorized for Checkpoint 1 execution**. Checkpoint 1 allows only synthetic tests and tiny calibration, with no PPO training, full 17-controller sweep, final partition access, or scientific outcome claim.

Before Checkpoint 2, the user must approve a separately frozen scientific protocol: maximum disturbance, grid resolution, near-miss threshold, controller set, wall-time cap, development search domain and seed namespace, exclusion policy, protocol hash and commit point. Historical Phase 3.3–3.6 outcomes are unchanged.
