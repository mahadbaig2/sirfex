# Avoidable failure discovery

Phase 3.2 adds a candidate filter to the existing random/TPE search runner. The
objective remains `pump.constraint_violation`; neither reward nor tracking error
substitutes for physical failure. Search-space bounds are enforced before simulation.
Invalid proposals, valid but rejected candidates, completed evaluations, references
and replays are counted separately.

```powershell
sirfex search avoidable --config configs/search/pump_avoidable.json --baseline <competence-study-id>
sirfex search show <search-id>
```

The baseline must be a completed competence study with a selected checkpoint. Its
hash is verified and its frozen nominal validation performance is rechecked before
search. Selection remains the first passing checkpoint for the configured seed;
search outcomes cannot select a different checkpoint. Candidate physical conditions
and seeds are checked for overlap with the study's frozen partitions. Final holdout
outcomes are never evaluated by search.

The declared experiment uses random search, seed 32, 24 proposals and a 120-second
overall budget, including references. Horizon is 200 at dt=.1. Initial level spans
.15–.85, demand .35–.95, capacity .5–1.4, efficiency .6–1 and leakage 0–.12. Target
is .5 and operating limits are .05/.95. These are model search bounds, not a
validated industrial operating envelope. Infeasible or undetermined finite-horizon
candidates are rejected before simulation. Equilibrium feasibility alone is insufficient.

For up to eight ranked failures, both analytical and hysteresis references run on
the identical scenario and seed. A label requires a completed physical violation
by the tested controller, full-horizon constraint-free reference execution, admissible
commands, finite-horizon viability, and matching replays for both experiment IDs.
Reference errors or replay mismatches remain explicit exclusions.

`reference-demonstrated` identifies the successful reference. Its recorded and
replayed command sequence also supplies a constructive `feasibility-proven` witness
for this deterministic model. The labels can coexist. A viable classification by
itself does not receive either discovery label; the implementation does not claim
unexecuted open-loop witnesses. Each witness includes its commands, horizon,
reference ID, failure event, physical scenario, checkpoint SHA-256 and replay reports.

`avoidable.json` is an additive search artifact, loaded by the existing report
command. It does not mutate historical discoveries or failure categories. Deadline
checks occur between bounded operations; an in-flight simulation/replay may finish
after the deadline, which is reported rather than concealed.

Readiness is `ready` when a completed study replicated competence across at least
three seeds and search has a verified avoidable failure; `partially_ready` when a
competent selected PPO lacks those combined prerequisites; otherwise `blocked`.
This is scientific readiness for designing a matched continuation, not evidence
of robustness improvement or an instruction to run that study. See the measured
[Phase 3.2 report](PHASE_3_2_REPORT.md).
