# Robustness validation and uncertainty

Use the Phase 3.1 `train validate` workflow for scientific comparisons. The older
Phase 3 `baseline`/`improve` commands remain available as historical pipeline
demonstrations; their shared nominal scenarios, potentially infeasible stress
sets and single-run classifications do not establish independent robustness.
See the [audit](PHASE_3_1_AUDIT.md) and [frozen protocol](EXPERIMENT_PROTOCOL.md).

Physical outcomes are constraint violation frequency, maximum/mean violation
magnitude, full-horizon completion, and time to first violation. Tracking and
reward are separate performance metrics. All-failed execution yields null
physical metrics and an inconclusive comparison, never zero physical failures.
Tracking is averaged within each episode and then across scenarios so long
episodes do not receive more weight. Per-episode results retain durations,
tracking extremes, reward components, and execution errors.

Normal continuation is the control for additional learning. Known-stress changes
are descriptive because those scenarios enter the curriculum. Primary endpoints
are held-out physical violation frequency and maximum violation magnitude.
Nominal guardrails allow at most 10% worse mean tracking and no increase in
violation frequency relative to normal continuation, checked for every seed.

Independent training seeds are the inference unit. Paired differences are
`failure-informed - normal-only` for each seed on identical final scenarios.
At least three distinct seeds are required for the 5,000-resample paired percentile
bootstrap interval. Fewer seeds have no confidence interval and always remain
preliminary/inconclusive. Repeated deterministic scenario seeds are not treated
as independent learned policies.

An `improved` label requires the upper 95% interval below zero for at least one
held-out physical endpoint, no increase on either endpoint for any seed, and all
nominal guardrails passing. `regressed` requires failed nominal guardrails or a
clearly worse held-out endpoint without conflicting improvement. Other cases are
`inconclusive`. Three seeds still provide weak uncertainty coverage; these
intervals are exploratory, without a confirmatory significance claim or adjustment
for multiple endpoints. They describe training variation over a fixed scenario
set, not uncertainty over all possible plants.

In the measured Phase 3.1 pilot, both competence and feasible-discovery gates
failed. There were zero matched continuation pairs. Nominal-final, known-stress
and held-out robustness metrics and confidence intervals are **not available**.
The result is inconclusive, not improvement and not evidence of a regression
caused by failure-informed training.
