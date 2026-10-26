# Paired robustness statistical analysis

The unit of replication is a baseline PPO training seed, not an episode. The
three frozen baselines yield three matched pairs: seed 3, seed 17 and seed 41.
For each physical metric and seed, the paired difference is robust continuation
minus normal continuation; negative frequency, violation magnitude and tracking
differences favor the failure-informed arm. Per-seed summaries retain the
individual scenario outcomes and the original baseline policy for context.

The predeclared primary outcome is held-out feasible-stress violation frequency.
Known-stress gains cannot alone justify an overall robustness improvement claim.
The per-seed guardrails cap nominal tracking degradation at 10% of the matched
normal arm, allow no nominal or held-out violation-frequency increase, allow no
maximum held-out magnitude increase and allow no execution-failure increase.
An overall `improved` classification requires all three seeds to reduce the
primary frequency and pass guardrails. Complete guardrail failures are
`regressed`; mixed or incomplete evidence is `inconclusive`.

Descriptive analysis reports the mean, median, range and improvement/unchanged/
worsening count of the three paired differences. The repository's existing
paired training-seed percentile bootstrap (5,000 draws, fixed seed 3101) is
reported as **exploratory**. With n=3, its interval is too unstable to support
a strong population-level or industrial-safety claim; it must not be treated
as an episode-level confidence interval. A 60-scenario physical set helps
describe where controllers differ, but does not create 60 independent trained
controllers.

## Observed paired result

For the primary held-out violation-frequency outcome, robust minus normal was
`[0/60, 0/60, 0/60]` across seeds 3, 17 and 41. Mean, median and range are all
zero; the direction count is 0 improved, 3 unchanged, 0 worsened. The fixed
5,000-draw percentile bootstrap interval is [0,0] for both frequency and
maximum-magnitude differences because all three observed differences are
exactly zero. This interval is an exact description of the input differences,
not evidence that an unseen population has zero effect. The endpoint is at a
floor in both arms, and n=3 cannot resolve heterogeneity in another stress
family. A single event would change the 60-case frequency by 1/60.

The nominal tracking errors (normal → robust) were .03313 → .11128,
.03038 → .04276 and .02556 → .05276. Relative degradation was 235.93%,
40.76% and 106.43%, respectively, over the fixed 10% guardrail. All three
pairs therefore classify `regressed`, as does the overall study. Known-set
tracking improved on all seeds, but both arms already reached zero known-set
violations and known cases were part of the robust training curriculum.

See the [study report](PHASE_3_3_REPORT.md) and the
[paired machine-readable results](evidence/phase_3_3_results.json).
