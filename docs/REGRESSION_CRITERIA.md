# Regression Criteria

The failure-informed continuation is classified against the matched normal continuation, not against the rule-based controller and not against training reward alone.

The default smoke/quick policy is:

- nominal mean tracking-error degradation must be at most 10%;
- held-out violation frequency must not increase;
- held-out maximum violation magnitude must not increase;
- known-stress maximum violation magnitude must improve for an `improved` result.

Classification is explicit:

- `improved`: all guardrails pass and known-stress magnitude improves;
- `regressed`: one or more guardrails fail;
- `inconclusive`: guardrails pass but known-stress improvement is not demonstrated.

These are initial configurable experiment policy thresholds, not universal safety standards. Small smoke runs should normally be interpreted as `inconclusive`.

## Frozen Phase 3.3 policy

The matched study has its own preregistered classification and does not
reinterpret historical smoke results. For each baseline seed, robust minus
normal nominal tracking error must be at most 10% of the normal value; nominal
and held-out violation frequencies, maximum held-out violation magnitude and
execution failures must not increase. A per-seed `improved` additionally needs
a strict decrease in held-out frequency or maximum magnitude. The overall
`improved` label requires all three seeds to strictly reduce held-out violation
frequency and pass every guardrail. Complete guardrail failure is `regressed`;
mixed or incomplete evidence is `inconclusive`. The paired-seed bootstrap is
exploratory at n=3. See [the frozen protocol](MATCHED_ROBUSTNESS_STUDY.md) and
[analysis](ROBUSTNESS_STATISTICAL_ANALYSIS.md).

## Frozen Phase 3.4 workflow outcome

`no_eligible_curriculum` is a development workflow status, not an `improved`,
`regressed` or `inconclusive` controller verdict. It requires stopping before
confirmatory training. If a candidate had qualified, the frozen final rule
would have applied a 10% nominal tracking guardrail, no additional nominal or
stress violations or execution failures, paired training-seed comparison, and
the prespecified held-out tracking fallback at a zero-violation floor. In the
observed screen, no candidate qualified, so no Phase 3.4 final regression
classification was computed. See [nominal preservation](NOMINAL_PRESERVATION.md).
