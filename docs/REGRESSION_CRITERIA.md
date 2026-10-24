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
