# Phase 3.1 baseline audit

Baseline: e224181, clean main, 46 tests passing (three upstream SWIG warnings).

The equation is Euler mass balance, checked against constraints **before** level
clipping. Demand is constant, and the seed has no effect on pump dynamics.
The three-value observation omits efficiency, capacity, leakage, dt and bounds.
Across varied plants it is partially observed; changing seeds on the same physical
configuration does not create independent physical test cases. Six calibrated
values address plant uncertainty but still require fixed dt/bounds for a Markov
transition contract; finite-horizon remaining time is not observed.

Phase 3 validation uses the training pool itself. Its two held-out stress cases
have maximum inflows 0.44/0.48 versus demands 0.90/0.82 before leakage. They
cannot establish improvable controller weakness. Selection trusts discovery
replay labels and does not inspect supporting experiments. Curriculum resamples a
small prefabricated list, so its actual stress proportion can differ from 30%.
Evaluation swallows execution errors and reports zero-valued metrics when no
episode succeeds. Continuation metadata can claim requested hyperparameters even
though SB3 loads the baseline's configuration. Budgets round up to rollouts;
actual steps, wall limits, partitions and per-episode evidence were not recorded.
Prior artifacts and Phase 3 report are retained as historical smoke evidence.

Reward: -tracking_error² -0.01*command² -10*violation_magnitude. The actuator cost
can discourage pumping when tracking is near target; the failure penalty occurs
only at termination. Short failed episodes can have a less-negative cumulative
reward than a successful full episode. Longer training must be judged by physical
violations, completed horizons and tracking, not by return alone. No reward or
physics change is introduced in the observation comparison.

Phase 3.1 will predeclare competence (zero nominal validation violations/errors,
full horizons and mean absolute tracking error <=0.10), use different physical
parameters across partitions, and block continuation if competence or eligible
feasible discoveries are absent. These are study criteria, not safety standards.
