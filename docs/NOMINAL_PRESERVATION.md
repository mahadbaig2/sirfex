# Nominal preservation and Phase 3.4 classification

Phase 3.3 observed higher nominal MAE in all three 30%-stress continuations.
Its saved evidence also shows altered command distributions and improved
known-discovery tracking. Those associations cannot isolate stress ratio as a
cause because the small pool, repeated exposures and stress severity changed
with it. Phase 3.4 lowers stress probability under matched development budgets,
uses the same pool when comparing ratios, and optionally tests an expanded pool
at fixed ratio. It keeps the versioned observation, action, reward, termination
and physical constraints unchanged.

Development selection requires nominal MAE within 10% of equal-budget normal
training, no extra nominal physical violation, and no extra execution error.
The new final nominal holdout is not used for this choice. The confirmatory
comparison applies the same nominal guardrails per independent baseline seed.
This is an empirical selection and regression rule; no constrained PPO optimizer
or formal control guarantee is introduced.

The final stress primary endpoint is physical violation frequency. If both
matched arms have zero stress violations for all seeds, the prespecified
continuous fallback is held-out mean absolute tracking error. `improved`
requires every seed to satisfy nominal and safety guardrails, every seed to
strictly improve the active held-out endpoint, and the exploratory paired-seed
bootstrap upper 95% bound to be below zero. Any complete nominal or safety
guardrail failure is `regressed`; incomplete or mixed evidence is
`inconclusive`. Known-stress gains alone never satisfy the improvement rule.
Three independent seeds support descriptive paired inference only; scenario
count does not enlarge the number of independently trained policies.

The actual Phase 3.4 screen found no curriculum that met both development
stress-gain and nominal-preservation conditions. It stopped before final
training. The final classification rule above was frozen but **not applied**;
there is no Phase 3.4 controller regression verdict. See the
[measured report](PHASE_3_4_REPORT.md).
