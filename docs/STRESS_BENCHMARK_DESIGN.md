# Phase 3.4 independent stress benchmark

The Phase 3.3 final set is historical development evidence. Phase 3.4 generates
new pump configurations with fixed random seed `3401`, before it trains or
evaluates any Phase 3.4 continuation. Its primary physical endpoint is
constraint-violation frequency; mean absolute level tracking error is the
prespecified co-primary outcome if both arms reach zero violations.

The controller-independent sampler uses dt `.1`, horizon `200`, safe level
`[.05,.95]`, target `.5`, effective capacity uniformly `.75–1.30`, efficiency
uniformly `.70–1.0`, and leakage uniformly `0–.10`. Demand is determined by a
sampled ratio of target-required inflow to effective capacity. Initial level
is sampled independently. These are numerical model conditions, not a claim
about a validated industrial operating domain.

| Declared stratum | Target-required inflow / effective capacity | Initial level | Development cases | Final cases |
|---|---:|---:|---:|---:|
| Nominal-like | .53–.70 | .30–.70 | 6 | 16 |
| Moderate | .72–.86 | .30–.70 | 6 | 16 |
| High | .88–.99 | .30–.70 | 6 | 16 |
| Near-boundary | .94–1.06 | .08–.20 or .80–.92 | 6 | 16 |

Twelve additional nominal-like configurations form a separate final nominal
holdout. Screening uses only the 24 development stress cases and the untouched
Phase 3.2 validation partition. The 64 final stress cases and 12 final nominal
cases cannot be used for discovery, curriculum selection or checkpoint tuning.
The fixed sampling ranges define difficulty strata without reading any policy
outcome. Acceptance requires the existing pump validator and exact-rational
finite-horizon viability analyzer. Infeasible, invalid and near-duplicate
proposals are counted. A target equilibrium is not itself a viability proof;
some safe finite-horizon cases need not sustain the target indefinitely.

Every accepted case records target pumping authority margin, demand-to-effective
capacity ratio, initial distance to a safe bound, corrective inflow margin,
initial tracking error, leakage, horizon and the full feasibility result.
The existing physical-key check excludes configurations within absolute
`1e-4` on all physical parameters and excludes shared seeds across old training,
validation, discoveries, opened Phase 3.3 sets and new development/final sets.
A seed change alone does not make deterministic physics independent.

The new final benchmark is frozen by scenario records and SHA-256 hashes in the
Phase 3.4 protocol before screening. Designated historical baseline and rule/
analytical references may characterize development stress; no continued policy
sees the new final set until a candidate is selected and all matched final
checkpoints are saved and verified. Reference performance is diagnostic; a
valid constructive safe trace can establish feasibility even if either
reference controller fails.

The generated benchmark accepted 24 development stress, 64 sealed final stress
and 12 sealed final nominal cases. The designated original seed-3 policy
violated 6/24 development stress cases (one high, five near-boundary); both
references had zero development violations. The final policy outcomes remain
unmeasured because [curriculum screening](CURRICULUM_SCREENING.md) found no
eligible candidate. Proposal, feasibility and difficulty distributions are
reported in the [Phase 3.4 report](PHASE_3_4_REPORT.md).
