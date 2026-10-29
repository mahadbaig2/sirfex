# Phase 3.7 GitHub CI audit

## Workflow and triggers

The repository has one workflow, [CI](../.github/workflows/ci.yml), on Ubuntu with Python 3.12. It runs on pushes to `main` and on pull requests. Pushes to `phase-3.7-checkpoint-2a-completion` do not directly trigger CI; its open [draft PR #1](https://github.com/mahadbaig2/sirfex/pull/1) targeting `main` triggers CI after each push. No manual dispatch is configured. The `test` job installs editable `.[dev,simulation,search,agents]`, checks Ruff format and lint, then runs all collected tests. The `training` job installs CPU Torch and `.[dev,training]` and runs the optional PPO tests. Ordinary tests require no GitHub secrets. A green PR run does not change the historical status of `main` until code is merged and a new `main` run executes.

## Observed runs and causes

| Run | Ref and result | Evidence |
| --- | --- | --- |
| [40891587732](https://github.com/mahadbaig2/sirfex/actions/runs/40891587732) | `main` at `0b244a0`: failed `test`, passed `training` | Ruff rejected the string formatting in `scripts/export_phase36_smoke.py`; lint and test were not reached. |
| [40901665888](https://github.com/mahadbaig2/sirfex/actions/runs/40901665888) | PR at `80a917a`: failed `test`, passed `training` | Formatting and lint passed; test collection lacked `pydantic` because CI omitted the declared `agents` extra. |
| [40901968701](https://github.com/mahadbaig2/sirfex/actions/runs/40901968701) | PR at `1508eff`: failed `test`, passed `training` | Test collection passed. Four tests assumed ignored local PPO checkpoint files; one real PPO test ran in the core job without Stable-Baselines3. |
| [40902410991](https://github.com/mahadbaig2/sirfex/actions/runs/40902410991) | PR at `04660f2`: **passed `test` and `training`** | GitHub completed both jobs on the working branch. |
| [40953966709](https://github.com/mahadbaig2/sirfex/actions/runs/40953966709) | PR at `b646e12`: **passed both jobs** | CI audit documentation push triggered the PR workflow. |
| [40954416235](https://github.com/mahadbaig2/sirfex/actions/runs/40954416235) | PR at `61dd066`: **passed both jobs** | Physical-feasibility report correction. |
| [40955501083](https://github.com/mahadbaig2/sirfex/actions/runs/40955501083) | PR at `98d85eb`: **passed both jobs** | Unclean invocation accounting and budget tests. |
| [40955969350](https://github.com/mahadbaig2/sirfex/actions/runs/40955969350) | PR at `2b3bfcf`: **passed both jobs** | Separate authorization record and gate tests. |
| [40971519690](https://github.com/mahadbaig2/sirfex/actions/runs/40971519690) | PR at `4671b79`: **passed both jobs** | Final report, ledger and synthetic-pilot corrections. |

The formatting defect was fixed in `80a917a`. `1508eff` installed the existing agent extra in the core test job. `04660f2` made plan-structure tests independent of ignored checkpoint binaries, retained the checkpoint-dependent dry-run test for environments with installed artifacts, and routed the optional real PPO smoke test to the CPU training job. Scientific execution still verifies checkpoint bytes unconditionally before running; CI does not fabricate them or weaken that gate.

## Local reproduction and limits

From the repository root, with the appropriate extras installed:

```powershell
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
```

The dedicated CI job runs the optional PPO subset on Ubuntu with CPU Torch. Local Windows testing is useful but does not replace this Linux result. The checkpoint-dependent real-candidate dry run requires the ignored 15 PPO binaries; CI records a skip when they are absent. This is an environmental limitation, not evidence that checkpoint bytes were verified on GitHub. The prior local preflight artifact and a local dry run with those binaries provide separate evidence. Future contributions should run format and lint before pushing, keep numerical tests independent of optional ML/network dependencies, and use the draft PR checks to validate changes on the implementation branch. Do not put sealed final partitions or checkpoint binaries in CI.
