# SIRFEX — Technical Product Requirements & System Architecture

**Version:** 1.0 · **Status:** Implementation-ready proposal · **Target:** Solo-built, portfolio-grade, open-source release

**Official name:** SIRFEX  
**Expansion:** Simulation-based Integrity, Robustness & Failure Exploration  
**Description:** An open-source, local-first Physical AI testing engine for autonomous controllers.

## 1. Executive definition

SIRFEX is a **local-first, CPU-first, controller-agnostic evaluation and failure-discovery framework** for autonomous controllers operating in simulated physical environments. It searches for reproducible failures under bounded physical disturbances, records evidence, compares controller revisions, and optionally retrains supported reinforcement-learning (RL) policies.

**Not claimed:** certification of real-world safety, universal automated repair, physically validated industrial digital twins, or compatibility with arbitrary simulators/controllers without adapters.

### Primary success claim

> Given a compatible controller and a documented simulation adapter, SIRFEX can execute reproducible stress tests, discover and replay failure-inducing scenarios, and quantify changes in robustness relative to baselines and held-out tests.

### Non-negotiable constraints

- Developer machine: Intel Core i5 8th generation, 16 GB RAM, integrated graphics only.
- No mandatory cloud services, paid APIs, hosted GPUs, or simulation/retraining on owner-funded infrastructure.
- All compute and model execution happen on the end user's machine; public website is static and shows precomputed demonstrations.
- Fully usable through a Python CLI without a browser UI.
- No user accounts or telemetry by default.
- Two v1 environments: LunarLander (Gymnasium/Box2D) and a documented simplified industrial pump-control simulation.
- Environment and controller extension through versioned Python adapters.
- Test first, optional controller-specific improvement second.

## 2. Intended users and jobs

**Primary:** RL researchers, applied ML engineers, robotics/control students, and simulation engineers who need repeatable evaluation of learned or classical controllers.

**Jobs to be done:**
1. Import or implement a compatible controller.
2. Establish normal-condition performance.
3. Define physically bounded disturbances and safety constraints.
4. Discover failures within a fixed evaluation budget.
5. Inspect and reproduce trajectories and failure conditions.
6. Compare controllers fairly on identical test suites.
7. Optionally retrain a supported RL policy and test generalization.

## 3. V1 scope

### In scope (must ship)

- Environment SDK with reset, step, disturbance specification, metric collection, and failure criteria.
- Controller SDK with `reset` and `act`, including PPO/SB3 and rule-based baseline adapters.
- Both initial environments executable headlessly on CPU.
- Deterministic-seed best-effort experiment reproducibility, with versions and artifact hashes recorded.
- Baseline evaluation; uniform random stress search; Optuna/TPE search; equal-budget comparison.
- Trajectory logging, scenario manifest, replay support, failure classification.
- SQLite experiment index and filesystem artifact store.
- Local FastAPI API and responsive browser-based Studio for experiment creation, status, failure exploration, comparisons.
- CLI and reproducible example scripts.
- Public static website with documented precomputed benchmark results and clips.
- Optional PPO retraining using failure-informed scenario sampling; report both gains and regressions.

### In scope (limited/experimental)

- MPC controller through one reference implementation in pump environment, after rule-based and PPO adapters are stable.
- Controller import by Python entry point in a **trusted local environment only**.
- Numerical trajectory replay; pixel-perfect re-render is not guaranteed.

### Out of scope

- Real robot integration; Isaac Sim; CARLA; photorealistic 3D; hosted simulation; managed database; multi-user collaboration; universal controller repair; LLM-based explanations; arbitrary untrusted Python sandboxing; safety certification.

## 4. Functional requirements and acceptance tests

| ID | Requirement | Acceptance |
|---|---|---|
| FR-01 | Register simulation adapter | Both environments pass shared conformance tests |
| FR-02 | Register controller adapter | PPO and rule-based controllers execute through same runner |
| FR-03 | Evaluate baseline | Episode metrics and trajectories persisted with seeds/configs |
| FR-04 | Generate bounded scenarios | All generated scenarios satisfy environment-declared parameter bounds and cross-parameter constraints |
| FR-05 | Detect failures | Explicit safety/termination rules, not reward-only heuristics |
| FR-06 | Reproduce failure | Saved manifest and compatible environment/controller reproduce failure category across repeated runs; deviations recorded |
| FR-07 | Search failures | Uniform random and TPE use equal episode budgets; discovery rates reported |
| FR-08 | Compare controllers | Same environment build, seeds, scenario list, metrics, and evaluation budget |
| FR-09 | Improve PPO | Optional retraining produces versioned checkpoint and held-out comparison; improvement is not guaranteed |
| FR-10 | Run locally | End-to-end workflows run without network after dependency installation |
| FR-11 | Visualize | Studio shows job status, scenario table, trajectory chart/replay, and comparison summary |
| FR-12 | Export | Portable non-executable experiment bundle with manifest and metrics |

## 5. Non-functional requirements

- **Cost:** $0 mandatory recurring owner compute/API expenditure; free hosting subject to provider limits. End-user hardware and electricity are not free.
- **Hardware:** baseline operation on 16 GB RAM / CPU-only; profile before declaring performance targets.
- **Resource limits:** configurable max episodes, steps/episode, wall-clock time, worker count (default 1–2), and training steps.
- **Reliability:** local jobs isolated from UI process; status persisted; interrupted runs marked interrupted; resumability only where explicitly supported.
- **Reproducibility:** log git revision, Python/dependency versions, environment build, controller artifact hash, scenario config, seeds, and hardware summary.
- **Security:** API bound to 127.0.0.1; restrictive CORS; no public execution service; Python controllers are trusted executable code and never described as safely sandboxed.
- **Privacy:** no default telemetry or artifact uploads.
- **Documentation:** environment model assumptions, controller contracts, evaluation methodology, and failure taxonomy.

## 6. Architecture

```text
                         PUBLIC (STATIC)
   Static website / documentation / precomputed experiments
                         (no simulation API)

                       USER'S MACHINE
+-------------------------------------------------------------+
| SIRFEX Studio (Next.js/React UI, locally served)             |
|  Experiment setup | Jobs | Failure explorer | Comparisons    |
+---------------------------+---------------------------------+
                            | localhost HTTP
+---------------------------v---------------------------------+
| FastAPI local service                                       |
|  Validation | Job control | Experiment queries | Export      |
+---------------------------+---------------------------------+
                            | subprocess / job worker
+---------------------------v---------------------------------+
| sirfex-core (Python)                                        |
|  Runner | Scenario engine | Metrics | Search | Comparison    |
|  Controller adapters | Environment adapters | Improvement    |
+----------+------------------+-------------------------------+
           |                  |
+----------v---------+ +------v------------------------------+
| Environments       | | Controllers                          |
| LunarLander/Box2D  | | PPO/SB3 | Rule-based | reference MPC  |
| Pump ODE model     | | trusted Python adapters             |
+--------------------+ +-------------------------------------+
           |
+----------v--------------------------------------------------+
| Local persistence: SQLite + versioned JSON/Parquet/NPZ/files |
+-------------------------------------------------------------+
```

**Design principle:** the core is importable as a Python library and runs from the CLI without FastAPI or the UI. Simulation code never depends on web components.

## 7. Suggested monorepo

```text
sirfex/
  pyproject.toml
  packages/
    sirfex-core/              # runner, protocols, search, metrics, persistence
    sirfex-environments/      # lunarlander, pump, conformance tests
    sirfex-controllers/       # SB3, rules, reference MPC
    sirfex-api/               # localhost FastAPI, job management
  apps/
    studio/                  # React/Next.js local UI
    website/                 # static portfolio showcase
  experiments/               # versioned experiment manifests
  benchmarks/                # fair comparisons and analysis
  examples/                  # runnable controllers, sirfex CLI scripts
  tests/                     # unit, integration, reproducibility tests
  docs/                      # SDK, physics, limitations, installation
```

## 8. Core Python contracts (conceptual, refine before implementation)

```python
from typing import Protocol, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class Scenario:
    environment_id: str
    parameters: dict[str, float | int | bool]
    seed: int
    version: str

@dataclass
class Transition:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

class EnvironmentAdapter(Protocol):
    id: str
    version: str
    def scenario_spec(self) -> dict: ...
    def validate_scenario(self, scenario: Scenario) -> None: ...
    def reset(self, scenario: Scenario) -> Any: ...
    def step(self, action: Any) -> Transition: ...
    def evaluate_failure(self, trace: list[dict]) -> dict: ...
    def close(self) -> None: ...

class ControllerAdapter(Protocol):
    id: str
    def reset(self) -> None: ...
    def act(self, observation: Any, deterministic: bool = True) -> Any: ...

class ScenarioSearch(Protocol):
    def propose(self, history: list[dict], budget_remaining: int) -> Scenario: ...
    def observe(self, scenario: Scenario, result: dict) -> None: ...
```

**Critical detail:** an environment's observation/action specification must be checked against a controller before execution. A controller trained on LunarLander is not interchangeable with a pump controller. Scenario specs must also support conditional/correlated parameters and reject physically invalid combinations.

## 9. Environment A — LunarLander

- Base: Gymnasium LunarLander-v3, discrete action mode for initial PPO and rule-based controllers.
- Record standard state and outcome metrics: landing success, crash/out-of-bounds, elapsed steps, fuel/action proxy, impact speed, cumulative reward, safety violation count.
- Stress axes, contingent on actual Gymnasium API: gravity, wind/turbulence settings, seeded initial conditions; add sensor noise via explicit observation wrapper and engine degradation via tested custom dynamics only.
- **Important:** do not claim engine-thrust degradation or arbitrary runtime gravity changes are supported by stock Gymnasium without implementation and tests.
- Failure severity: explicit crash and unsafe touchdown penalties plus constraint violations; never equate negative reward automatically with unsafe behavior.
- Controller examples: PPO MLP; rule-based heuristic matched to discrete actions.

## 10. Environment B — Simplified pump control

**Model:** a documented single-tank, pump-fed fluid system with demand outflow, sensor observation, and delayed actuator dynamics. Start with a transparent discretized mass-balance model, e.g. `h[t+1] = h[t] + dt * (q_in(u[t], efficiency) - q_out(h[t], demand[t])) / A`. If pressure is included, define the pressure relationship and assumptions explicitly. Avoid representing the simplified model as a real plant.

- State: tank level, inferred/observed flow, target level or pressure, actuator state, optional demand estimate.
- Actions: bounded pump command (continuous for PPO/SAC in later stages, or discretized for PPO initial compatibility).
- Disturbances: demand spikes, efficiency loss, sensor bias/noise, actuator delay, bounded leaks.
- Constraints: low/high tank level, pressure limits where modeled, actuator switching rate, sustained target deviation.
- Baselines: hysteresis/PID-like rule controller, reference MPC when implemented, PPO.
- Metrics: constraint-violation rate and duration, tracking error, control effort, episode completion, cost proxy.
- Tests: conservation/mass balance, units, numerical stability, monotonic relationships, disturbance bounds.

## 11. Scenario and failure taxonomy

Each environment declares `parameter_name`, `unit`, `bounds`, `distribution`, `validity_constraints`, `severity_class`, and `version`.

Classify each candidate as: (a) nominal, (b) plausible degraded, (c) extreme but modeled, or (d) invalid/unrecoverable. Do not count invalid or physically impossible scenarios as useful failure discoveries.

Failure event fields: `failure_type`, `timestamp/step`, `violated_constraint`, `severity`, `scenario_id`, `seed`, `controller_hash`, `evidence_path`, `reproduction_status`.

## 12. Search methodology

**Baseline:** uniform random sampling over *valid* scenarios. **Challenger:** Optuna TPE over the same valid scenario space. **Optional third:** simple boundary sampling as a strong baseline.

- Equal total **episode** budgets, not merely equal optimizer trials.
- Use initial single-seed screening for cheap search, then reevaluate shortlisted scenarios on multiple distinct seeds.
- Optimize a predeclared severity objective incorporating failures and violations; report separate success and safety metrics.
- Track first failure found, distinct reproducible failure modes, failure yield per 100 episodes, worst-case severity, and wall-clock cost.
- Repeat search comparisons over multiple random seeds; show uncertainty. If TPE does not beat random search, report it.
- Separate optimization/discovery scenarios from held-out evaluation distributions.

## 13. Improvement methodology

**Only PPO initially.** Construct a bounded curriculum mixing nominal and failure-focused scenarios; do not train exclusively on worst cases. Version all checkpoints and training configurations.

Evaluate baseline and improved policies on:
1. Nominal test set.
2. Previously discovered failure set (known vulnerabilities).
3. Held-out disturbance set not used in search or training.

Report success rate, safety violations, return, constraint severity, and uncertainty across multiple seeds. Improvement is a hypothesis, not a product guarantee. Retain the baseline policy and reject a new policy as an overall improvement if it substantially degrades required nominal metrics.

## 14. Storage schema

**SQLite entities:**
- `environment_versions(id, environment_id, version, manifest_hash)`
- `controllers(id, kind, environment_id, artifact_hash, metadata_json)`
- `experiments(id, created_at, config_hash, status, environment_version_id, controller_id)`
- `jobs(id, experiment_id, type, status, started_at, ended_at, error)`
- `scenarios(id, environment_version_id, params_json, scenario_hash, validity_class)`
- `episodes(id, experiment_id, scenario_id, seed, metrics_json, outcome, trajectory_path)`
- `failures(id, episode_id, type, severity, evidence_path, reproduced)`
- `comparisons(id, baseline_controller_id, candidate_controller_id, suite_hash, summary_path)`

**Files:** `artifacts/{experiment_id}/manifest.json`, `episodes/*.parquet` or `*.npz`, `policies/*.zip`, `reports/*.json`, `replays/*`. Hash and version each executable/model artifact. Portable exports exclude executable controllers unless the user explicitly includes them.

## 15. Local API (initial endpoints)

- `GET /health`
- `GET /environments`, `GET /environments/{id}/scenario-spec`
- `GET /controllers`, `POST /controllers/register` (trusted local path only)
- `POST /experiments`, `GET /experiments/{id}`
- `POST /experiments/{id}/run`, `POST /jobs/{id}/cancel`, `GET /jobs/{id}`
- `GET /experiments/{id}/episodes`, `GET /experiments/{id}/failures`
- `POST /searches`, `GET /searches/{id}`
- `POST /comparisons`, `GET /comparisons/{id}`
- `POST /improvements` (explicit opt-in; PPO only)
- `GET /experiments/{id}/export`

The API must validate inputs, enforce limits, bind to localhost, restrict CORS, and keep worker processes separate from request handlers.

## 16. UI information architecture

1. **Overview:** local machine readiness, recent experiments, CPU/resource settings.
2. **New Experiment:** environment, controller, scenario space, evaluation budget.
3. **Experiment Run:** job status, episode count, resource usage, cancel.
4. **Failure Explorer:** scenario ranking, violations, traces, reproducibility indicators.
5. **Policy Comparison:** same-suite paired metrics, failure regressions, uncertainty.
6. **Improvement Lab:** opt-in PPO retraining, nominal/failure mix, checkpoints.
7. **Settings:** storage path, worker limits, offline mode, exports.

The public website is **not** a cloud-hosted Studio. It is a static story of actual recorded results with install instructions and precomputed demonstrations.

## 17. Execution profiles

| Profile | Purpose | Initial budget |
|---|---|---|
| Quick | Smoke test, contract validation | 10–20 episodes, 1 worker |
| Standard | Stress search and comparison | 100–300 episodes, 1–2 workers |
| Research | Larger multi-seed tests, retraining | Explicit user-defined limits |

Do not promise runtimes before benchmarking the i5 machine. Run headless by default; store numerical traces and render selected replays on demand. Keep torch CPU thread counts bounded to avoid oversubscription.

## 18. Security and threat model

- A user-supplied Python controller is arbitrary code; only load trusted controllers.
- No public upload/execute endpoint, no cloud execution, no silent code downloads.
- Localhost binding and CORS restrictions reduce exposure but are not sufficient to sandbox arbitrary code.
- Job timeout, process isolation, file-path validation, and resource limits are required; strict OS-level sandboxing deferred.
- Avoid serializing/deserializing arbitrary Python pickle from untrusted sources.

## 19. Test strategy

- Unit: scenario validation, metric calculations, severity scoring, hash/version logic, storage.
- Contract: every environment and controller adapter passes common conformance suite.
- Physics: pump mass balance, bounds, stable integration; LunarLander disturbance application tests.
- Integration: create-run-save-reload-replay; cancel/interrupt job.
- Reproducibility: same build/controller/seed/scenario produces comparable traces; document nondeterminism.
- Benchmark: random vs TPE with equal episode budgets and repeated search seeds.
- Regression: baseline vs improved PPO on untouched test suites.
- Security: localhost binding, CORS checks, invalid paths, oversized jobs.

## 20. Phased delivery and gates

### Gate 0 — Machine benchmark (2–4 days)
- Install minimal dependencies; benchmark 100/1,000 headless episodes per environment using simple controllers.
- Measure wall-clock, peak RAM, CPU usage, episode throughput, and PPO 50k/100k training-step samples.
- Output a hardware report and revise default budgets. **Do not invent expected throughput.**

### Gate 1 — Research feasibility (approximately 3 weeks)
- Two environments, shared runner, PPO + rule controller, CLI, artifact persistence.
- Valid scenario generators and failure criteria.
- Random and TPE failure search with reproducible reports.
- **Go/no-go:** at least one nontrivial reproducible failure in each environment; fair baseline comparison; no need to demonstrate that TPE wins.

### Gate 2 — Scientific robustness (3–5 additional weeks)
- Multi-seed evaluation, scenario validity tiers, severity analysis, held-out suites.
- PPO failure-focused retraining and comparative report; negative results acceptable.
- **Gate:** methodologically defensible result, not guaranteed improvement.

### Gate 3 — Productization (4–6 additional weeks)
- Local API, job process management, Studio UI, export bundles, docs.
- **Gate:** independent developer installs and reproduces benchmark offline.

### Gate 4 — Public portfolio release (2–3 additional weeks)
- Precomputed demonstrations, static website, reproducibility guide, tagged open-source release.
- **Gate:** visitors can understand value without running local training; developers can use CLI/Studio.

**Planning range:** approximately 3–5 months at sustained part-time effort, subject to benchmark results and research iteration.

## 21. First implementation backlog (ordered)

1. Create monorepo, Python packaging, lint/test CI, environment locks.
2. Define `Scenario`, `Transition`, `EnvironmentAdapter`, `ControllerAdapter`, `EpisodeResult`.
3. Implement core runner and deterministic-seed manifest generation.
4. Integrate LunarLander and baseline controllers.
5. Implement simplified pump physics and tests.
6. Add SQLite/file artifact persistence.
7. Implement scenario validity/bounds and uniform random search.
8. Implement Optuna/TPE search behind same search interface.
9. Run equal-budget multi-seed benchmark and write report.
10. Add optional PPO retraining and held-out evaluation.
11. Build local API and subprocess job manager.
12. Build Studio and static showcase from recorded evidence.

## 22. Open decisions / assumptions

- Exact OS and Python version not yet specified.
- Whether LunarLander uses exclusively discrete actions for v1 (recommended yes).
- Pump simulation target variable: tank level first; pressure only if physical assumptions are documented.
- Specific disturbance bounds to be calibrated after environment verification.
- Final UI framework and local distribution format after prototype benchmarks.
- Whether reference MPC is included in first public release or a follow-up point release.

## 23. Definition of done

SIRFEX v1 is complete when a new developer can install it on a CPU-only machine, evaluate two controller types in two environments via the `sirfex` CLI/core API, find and replay meaningful failures, compare random vs optimized search fairly, optionally retrain PPO, examine held-out results, and explore the evidence in a local UI—without paying SIRFEX or requiring a cloud account.
