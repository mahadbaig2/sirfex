# SIRFEX Evidence Explorer

Phase 2.5 adds an optional local Streamlit interface for inspecting evidence already produced by SIRFEX. The headless engine and CLI remain usable without Studio dependencies.

## Installation and launch

From the repository root on Python 3.12:

```powershell
pip install -e ".[studio]"
streamlit run studio/app.py
```

The app binds to Streamlit's local development server. It does not provide a remote service, authentication, or background execution.

By default it reads `artifacts/experiments` and `artifacts/searches`. Alternate paths can be supplied after `--`:

```powershell
streamlit run studio/app.py -- --experiment-store D:\evidence\experiments --search-store D:\evidence\searches
```

## Navigation

- **Overview** counts actual saved experiments, searches, completed trials, discoveries, observed failures, inferred proxy risks, and replay-verified discoveries.
- **Experiments** shows manifest/result metadata, failure records, recorded trajectory charts, and replay verification.
- **Failure searches** shows strategy/objective metadata, trial and best-so-far history, trial records, and persisted discoveries.
- Phase 3.2 searches also show read-only avoidability counts and per-reference failure, witness and replay evidence from `avoidable.json`.
- **Comparisons** checks compatible search evidence before presenting descriptive summaries.

The comparison section also exposes the existing descriptive experiment comparison when at least two saved experiments exist.

## Supported visualizations

Pump charts use recorded tank level, target and configured bounds, command values, demand where present, and violation markers. The playback slider only selects a JSONL record; it never calls `Environment.step()` or re-simulates the episode.

LunarLander views plot recorded horizontal/vertical observations and show the remaining observation fields for a selected step. This is explicitly an observation-based trajectory visualization, not a pixel-perfect Box2D reconstruction. Terminal instability remains a proxy unless the adapter recorded reliable explicit outcome evidence.

## Replay verification

The Explorer's only simulation-related action is **Verify replay**. It calls the existing `sirfex.experiments.replay.replay_experiment` API and displays exact match, tolerance match, mismatch, or unavailable status plus mismatch details. The Explorer does not implement a second runner or write over original artifacts.

## Evidence integrity and limitations

Artifacts are read through the existing stores, including schema and path validation. Missing, malformed, incomplete, unsupported, or unavailable evidence is reported rather than replaced with fabricated values. The reader does not load trajectories while listing experiments or searches; detailed traces are loaded only after selection.

The interface is a local inspection tool. It does not train controllers, run searches, edit scenarios, create environments, expose an API, or provide a universal safety score. Plotly and Streamlit are optional and are never imported by `sirfex.core`, `sirfex.experiments`, `sirfex.search`, or `sirfex.environments`.
