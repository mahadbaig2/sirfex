"""Optional local SIRFEX Evidence Explorer.

Run from the repository root with ``streamlit run studio/app.py``.  Streamlit
and Plotly are intentionally imported only by this optional application.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sirfex.experiments.comparison import compare_experiments
from sirfex.visualization.readers import EvidenceReader, EvidenceReadError
from sirfex.visualization.summaries import replay_label, search_comparison
from sirfex.visualization.trajectories import (
    failure_events,
    lunarlander_trajectory,
    pump_trajectory,
    search_history,
)


def _plotly():
    import plotly.graph_objects as go

    return go


def _paths() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-store", default="artifacts/experiments")
    parser.add_argument("--search-store", default="artifacts/searches")
    args, _ = parser.parse_known_args()
    return Path(args.experiment_store), Path(args.search_store)


def _overview(st, reader: EvidenceReader) -> None:
    st.header("Overview")
    overview = reader.overview()
    columns = st.columns(7)
    labels = (
        ("Experiments", "experiments"),
        ("Search runs", "search_runs"),
        ("Completed trials", "completed_trials"),
        ("Discoveries", "discoveries"),
        ("Observed failures", "observed_discoveries"),
        ("Proxy risks", "proxy_discoveries"),
        ("Replay verified", "replay_verified_discoveries"),
    )
    for column, (label, key) in zip(columns, labels, strict=True):
        column.metric(label, overview[key])
    if not overview["experiments"] and not overview["search_runs"]:
        st.info(
            "No local evidence found. Generate evidence with the existing "
            "sirfex experiments or search CLI commands."
        )


def _experiment_view(st, reader: EvidenceReader) -> None:
    st.header("Experiments")
    experiments = reader.list_experiments()
    if not experiments:
        st.info("No complete experiments are available in the configured store.")
        return
    selected = st.selectbox("Experiment", [item.experiment_id for item in experiments])
    try:
        stored = reader.load_experiment(selected)
    except EvidenceReadError as exc:
        st.error(str(exc))
        return
    manifest, result = stored.manifest, stored.result
    st.json(
        {
            "environment": manifest.environment_id,
            "controller": manifest.controller_id,
            "seed": manifest.scenario.get("seed"),
            "execution_status": manifest.execution_status,
            "outcome": result.get("outcome"),
            "observed_outcome": result.get("observed_outcome"),
        }
    )
    st.subheader("Failure evidence")
    events = failure_events(result)
    st.dataframe(
        events if events else [{"message": "No failure event recorded."}], use_container_width=True
    )
    st.subheader("Recorded trajectory")
    if manifest.environment_id == "pump":
        _pump_view(st, stored)
    elif manifest.environment_id == "lunarlander":
        _lander_view(st, stored)
    else:
        st.warning(f"No visualization mapping is available for {manifest.environment_id!r}.")
    if st.button("Verify replay"):
        report = reader.replay(selected)
        st.write(replay_label(report.status))
        st.json(report.to_dict())


def _pump_view(st, stored) -> None:
    go = _plotly()
    points = pump_trajectory(stored.trajectory, stored.manifest.scenario)
    if not points:
        st.info("This experiment has no recorded trajectory.")
        return
    figure = go.Figure()
    for key, label, color in (
        ("level", "Tank level", "#245b8a"),
        ("target_level", "Target", "#555"),
        ("min_level", "Minimum", "#b44"),
        ("max_level", "Maximum", "#b44"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[p["time"] for p in points],
                y=[p[key] for p in points],
                name=label,
                line={"color": color, "dash": "dot" if key != "level" else "solid"},
            )
        )
    violations = [p for p in points if p["constraint_violation"]]
    figure.add_trace(
        go.Scatter(
            x=[p["time"] for p in violations],
            y=[p["level"] for p in violations],
            mode="markers",
            name="Observed violation",
            marker={"color": "#b11", "size": 10},
        )
    )
    figure.update_layout(xaxis_title="Simulation step", yaxis_title="Normalized tank level")
    st.plotly_chart(figure, use_container_width=True)
    command = go.Figure(
        go.Scatter(
            x=[p["time"] for p in points], y=[p["command"] for p in points], name="Pump command"
        )
    )
    command.update_layout(xaxis_title="Simulation step", yaxis_title="Command")
    st.plotly_chart(command, use_container_width=True)
    step = st.slider("Recorded playback step", 0, len(points) - 1, 0)
    current = points[step]
    st.caption(
        "Playback uses the stored trajectory; changing the slider does not call environment.step()."
    )
    st.json(current)


def _lander_view(st, stored) -> None:
    go = _plotly()
    points = lunarlander_trajectory(stored.trajectory)
    st.caption(
        "Observation-based trajectory visualization. This is not a pixel-perfect "
        "Box2D reconstruction."
    )
    if not points:
        st.info("This experiment has no recorded trajectory.")
        return
    step = st.slider("Recorded playback step", 0, len(points) - 1, 0)
    figure = go.Figure(
        go.Scatter(
            x=[p["x"] for p in points],
            y=[p["y"] for p in points],
            mode="lines+markers",
            name="Recorded position",
        )
    )
    figure.add_trace(
        go.Scatter(x=[points[0]["x"]], y=[points[0]["y"]], mode="markers", name="Initial")
    )
    figure.add_trace(
        go.Scatter(x=[points[-1]["x"]], y=[points[-1]["y"]], mode="markers", name="Final")
    )
    figure.add_trace(
        go.Scatter(
            x=[points[step]["x"]],
            y=[points[step]["y"]],
            mode="markers",
            name="Selected step",
            marker={"size": 12},
        )
    )
    figure.update_layout(xaxis_title="Horizontal position", yaxis_title="Vertical position")
    st.plotly_chart(figure, use_container_width=True)
    st.json(points[step])


def _search_view(st, reader: EvidenceReader) -> None:
    st.header("Failure searches")
    searches = reader.list_searches()
    if not searches:
        st.info("No complete search runs are available in the configured store.")
        return
    selected = st.selectbox("Search run", [item["search_id"] for item in searches])
    try:
        loaded = reader.load_search(selected)
    except EvidenceReadError as exc:
        st.error(str(exc))
        return
    manifest, summary = loaded["manifest"], loaded["summary"]
    st.json(
        {
            "strategy": manifest.get("strategy"),
            "objective": manifest.get("objective_id"),
            "budget": manifest.get("trial_budget"),
            "summary": summary,
        }
    )
    go = _plotly()
    history = search_history(
        loaded["trials"], summary.get("direction", manifest.get("direction", "maximize"))
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[p["trial_index"] for p in history],
            y=[p["value"] for p in history],
            mode="markers",
            name="Trial objective",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[p["trial_index"] for p in history],
            y=[p["best"] for p in history],
            mode="lines+markers",
            name="Best so far",
        )
    )
    figure.update_layout(xaxis_title="Trial index", yaxis_title="Objective")
    st.plotly_chart(figure, use_container_width=True)
    st.subheader("Discoveries")
    st.dataframe(loaded["discoveries"], use_container_width=True)
    discoveries = loaded["discoveries"]
    if discoveries:
        discovery_id = st.selectbox(
            "Discovery evidence",
            [item["discovery_id"] for item in discoveries],
        )
        discovery = next(item for item in discoveries if item["discovery_id"] == discovery_id)
        st.json(
            {
                "discovery": discovery,
                "supporting_experiments": discovery.get("experiment_ids", []),
            }
        )
        supporting = discovery.get("experiment_ids", [])
        if supporting:
            experiment_id = st.selectbox("Open supporting experiment", supporting)
            try:
                evidence = reader.load_experiment(experiment_id)
                st.json({"manifest": evidence.manifest.to_dict(), "result": evidence.result})
            except EvidenceReadError as exc:
                st.error(str(exc))
    st.subheader("Trials")
    st.dataframe(loaded["trials"], use_container_width=True)


def _comparison_view(st, reader: EvidenceReader) -> None:
    st.header("Comparisons")
    experiments = reader.list_experiments()
    if len(experiments) >= 2:
        st.subheader("Experiment comparison")
        experiment_ids = [item.experiment_id for item in experiments]
        first_experiment = st.selectbox("First experiment", experiment_ids, key="experiment_first")
        second_experiment = st.selectbox(
            "Second experiment", experiment_ids, index=1, key="experiment_second"
        )
        if first_experiment != second_experiment:
            try:
                report = compare_experiments(
                    reader.experiments, first_experiment, second_experiment
                )
                st.json(report.to_dict())
            except (EvidenceReadError, ValueError) as exc:
                st.error(str(exc))
        else:
            st.warning("Select two different experiments.")
    searches = reader.list_searches()
    if len(searches) < 2:
        st.info("At least two search runs are needed for a comparison.")
        return
    ids = [item["search_id"] for item in searches]
    first = st.selectbox("First search", ids, key="comparison_first")
    second = st.selectbox("Second search", ids, index=1, key="comparison_second")
    if first == second:
        st.warning("Select two different search runs.")
        return
    result = search_comparison(reader.load_search(first), reader.load_search(second))
    st.write(
        "Compatible evidence"
        if result["comparable"]
        else "Incompatible evidence; charts are not overlaid."
    )
    st.json(result)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SIRFEX Evidence Explorer", layout="wide")
    experiment_root, search_root = _paths()
    reader = EvidenceReader(experiment_root, search_root)
    st.title("SIRFEX Evidence Explorer")
    st.caption("Local, read-only inspection of persisted simulation evidence")
    section = st.sidebar.radio(
        "Section", ["Overview", "Experiments", "Failure searches", "Comparisons"]
    )
    if section == "Overview":
        _overview(st, reader)
    elif section == "Experiments":
        _experiment_view(st, reader)
    elif section == "Failure searches":
        _search_view(st, reader)
    else:
        _comparison_view(st, reader)


if __name__ == "__main__":
    main()
