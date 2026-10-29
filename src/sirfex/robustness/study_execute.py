"""Work-item execution for a gated Phase 3.7 pump study."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sirfex.core import Scenario
from sirfex.experiments.replay import compare_results
from sirfex.robustness.differential import classify_pair
from sirfex.robustness.disturbance import evaluate_direction, prepare_grid
from sirfex.robustness.evidence import archive_selected_findings
from sirfex.robustness.identity import scenario_identity, semantic_hash
from sirfex.robustness.margins import evaluate_margin
from sirfex.robustness.service import _Session
from sirfex.robustness.store import RobustnessStorageError, _atomic_json, atomic_text
from sirfex.robustness.study_report import build_study_report, render_study_markdown
from sirfex.robustness.study_store import InvocationBudget, StudyBudgetExhausted, StudyStore
from sirfex.search.space import SearchSpace
from sirfex.training.checkpoints import verify_checkpoint
from sirfex.training.feasibility import analyze_feasibility
from sirfex.training.protocol import leakage_check


def _specs(protocol: dict, root: Path) -> dict[str, dict]:
    result = {}
    for row in protocol["controller_inventory"]:
        key = row.get("source_key") or row.get("controller_id") or row["id"]
        if "source_artifact" in row:
            path = (root / row["source_artifact"]).resolve()
            verify_checkpoint(path, row["sha256"])
            metadata_path = path.with_suffix(".metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["checkpoint_id"] != key
                or metadata["sha256"] != row["sha256"]
                or metadata["training_run_id"] != row["source_run"]
            ):
                raise RobustnessStorageError("Checkpoint metadata identity mismatch")
            metadata["path"] = str(path)
            result[key] = {"id": "ppo", "config": {"checkpoint": metadata}}
        else:
            result[key] = {
                "id": row.get("controller_id", row.get("id")),
                "config": row.get("config", {}),
            }
    return result


class ControllerPool:
    def __init__(self, specs: dict[str, dict], *, max_open: int = 2):
        self.specs = specs
        self.max_open = max_open
        self.sessions: dict[str, _Session] = {}

    def get(self, key: str) -> _Session:
        if key not in self.specs:
            raise RobustnessStorageError("Controller is outside the study inventory")
        if key not in self.sessions:
            if len(self.sessions) >= self.max_open:
                oldest = next(iter(self.sessions))
                self.sessions.pop(oldest).close()
            self.sessions[key] = _Session(self.specs[key])
        return self.sessions[key]

    def evaluate(self, key: str, scenario: Scenario):
        return self.get(key).evaluate(scenario)

    def seconds(self, key: str) -> float:
        return self.sessions[key].seconds if key in self.sessions else 0.0

    def close(self) -> None:
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()


def compact_episode(result) -> dict:
    margin = evaluate_margin(result)
    if margin.evidence_status != "valid":
        raise ValueError("Incomplete episode cannot enter a scientific comparison")
    scenario = Scenario.from_dict(result.scenario)
    target = float(scenario.parameters.get("target_level", 0.5))
    levels = [float(row["info"]["level"]) for row in result.trace]
    if len(levels) != result.episode_length or not levels:
        raise ValueError("Tracking evidence is missing")
    return {
        "scenario_id": scenario_identity(scenario),
        "controller_id": result.controller_id,
        "episode_sha256": semantic_hash(result.to_dict()),
        "tracking_mae": sum(abs(level - target) for level in levels) / len(levels),
        "margin": margin.to_dict(),
        "episode_length": result.episode_length,
    }


class StudyEngine:
    def __init__(
        self,
        protocol: dict,
        manifest: dict,
        plan: dict,
        root: Path | str,
        output: Path | str,
        *,
        synthetic: bool = False,
        clock=time.monotonic,
    ):
        if synthetic != (plan["profile"] == "synthetic_test_only"):
            raise ValueError("Synthetic execution profile mismatch")
        if (
            plan["protocol_sha256"] != protocol["semantic_sha256"]
            or plan["scenario_manifest_sha256"] != manifest["semantic_sha256"]
        ):
            raise RobustnessStorageError("Execution plan input identity changed")
        self.protocol = protocol
        self.manifest = manifest
        self.plan = plan
        self.root = Path(root).resolve()
        self.store = StudyStore(output, plan)
        self.store.open()
        self.budget = InvocationBudget(self.store, clock=clock)
        self.specs = _specs(protocol, self.root)
        self.pool = ControllerPool(self.specs)
        self.scenarios = {
            row["scenario_id"]: Scenario.from_dict(row["scenario"]) for row in manifest["scenarios"]
        }
        self.prepared = {}
        self.proposals: dict[str, list[dict]] = {}

    def close(self) -> None:
        self.pool.close()

    def prepare(self) -> None:
        """Recompute shared physics and bind its ordered hashes before evaluation."""
        started = self.budget.clock()
        grid = self.plan["grid"]
        for anchor_id in self.plan["nominal_scenarios"]:
            anchor = self.scenarios[anchor_id]
            for direction in self.plan["directions"]:
                self.budget.check()
                key = (anchor_id, tuple(direction))
                self.prepared[key] = prepare_grid(
                    anchor,
                    tuple(direction),
                    grid,
                    protocol_sha256=self.plan["protocol_sha256"],
                )
        record = {
            "schema_version": "pump-study-prepared-grids:1",
            "plan_sha256": self.plan["semantic_sha256"],
            "grids": [
                {
                    "anchor_id": anchor_id,
                    "direction": list(direction),
                    "semantic_sha256": grid.semantic_sha256,
                }
                for (anchor_id, direction), grid in self.prepared.items()
            ],
        }
        record["semantic_sha256"] = semantic_hash(record)
        path = self.store.directory / "prepared_grids.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != record:
                raise RobustnessStorageError("Prepared physical grids changed on resume")
        else:
            self.store._check_storage(record)
            _atomic_json(path, record)
        self.store.charge(self.budget.clock() - started, {})
        self.budget.charged_until = self.budget.clock()

    def _run_item(self, item: dict, evaluate, *, controllers: tuple[str, ...], resume: bool):
        previous = self.store.read(item["work_id"])
        if previous and previous["status"] in {"completed", "censored"}:
            return previous
        if previous and not resume:
            raise RobustnessStorageError("Incomplete work item requires explicit resume")
        self.budget.check(controllers)
        self.store.write(item, "running", resume=resume)
        before = {key: self.pool.seconds(key) for key in controllers}
        try:
            result = evaluate()
            status = "censored" if result.get("censored", False) else "completed"
            saved = self.store.write(item, status, result, resume=resume)
        except StudyBudgetExhausted:
            self.budget.charge(
                {key: max(0.0, self.pool.seconds(key) - before[key]) for key in controllers}
            )
            raise
        except Exception as exc:
            try:
                self.store.write(
                    item,
                    "invalid" if hasattr(exc, "evidence") else "failed",
                    {"error_type": type(exc).__name__, "message": str(exc)},
                    resume=resume,
                )
            finally:
                self.budget.charge(
                    {key: max(0.0, self.pool.seconds(key) - before[key]) for key in controllers}
                )
            raise
        self.budget.charge(
            {key: max(0.0, self.pool.seconds(key) - before[key]) for key in controllers}
        )
        return saved

    def execute_fixed_and_grid(self, *, resume: bool = False, interrupt_after: int | None = None):
        self.prepare()
        completed_this_invocation = 0
        try:
            for item in self.plan["work_items"]:
                if item["stage"] not in {"fixed_stress", "grid"}:
                    continue
                if interrupt_after is not None and completed_this_invocation >= interrupt_after:
                    raise StudyBudgetExhausted("synthetic_interruption")
                controller = item["controller"]
                if item["stage"] == "fixed_stress":
                    scenario = self.scenarios[item["scenario_id"]]

                    def evaluate_fixed(controller=controller, scenario=scenario):
                        return compact_episode(self.pool.evaluate(controller, scenario))

                    self._run_item(item, evaluate_fixed, controllers=(controller,), resume=resume)
                else:
                    self._run_item(
                        item,
                        lambda item=item: self._evaluate_grid_item(item),
                        controllers=(controller,),
                        resume=resume,
                    )
                completed_this_invocation += 1
        finally:
            self.pool.close()
        return self.store.status()

    def _evaluate_grid_item(self, item: dict) -> dict:
        controller = item["controller"]
        anchor_id = item["anchor_id"]
        anchor = self.scenarios[anchor_id]
        direction = tuple(item["direction"])
        prepared = self.prepared[(anchor_id, direction)]
        nominal = None

        def evaluator(scenario):
            nonlocal nominal
            result = self.pool.evaluate(controller, scenario)
            if (
                direction == tuple(self.plan["directions"][0])
                and scenario_identity(scenario) == anchor_id
            ):
                nominal = compact_episode(result)
            return result

        cell = evaluate_direction(
            anchor,
            direction,
            self.plan["grid"],
            self.specs[controller]["id"],
            evaluator,
            prepared=prepared,
            refine_feasibility=False,
            protocol_sha256=self.plan["protocol_sha256"],
        ).to_dict()
        return {
            "cell": cell,
            "nominal": nominal,
            "prepared_grid_sha256": prepared.semantic_sha256,
            "censored": cell["status"].startswith("censored") or cell["status"] == "undefined_axis",
        }

    def prepare_proposals(self) -> None:
        """Generate all proposals independent of outcomes and verify them on resume."""
        search = self.protocol["search"]
        if isinstance(search.get("space"), dict):
            space = SearchSpace.from_dict(search["space"])
        else:
            space = SearchSpace.from_file(str(self.root / search["space"]))
        excluded = tuple(self.scenarios.values())
        for pair_index, pair in enumerate(self.plan["pairs"]):
            self.budget.check()
            rng = np.random.default_rng(search["search_seed"] + pair_index)
            accepted = []
            rows = []
            for index in range(search["proposals_per_pair"]):
                params = space.sample_parameters(rng)
                seed = (
                    search["scenario_seed_start"]
                    + pair_index * search["proposals_per_pair"]
                    + index
                )
                try:
                    scenario = space.scenario(params, seed)
                    scenario_id = scenario_identity(scenario)
                    separate = leakage_check(
                        {"excluded": excluded + tuple(accepted), "candidate": [scenario]}
                    )["passed"]
                    if not separate:
                        status = "excluded_duplicate"
                    else:
                        feasibility = analyze_feasibility(scenario)
                        status = (
                            "evaluated"
                            if feasibility["finite_horizon_viable"] is True
                            else "excluded_infeasible"
                        )
                    if status == "evaluated":
                        accepted.append(scenario)
                    row = {
                        "proposal_index": index,
                        "scenario_id": scenario_id,
                        "scenario": scenario.to_dict(),
                        "status": status,
                    }
                except ValueError as exc:
                    row = {
                        "proposal_index": index,
                        "status": "excluded_invalid",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "parameters": params,
                        "scenario_seed": seed,
                    }
                rows.append(row)
            record = {
                "schema_version": "pump-study-proposals:1",
                "plan_sha256": self.plan["semantic_sha256"],
                "pair_id": pair["id"],
                "rows": rows,
            }
            record["semantic_sha256"] = semantic_hash(record)
            path = self.store.directory / f"proposals_{pair_index:02d}.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved != record:
                    raise RobustnessStorageError("Differential proposal sequence changed")
            else:
                self.store._check_storage(record)
                _atomic_json(path, record)
            self.proposals[pair["id"]] = rows

    def _evaluate_proposal(self, item: dict) -> dict:
        pair = next(row for row in self.plan["pairs"] if row["id"] == item["pair_id"])
        proposal = self.proposals[pair["id"]][item["proposal_index"]]
        if proposal["status"] != "evaluated":
            return {
                "proposal_status": proposal["status"],
                "scenario_id": proposal.get("scenario_id"),
                "proposal_index": item["proposal_index"],
            }
        scenario = Scenario.from_dict(proposal["scenario"])
        x, y = pair["x"], pair["y"]
        first = self.pool.evaluate(x, scenario)
        second = self.pool.evaluate(y, scenario)
        mx, my = evaluate_margin(first), evaluate_margin(second)
        if mx.scenario_id != proposal["scenario_id"] or my.scenario_id != proposal["scenario_id"]:
            raise RobustnessStorageError("Differential controller changed proposed scenario")
        initial = classify_pair(
            mx,
            my,
            finite_horizon_viable=True,
            near_miss_delta=self.protocol["search"]["class_a_margin_gap"],
        )
        replay = None
        if initial.classification == "class_b_unverified":
            first_repeat = self.pool.evaluate(x, scenario)
            second_repeat = self.pool.evaluate(y, scenario)
            replay = (
                compare_results(first.to_dict(), first_repeat.to_dict()).status,
                compare_results(second.to_dict(), second_repeat.to_dict()).status,
            )
        finding = classify_pair(
            mx,
            my,
            finite_horizon_viable=True,
            near_miss_delta=self.protocol["search"]["class_a_margin_gap"],
            replay_statuses=replay,
        ).to_dict()
        finding["controller_x"] = x
        finding["controller_y"] = y
        return {
            "proposal_status": "evaluated",
            "proposal_index": item["proposal_index"],
            "scenario_id": proposal["scenario_id"],
            "scenario": proposal["scenario"],
            "finding": finding,
            "margin_x": mx.to_dict(),
            "margin_y": my.to_dict(),
        }

    def _archive_pair(self, pair: dict) -> dict:
        proposals = [
            row
            for row in self.plan["work_items"]
            if row["stage"] == "differential" and row["pair_id"] == pair["id"]
        ]
        records = []
        for item in proposals:
            saved = self.store.read(item["work_id"])
            if saved is None or saved["status"] != "completed":
                raise RobustnessStorageError("Differential pair has incomplete proposals")
            result = saved["result"]
            records.append({"status": result["proposal_status"], **result})
        ranked = sorted(
            (row["finding"] for row in records if row["status"] == "evaluated"),
            key=lambda row: (
                {"class_b_unverified": 0, "class_a": 1, "none": 2}[row["classification"]],
                -row["signed_margin_gap"],
                row["scenario_id"],
            ),
        )
        top = self.protocol["search"]["top_per_class"]
        result = {
            "schema_version": "pump-differential-search:1",
            "method": "seeded_random_nonadaptive",
            "controller_x": pair["x"],
            "controller_y": pair["y"],
            "proposal_count": len(proposals),
            "search_seed": proposals[0]["search_seed"],
            "scenario_seed_start": proposals[0]["scenario_seed"],
            "near_miss_delta": self.protocol["search"]["class_a_margin_gap"],
            "coverage": {
                "proposed": len(proposals),
                **{
                    status: sum(row["status"] == status for row in records)
                    for status in (
                        "evaluated",
                        "excluded_duplicate",
                        "excluded_infeasible",
                        "excluded_invalid",
                    )
                },
            },
            "findings": {
                "class_b": [],
                "class_b_unverified": [
                    row for row in ranked if row["classification"] == "class_b_unverified"
                ][:top],
                "class_a": [row for row in ranked if row["classification"] == "class_a"][:top],
            },
            "records": records,
        }
        self.budget.check((pair["x"], pair["y"]), reserve_bytes=1_000_000)
        archived = archive_selected_findings(
            result,
            [self.specs[pair["x"]], self.specs[pair["y"]]],
            self.pool.evaluate,
            self.store.directory / pair["id"],
            self.plan["protocol_sha256"],
            label_specs={pair["x"]: self.specs[pair["x"]], pair["y"]: self.specs[pair["y"]]},
            storage_guard=lambda reserve: self.store.check_budget(reserve_bytes=reserve),
        )
        self.store.check_budget()
        return {"pair_id": pair["id"], "result": archived}

    def execute_differential(
        self, *, resume: bool = False, interrupt_after: int | None = None
    ) -> dict:
        self.prepare_proposals()
        completed_this_invocation = 0
        try:
            for item in self.plan["work_items"]:
                if item["stage"] not in {"differential", "archive"}:
                    continue
                if interrupt_after is not None and completed_this_invocation >= interrupt_after:
                    raise StudyBudgetExhausted("synthetic_interruption")
                pair = next(row for row in self.plan["pairs"] if row["id"] == item["pair_id"])
                controllers = (pair["x"], pair["y"])
                self._run_item(
                    item,
                    (lambda item=item: self._evaluate_proposal(item))
                    if item["stage"] == "differential"
                    else (lambda pair=pair: self._archive_pair(pair)),
                    controllers=controllers,
                    resume=resume,
                )
                completed_this_invocation += 1
        finally:
            self.pool.close()
        return self.store.status()

    def regenerate_report(self, *, persist: bool = True) -> dict:
        """Regenerate from verified cases, including durable selected experiments."""
        report = build_study_report(self.store, self.protocol, self.manifest, self.specs)
        markdown = render_study_markdown(report)
        if persist:
            prefix = "report" if report["execution_status"] == "complete" else "partial_report"
            json_path = self.store.directory / f"{prefix}.json"
            markdown_path = self.store.directory / f"{prefix}.md"
            for path, value in ((json_path, report), (markdown_path, markdown)):
                if path.exists():
                    saved = (
                        json.loads(path.read_text(encoding="utf-8"))
                        if path.suffix == ".json"
                        else path.read_text(encoding="utf-8")
                    )
                    if saved != value:
                        raise RobustnessStorageError(
                            "Stored scientific report differs from verified evidence"
                        )
                else:
                    self.store._check_storage(report)
                    if path.suffix == ".json":
                        _atomic_json(path, report)
                    else:
                        atomic_text(path, markdown)
        return report

    def execute_report(self, *, resume: bool = False) -> dict:
        item = next(row for row in self.plan["work_items"] if row["stage"] == "report")
        stage_status = self.store.status()["stages"]
        if any(
            counts["pending"] or counts["running"] or counts["failed"] or counts["invalid"]
            for stage, counts in stage_status.items()
            if stage != "report"
        ):
            return self.regenerate_report(persist=False)
        self._run_item(
            item,
            lambda: {"report_sha256": self.regenerate_report(persist=True)["semantic_sha256"]},
            controllers=(),
            resume=resume,
        )
        return self.regenerate_report(persist=True)
