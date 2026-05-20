import importlib.util
import csv
import json
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openevolve.multi_task_shared_then_adapt import posthoc_ood
from openevolve.multi_task_shared_then_adapt.circle_packing_rectangle import (
    CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR,
    resolve_eval_task_specs as resolve_rectangle_eval_task_specs,
    resolve_ood_task_specs as resolve_rectangle_ood_task_specs,
    resolve_task_specs as resolve_rectangle_task_specs,
)
from openevolve.multi_task_shared_then_adapt.heilbronn_triangle import (
    HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR,
    resolve_eval_task_specs as resolve_heilbronn_eval_task_specs,
    resolve_ood_task_specs as resolve_heilbronn_ood_task_specs,
    resolve_task_specs as resolve_heilbronn_task_specs,
)
from openevolve.multi_task_shared_then_adapt.workflow import load_manifest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_program(run_dir: Path, *, code: str = "def placeholder():\n    return None\n"):
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "best_program.py").write_text(code, encoding="utf-8")
    (best_dir / "best_program_info.json").write_text(
        json.dumps({"language": "python", "metrics": {"score": 1.0}}),
        encoding="utf-8",
    )
    return best_dir / "best_program.py"


def _fake_results_tree(tmp_path: Path, family: str, task_ids: list[str]) -> Path:
    results_dir = tmp_path / family / "run_01"
    _write_program(results_dir / "shared_run")
    tasks = {}
    for task_id in task_ids:
        adaptation_dir = results_dir / "adaptation" / task_id
        baseline_dir = results_dir / "baselines" / task_id
        _write_program(adaptation_dir)
        _write_program(baseline_dir)
        tasks[task_id] = {
            "adaptation_output_dir": str(adaptation_dir),
            "baseline_output_dir": str(baseline_dir),
        }
    (results_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "workflow": "multi_task_shared_then_adapt",
                "family": family,
                "shared_run": {"output_dir": str(results_dir / "shared_run")},
                "tasks": tasks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results_dir


def _heilbronn_candidate_code() -> str:
    return (
        "import numpy as np\n\n"
        "_BARY = np.asarray([\n"
        "    [0.80, 0.10, 0.10],\n"
        "    [0.10, 0.80, 0.10],\n"
        "    [0.10, 0.10, 0.80],\n"
        "    [0.60, 0.30, 0.10],\n"
        "    [0.60, 0.10, 0.30],\n"
        "    [0.30, 0.60, 0.10],\n"
        "    [0.30, 0.10, 0.60],\n"
        "    [0.10, 0.60, 0.30],\n"
        "    [0.10, 0.30, 0.60],\n"
        "    [0.45, 0.35, 0.20],\n"
        "    [0.35, 0.20, 0.45],\n"
        "    [0.20, 0.45, 0.35],\n"
        "    [0.34, 0.33, 0.33],\n"
        "    [0.50, 0.25, 0.25],\n"
        "], dtype=float)\n\n"
        "def construct_points(n):\n"
        "    bary = _BARY[:n]\n"
        "    return np.column_stack((2.0 * bary[:, 1], bary[:, 2]))\n\n"
        "def run_heilbronn(n):\n"
        "    return construct_points(n)\n"
    )


def _rectangle_candidate_code() -> str:
    return (
        "import numpy as np\n\n"
        "def construct_packing(n):\n"
        "    alpha = 1.0\n"
        "    cols = int(np.ceil(np.sqrt(max(1, n))))\n"
        "    rows = int(np.ceil(n / cols))\n"
        "    xs = np.linspace(0.1, 0.9, cols)\n"
        "    ys = np.linspace(0.1, 0.9, rows)\n"
        "    centers = []\n"
        "    for y in ys:\n"
        "        for x in xs:\n"
        "            centers.append((float(x), float(y)))\n"
        "    centers = np.asarray(centers[:n], dtype=float)\n"
        "    radii = np.full(n, 0.001, dtype=float)\n"
        "    return centers, radii, alpha, float(np.sum(radii))\n\n"
        "def run_packing(n):\n"
        "    return construct_packing(n)\n"
    )


def test_family_resolvers_keep_all_training_only_and_allow_explicit_ood():
    assert [task.task_id for task in resolve_heilbronn_task_specs("all")] == [
        "heil_tri_n9",
        "heil_tri_n10",
        "heil_tri_n11",
        "heil_tri_n12",
    ]
    assert [task.task_id for task in resolve_heilbronn_eval_task_specs("all")] == [
        "heil_tri_n9",
        "heil_tri_n10",
        "heil_tri_n11",
        "heil_tri_n12",
    ]
    assert resolve_heilbronn_eval_task_specs("heil_tri_n8")[0].target_min_area == pytest.approx(
        0.06778914101959856
    )
    assert resolve_heilbronn_eval_task_specs("heil_tri_n13")[0].n_points == 13
    assert resolve_heilbronn_eval_task_specs("heil_tri_n14")[0].target_min_area == pytest.approx(
        0.02377577301721215
    )
    assert [task.task_id for task in resolve_heilbronn_ood_task_specs(None)] == [
        "heil_tri_n8",
        "heil_tri_n13",
        "heil_tri_n14",
    ]
    assert [task.task_id for task in resolve_heilbronn_ood_task_specs("heil_tri_n14")] == [
        "heil_tri_n14",
    ]
    with pytest.raises(ValueError):
        resolve_heilbronn_eval_task_specs("heil_tri_n15")
    with pytest.raises(ValueError):
        resolve_heilbronn_ood_task_specs("heil_tri_n9")

    assert [task.task_id for task in resolve_rectangle_task_specs("all")] == [
        "cp_rect_n20",
        "cp_rect_n21",
        "cp_rect_n22",
        "cp_rect_n23",
    ]
    assert [task.task_id for task in resolve_rectangle_eval_task_specs("all")] == [
        "cp_rect_n20",
        "cp_rect_n21",
        "cp_rect_n22",
        "cp_rect_n23",
    ]
    assert resolve_rectangle_eval_task_specs("cp_rect_n19")[0].target_sum_radii == pytest.approx(
        2.241
    )
    assert resolve_rectangle_eval_task_specs("cp_rect_n24")[0].n_circles == 24
    assert resolve_rectangle_eval_task_specs("cp_rect_n25")[0].target_sum_radii == pytest.approx(
        2.592
    )
    assert [task.task_id for task in resolve_rectangle_ood_task_specs(None)] == [
        "cp_rect_n19",
        "cp_rect_n24",
        "cp_rect_n25",
    ]
    assert [task.task_id for task in resolve_rectangle_ood_task_specs("cp_rect_n25")] == [
        "cp_rect_n25",
    ]
    with pytest.raises(ValueError):
        resolve_rectangle_eval_task_specs("cp_rect_n26")
    with pytest.raises(ValueError):
        resolve_rectangle_ood_task_specs("cp_rect_n20")


def test_evaluators_support_explicit_ood_tasks_and_keep_all_training_only(monkeypatch, tmp_path):
    heil_eval = _load_module(
        REPO_ROOT / "examples" / "heilbronn_triangle_emo_sta" / "evaluator.py",
        "heilbronn_ood_eval_test",
    )
    heil_candidate = tmp_path / "heil_candidate.py"
    heil_candidate.write_text(_heilbronn_candidate_code(), encoding="utf-8")

    for task_id in ("heil_tri_n8", "heil_tri_n13", "heil_tri_n14"):
        monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, task_id)
        result = heil_eval.evaluate(str(heil_candidate))
        assert result.artifacts["selected_task_ids"] == [task_id]
        assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
        assert result.metrics["target_min_area"] > 0.0

    monkeypatch.setenv(HEILBRONN_TRIANGLE_TASK_SELECTOR_ENV_VAR, "all")
    result = heil_eval.evaluate(str(heil_candidate))
    assert result.artifacts["selected_task_ids"] == [
        "heil_tri_n9",
        "heil_tri_n10",
        "heil_tri_n11",
        "heil_tri_n12",
    ]

    rect_eval = _load_module(
        REPO_ROOT / "examples" / "circle_packing_rectangle_emo_sta" / "evaluator.py",
        "rectangle_ood_eval_test",
    )
    rect_candidate = tmp_path / "rect_candidate.py"
    rect_candidate.write_text(_rectangle_candidate_code(), encoding="utf-8")

    for task_id in ("cp_rect_n19", "cp_rect_n24", "cp_rect_n25"):
        monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, task_id)
        result = rect_eval.evaluate(str(rect_candidate))
        assert result.artifacts["selected_task_ids"] == [task_id]
        assert result.metrics["combined_score"] == pytest.approx(result.metrics["score"])
        assert result.metrics["target_sum_radii"] > 0.0

    monkeypatch.setenv(CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR, "all")
    result = rect_eval.evaluate(str(rect_candidate))
    assert result.artifacts["selected_task_ids"] == [
        "cp_rect_n20",
        "cp_rect_n21",
        "cp_rect_n22",
        "cp_rect_n23",
    ]


def test_posthoc_program_discovery_uses_finished_static_layout(tmp_path):
    manifest = load_manifest(
        REPO_ROOT / "multi_task_shared_then_adapt" / "configs" / "heilbronn_triangle_emo_sta.yaml"
    )
    task_ids = [task.task_id for task in resolve_heilbronn_task_specs("all")]
    results_dir = _fake_results_tree(tmp_path, "heilbronn_triangle", task_ids)

    programs = posthoc_ood.discover_finished_programs(
        manifest=manifest,
        results_dir=results_dir,
    )

    assert [program.label for program in programs] == [
        "shared_best",
        "adapted__heil_tri_n9",
        "adapted__heil_tri_n10",
        "adapted__heil_tri_n11",
        "adapted__heil_tri_n12",
        "baseline__heil_tri_n9",
        "baseline__heil_tri_n10",
        "baseline__heil_tri_n11",
        "baseline__heil_tri_n12",
    ]
    assert all(program.program_path.name == "best_program.py" for program in programs)


def test_posthoc_summary_generation_writes_compact_json_and_csv(monkeypatch, tmp_path):
    manifest = load_manifest(
        REPO_ROOT
        / "multi_task_shared_then_adapt"
        / "configs"
        / "circle_packing_rectangle_emo_sta.yaml"
    )
    task_ids = [task.task_id for task in resolve_rectangle_task_specs("all")]
    results_dir = _fake_results_tree(tmp_path, "circle_packing_rectangle", task_ids)

    def fake_evaluate_program_on_ood_tasks(**kwargs):
        results = {}
        for task in kwargs["ood_task_specs"]:
            target = float(task.target_sum_radii)
            ratio = 1.0 if task.task_id == "cp_rect_n19" else 0.9
            results[task.task_id] = {
                "score": ratio,
                "combined_score": ratio,
                "metrics": {
                    "sum_radii": target * ratio,
                    "target_sum_radii": target,
                    "target_ratio": ratio,
                    "validity": 1.0,
                    "alpha": 0.95,
                },
            }
        return results

    monkeypatch.setattr(
        posthoc_ood,
        "evaluate_program_on_ood_tasks",
        fake_evaluate_program_on_ood_tasks,
    )

    summary = posthoc_ood.run_posthoc_ood_evaluation(
        manifest=manifest,
        results_dir=results_dir,
    )

    summary_path = Path(summary["summary_path"])
    csv_path = Path(summary["csv_path"])
    assert summary_path.is_file()
    assert csv_path.is_file()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["evaluation_regime"] == "posthoc_unseen_nearby_n"
    assert payload["note"].startswith("This is post-hoc OOD evaluation")
    assert payload["summary_path"] == str(summary_path)
    assert payload["csv_path"] == str(csv_path)
    assert payload["ood_tasks"] == ["cp_rect_n19", "cp_rect_n24", "cp_rect_n25"]
    assert set(payload["programs"]) == {
        "shared_best",
        "adapted__cp_rect_n20",
        "adapted__cp_rect_n21",
        "adapted__cp_rect_n22",
        "adapted__cp_rect_n23",
        "baseline__cp_rect_n20",
        "baseline__cp_rect_n21",
        "baseline__cp_rect_n22",
        "baseline__cp_rect_n23",
    }
    payload_json = json.dumps(payload)
    assert '"centers"' not in payload_json
    assert '"radii"' not in payload_json

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "backward" in csv_text
    assert "forward" in csv_text
    assert "objective_value" in csv_text
    assert "target_value" in csv_text


def test_posthoc_summary_generation_accepts_explicit_new_ood_tasks(
    monkeypatch,
    tmp_path,
):
    manifest = load_manifest(
        REPO_ROOT
        / "multi_task_shared_then_adapt"
        / "configs"
        / "circle_packing_rectangle_emo_sta.yaml"
    )
    task_ids = [task.task_id for task in resolve_rectangle_task_specs("all")]
    results_dir = _fake_results_tree(tmp_path, "circle_packing_rectangle", task_ids)

    def fake_evaluate_program_on_ood_tasks(**kwargs):
        results = {}
        for task in kwargs["ood_task_specs"]:
            target = float(task.target_sum_radii)
            ratio = 1.0 if task.task_id == "cp_rect_n19" else 0.9
            results[task.task_id] = {
                "score": ratio,
                "combined_score": ratio,
                "metrics": {
                    "sum_radii": target * ratio,
                    "target_sum_radii": target,
                    "target_ratio": ratio,
                    "validity": 1.0,
                    "alpha": 0.95,
                },
            }
        return results

    monkeypatch.setattr(
        posthoc_ood,
        "evaluate_program_on_ood_tasks",
        fake_evaluate_program_on_ood_tasks,
    )

    summary = posthoc_ood.run_posthoc_ood_evaluation(
        manifest=manifest,
        results_dir=results_dir,
        ood_task_ids="cp_rect_n19,cp_rect_n24,cp_rect_n25",
        include_adapted=False,
        include_baselines=False,
    )

    summary_path = Path(summary["summary_path"])
    csv_path = Path(summary["csv_path"])
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["ood_tasks"] == ["cp_rect_n19", "cp_rect_n24", "cp_rect_n25"]
    assert set(payload["programs"]) == {"shared_best"}

    n25 = payload["programs"]["shared_best"]["ood_results"]["cp_rect_n25"]
    assert n25["metrics"]["sum_radii"] == pytest.approx(2.592 * 0.9)
    assert n25["metrics"]["target_sum_radii"] == pytest.approx(2.592)
    payload_json = json.dumps(payload)
    assert '"centers"' not in payload_json
    assert '"radii"' not in payload_json

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    n25_row = next(row for row in rows if row["ood_task_id"] == "cp_rect_n25")
    assert n25_row["direction"] == "forward"
    assert float(n25_row["objective_value"]) == pytest.approx(2.592 * 0.9)
    assert float(n25_row["target_value"]) == pytest.approx(2.592)


def test_evaluate_program_on_ood_tasks_compacts_metrics(monkeypatch, tmp_path):
    program_path = tmp_path / "dummy.py"
    program_path.write_text("def run_packing(n):\n    return [], [], 1.0\n", encoding="utf-8")

    class FakeEvaluatorModule:
        @staticmethod
        def evaluate(program_path):
            task_id = os.environ[CIRCLE_PACKING_RECTANGLE_TASK_SELECTOR_ENV_VAR]
            targets = {
                "cp_rect_n19": 2.241,
                "cp_rect_n24": 2.535,
                "cp_rect_n25": 2.592,
            }
            target = targets[task_id]
            return {
                "metrics": {
                    "sum_radii": target,
                    "target_sum_radii": 999.0,
                    "target_ratio": 1.0,
                    "validity": 1.0,
                    "alpha": 0.9,
                    "centers": [[0.0, 0.0]],
                    "radii": [1.0],
                }
            }

    monkeypatch.setattr(
        posthoc_ood,
        "_load_evaluation_module",
        lambda evaluation_file: FakeEvaluatorModule,
    )

    results = posthoc_ood.evaluate_program_on_ood_tasks(
        program_path=program_path,
        family="circle_packing_rectangle",
        ood_task_specs=resolve_rectangle_ood_task_specs("cp_rect_n25"),
        evaluation_file=REPO_ROOT / "examples" / "circle_packing_rectangle_emo_sta" / "evaluator.py",
    )

    metrics = results["cp_rect_n25"]["metrics"]
    assert metrics == {
        "sum_radii": pytest.approx(2.592),
        "target_sum_radii": pytest.approx(2.592),
        "target_ratio": pytest.approx(1.0),
        "validity": pytest.approx(1.0),
        "alpha": pytest.approx(0.9),
    }
    results_json = json.dumps(results)
    assert '"centers"' not in results_json
    assert '"radii"' not in results_json
