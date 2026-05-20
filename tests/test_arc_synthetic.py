"""Tests for the synthetic ARC multitask prototype."""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.arc_synthetic import evaluator
from examples.arc_synthetic.synthetic_tasks import get_task_spec, list_task_ids
from openevolve.multitask.config import load_multitask_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _program_source_for(task_id: str) -> str:
    if task_id == "rotate_90_cw":
        transform_expr = "np.rot90(arr, -1)"
    elif task_id == "shift_right_wrap":
        transform_expr = "np.roll(arr, 1, axis=1)"
    elif task_id == "flip_horizontal":
        transform_expr = "np.fliplr(arr)"
    else:
        raise AssertionError(f"Unexpected task id: {task_id}")

    return textwrap.dedent(
        f"""
        import numpy as np

        def _validate(grid):
            arr = np.asarray(grid, dtype=np.int32)
            if arr.ndim != 2:
                raise ValueError("grid must be 2D")
            return arr

        def transform_grid_attempt_1(grid):
            arr = _validate(grid)
            return {transform_expr}.astype(np.int32)

        def transform_grid_attempt_2(grid):
            arr = _validate(grid)
            return {transform_expr}.astype(np.int32)
        """
    )


class TestArcSynthetic(unittest.TestCase):
    def test_task_catalog_contains_expected_tasks(self):
        self.assertEqual(
            list_task_ids(),
            ["rotate_90_cw", "shift_right_wrap", "flip_horizontal"],
        )

    def test_task_catalog_has_train_and_heldout_examples(self):
        for task_id in list_task_ids():
            spec = get_task_spec(task_id)
            self.assertEqual(len(spec.train_examples), 3)
            self.assertEqual(len(spec.heldout_examples), 1)

    def test_evaluator_scores_perfect_program_on_each_task(self):
        for task_id in list_task_ids():
            with self.subTest(task_id=task_id):
                with tempfile.TemporaryDirectory() as tmpdir:
                    program_path = Path(tmpdir) / "candidate.py"
                    program_path.write_text(_program_source_for(task_id), encoding="utf-8")

                    with patch.dict(os.environ, {"ARC_SYNTHETIC_TASK_ID": task_id}, clear=False):
                        result = evaluator.evaluate(str(program_path))

                    self.assertEqual(result.metrics["runs_successfully"], 1.0)
                    self.assertEqual(result.metrics["combined_score"], 1.0)
                    self.assertEqual(result.metrics["heldout_score"], 1.0)
                    self.assertEqual(result.metrics["train_combined_score"], 1.0)
                    self.assertEqual(result.metrics["heldout_combined_score"], 1.0)

    def test_multitask_config_loads_arc_synthetic_tasks(self):
        config_path = REPO_ROOT / "multi_task_evolve" / "multitask_arc_synthetic_rotate_shift_flip.yaml"
        config = load_multitask_config(config_path)

        self.assertEqual(
            [task.name for task in config.tasks],
            ["arc_rotate_90_cw", "arc_shift_right_wrap", "arc_flip_horizontal"],
        )
        self.assertEqual(
            [task.env.get("ARC_SYNTHETIC_TASK_ID") for task in config.tasks],
            ["rotate_90_cw", "shift_right_wrap", "flip_horizontal"],
        )


if __name__ == "__main__":
    unittest.main()
