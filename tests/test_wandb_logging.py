"""Focused tests for optional W&B tracking integration."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import yaml

from openevolve.config import Config, WandbConfig
from openevolve.controller import OpenEvolve
from openevolve.database import Program
from openevolve.multitask.config import load_multitask_config
from openevolve.multitask.controller import MultiTaskOpenEvolve, TaskIterationResult
from openevolve.utils.wandb_logger import create_wandb_logger


class TestWandbRunLogger(unittest.TestCase):
    def test_logger_resolves_edit_mode_placeholder_in_run_name(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                return None

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(
            WandbConfig(enabled=True, name="demo-{model}-{edit_mode}"),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={
                    "llm": {"models": [{"name": "gemini-2.5-pro"}]},
                    "diff_based_evolution": False,
                },
                metadata={},
                step_metric="iteration",
            )

        self.assertEqual(fake_wandb.init.call_args.kwargs["name"], "demo-gemini-2.5-pro-full")

    def test_logger_resolves_foreign_scores_placeholder_in_run_name(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                return None

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(
            WandbConfig(enabled=True, name="demo-{model}-{edit_mode}-{foreign_scores}"),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload={
                    "base_config": {
                        "llm": {"models": [{"name": "gemini-2.5-pro"}]},
                        "diff_based_evolution": True,
                    },
                    "multitask": {
                        "foreign_inspirations": {"include_scores": False},
                        "tasks": [{"name": "task_a"}],
                    },
                },
                metadata={},
                step_metric="multitask/global_iteration",
            )

        self.assertEqual(
            fake_wandb.init.call_args.kwargs["name"],
            "demo-gemini-2.5-pro-diff-noscores",
        )

    def test_logger_resolves_prompt_budget_placeholder_in_run_name(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                return None

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(
            WandbConfig(enabled=True, name="demo-{prompt_budget}"),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload={
                    "multitask": {
                        "foreign_inspirations": {
                            "prompt_overrides": {"num_top_programs": 1}
                        }
                    }
                },
                metadata={},
                step_metric="multitask/global_iteration",
            )

        self.assertEqual(fake_wandb.init.call_args.kwargs["name"], "demo-xferbudget")

        logger = create_wandb_logger(
            WandbConfig(enabled=True, name="demo-{prompt_budget}"),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload={
                    "multitask": {
                        "foreign_inspirations": {}
                    }
                },
                metadata={},
                step_metric="multitask/global_iteration",
            )

        self.assertEqual(fake_wandb.init.call_args.kwargs["name"], "demo-basebudget")

    def test_logger_resolves_reward_mode_placeholder_in_run_name(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                return None

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(
            WandbConfig(enabled=True, name="demo-{reward_mode}"),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload={
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "online_bandit",
                        }
                    }
                },
                metadata={},
                step_metric="multitask/global_iteration",
            )

        self.assertEqual(fake_wandb.init.call_args.kwargs["name"], "demo-sparse")

        logger = create_wandb_logger(
            WandbConfig(enabled=True, name="demo-{reward_mode}"),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload={
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "online_bandit",
                            "reward_mode": "rich",
                        }
                    }
                },
                metadata={},
                step_metric="multitask/global_iteration",
            )

        self.assertEqual(fake_wandb.init.call_args.kwargs["name"], "demo-rich")

    def test_disabled_logger_never_imports_wandb(self):
        logger = create_wandb_logger(WandbConfig(enabled=False), tempfile.mkdtemp())

        with patch("openevolve.utils.wandb_logger.importlib.import_module") as mock_import:
            logger.init_run(run_mode="single-task", config_payload={}, metadata={}, step_metric="iteration")

        mock_import.assert_not_called()
        self.assertFalse(logger.enabled)

        # No-op methods should remain safe when disabled.
        logger.log_metrics({"iteration": 1}, step=1)
        logger.update_summary({"best_fitness": 1.0})
        logger.finish()

    def test_missing_wandb_package_is_non_fatal(self):
        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            side_effect=ImportError("wandb not installed"),
        ):
            logger.init_run(run_mode="single-task", config_payload={}, metadata={}, step_metric="iteration")

        self.assertFalse(logger.enabled)
        logger.log_metrics({"iteration": 1}, step=1)
        logger.finish()

    def test_logger_namespaces_metrics_summary_and_artifacts_for_shared_run(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeArtifact:
            def __init__(self, name, type=None, metadata=None):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files = []

            def add_file(self, path, name=None):
                self.files.append((path, name))

        class FakeRun:
            def __init__(self):
                self.id = "shared-run"
                self.logged = []
                self.artifacts = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                self.artifacts.append((artifact, aliases))

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.side_effect = lambda name, type=None, metadata=None: FakeArtifact(
            name, type=type, metadata=metadata
        )

        logger = create_wandb_logger(
            WandbConfig(
                enabled=True,
                run_id="mtsts-123",
                resume="allow",
                allow_val_change=True,
                namespace="adaptation/task_a",
            ),
            tempfile.mkdtemp(),
        )

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={"phase": "adaptation"},
                step_metric="iteration",
            )

        init_kwargs = fake_wandb.init.call_args.kwargs
        self.assertEqual(init_kwargs["id"], "mtsts-123")
        self.assertEqual(init_kwargs["resume"], "allow")
        self.assertTrue(init_kwargs["allow_val_change"])

        fake_wandb.define_metric.assert_any_call("adaptation/task_a/iteration")
        fake_wandb.define_metric.assert_any_call(
            "adaptation/task_a/*",
            step_metric="adaptation/task_a/iteration",
        )
        self.assertNotIn(
            call("*", step_metric="adaptation/task_a/iteration"),
            fake_wandb.define_metric.call_args_list,
        )

        logger.log_metrics({"iteration": 2, "combined_score": 0.7}, step=2)
        payload = fake_run.logged[-1][0]
        self.assertEqual(payload["adaptation/task_a/iteration"], 2)
        self.assertEqual(payload["adaptation/task_a/combined_score"], 0.7)
        self.assertEqual(payload["adaptation/task_a/best_combined_score_so_far"], 0.7)

        logger.update_summary({"best_fitness": 0.7})
        self.assertEqual(fake_run.summary["adaptation/task_a/best_fitness"], 0.7)
        self.assertEqual(
            fake_run.summary["adaptation/task_a/run_metadata"]["phase"],
            "adaptation",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            program_path = Path(tmpdir) / "best_program.py"
            program_path.write_text("print('hi')\n")
            logger.log_best_program_artifact(str(program_path), metadata={"iteration": 2})

        artifact, aliases = fake_run.artifacts[0]
        self.assertIn("adaptation-task_a-best-program", artifact.name)
        self.assertEqual(aliases, ["latest", "iteration-2"])

    def test_out_of_order_iteration_logs_rely_on_iteration_metric_not_wandb_step(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        logger.log_metrics({"iteration": 2, "best_fitness_so_far": 0.8}, step=2)
        logger.log_metrics({"iteration": 1, "best_fitness_so_far": 0.5}, step=1)

        self.assertEqual(fake_run.logged[0][0]["iteration"], 2)
        self.assertEqual(fake_run.logged[1][0]["iteration"], 1)
        self.assertIsNone(fake_run.logged[0][1])
        self.assertIsNone(fake_run.logged[1][1])

    def test_logger_adds_best_so_far_metrics_for_evaluation_scores(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        logger.log_metrics(
            {
                "iteration": 1,
                "r2": 0.4,
                "nmse": 0.8,
                "combined_score": 0.55,
                "iteration_time_sec": 1.2,
                "evaluation_success": 1,
            },
            step=1,
        )
        logger.log_metrics(
            {
                "iteration": 2,
                "r2": 0.3,
                "nmse": 0.9,
                "combined_score": 0.6,
                "iteration_time_sec": 0.9,
                "evaluation_success": 1,
            },
            step=2,
        )

        first_payload = fake_run.logged[0][0]
        second_payload = fake_run.logged[1][0]

        self.assertEqual(first_payload["best_r2_so_far"], 0.4)
        self.assertEqual(first_payload["best_nmse_so_far"], 0.8)
        self.assertEqual(first_payload["best_combined_score_so_far"], 0.55)
        self.assertNotIn("best_iteration_time_sec_so_far", first_payload)
        self.assertNotIn("best_evaluation_success_so_far", first_payload)

        self.assertEqual(second_payload["best_r2_so_far"], 0.4)
        self.assertEqual(second_payload["best_nmse_so_far"], 0.8)
        self.assertEqual(second_payload["best_combined_score_so_far"], 0.6)

        fake_wandb.define_metric.assert_any_call("best_r2_so_far", summary="max")
        fake_wandb.define_metric.assert_any_call("best_nmse_so_far", summary="min")
        fake_wandb.define_metric.assert_any_call("best_combined_score_so_far", summary="max")

    def test_best_r2_so_far_is_clipped_for_visualization(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload={"multitask": {"tasks": [{"name": "vocab_scaling_law"}]}},
                metadata={},
                step_metric="multitask/global_iteration",
            )

        logger.log_metrics(
            {
                "multitask/global_iteration": 0,
                "task/vocab_scaling_law/r2": -6.0e21,
            },
            step=0,
        )
        logger.log_metrics(
            {
                "multitask/global_iteration": 1,
                "task/vocab_scaling_law/r2": 0.4,
            },
            step=1,
        )

        first_payload = fake_run.logged[0][0]
        second_payload = fake_run.logged[1][0]

        self.assertEqual(first_payload["task/vocab_scaling_law/best_r2_so_far"], -1.0)
        self.assertEqual(second_payload["task/vocab_scaling_law/best_r2_so_far"], 0.4)

    def test_finish_logs_best_so_far_plots_for_tracked_metrics(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        logger.log_metrics({"iteration": 2, "r2": 0.9, "nmse": 0.6}, step=2)
        logger.log_metrics({"iteration": 1, "r2": 0.5, "nmse": 0.7}, step=1)
        logger.finish()

        plot_calls = {
            call.kwargs["title"]: call.kwargs for call in fake_wandb.plot.line_series.call_args_list
        }
        self.assertEqual(plot_calls["best_r2_so_far over time"]["xs"], [1, 2])
        self.assertEqual(plot_calls["best_r2_so_far over time"]["ys"], [[0.5, 0.9]])
        self.assertEqual(plot_calls["best_r2_so_far over time"]["keys"], ["best_r2_so_far"])
        self.assertEqual(plot_calls["best_nmse_so_far over time"]["xs"], [1, 2])
        self.assertEqual(plot_calls["best_nmse_so_far over time"]["ys"], [[0.7, 0.6]])
        self.assertEqual(plot_calls["best_nmse_so_far over time"]["keys"], ["best_nmse_so_far"])

    def test_finish_clips_best_r2_custom_plot_for_visualization(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        logger.log_metrics({"iteration": 0, "r2": -6.0e21}, step=0)
        logger.log_metrics({"iteration": 1, "r2": 0.4}, step=1)
        logger.finish()

        plot_calls = {
            call.kwargs["title"]: call.kwargs for call in fake_wandb.plot.line_series.call_args_list
        }
        self.assertEqual(plot_calls["best_r2_so_far over time"]["xs"], [0, 1])
        self.assertEqual(plot_calls["best_r2_so_far over time"]["ys"], [[-1.0, 0.4]])
        self.assertEqual(plot_calls["best_r2_so_far over time"]["keys"], ["best_r2_so_far"])

    def test_finish_clips_extreme_min_metric_plot_outlier(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        logger.log_metrics({"iteration": 0, "nmae": 6.0e10}, step=0)
        logger.log_metrics({"iteration": 1, "nmae": 0.02}, step=1)
        logger.log_metrics({"iteration": 2, "nmae": 0.01}, step=2)
        logger.finish()

        plot_calls = {
            call.kwargs["title"]: call.kwargs for call in fake_wandb.plot.line_series.call_args_list
        }
        self.assertEqual(plot_calls["best_nmae_so_far over time"]["xs"], [0, 1, 2])
        self.assertEqual(plot_calls["best_nmae_so_far over time"]["ys"], [[0.02, 0.02, 0.01]])
        self.assertEqual(plot_calls["best_nmae_so_far over time"]["keys"], ["best_nmae_so_far"])

    def test_relative_best_program_artifact_does_not_disable_logging(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeArtifact:
            def __init__(self, name, type=None, metadata=None):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files = []

            def add_file(self, path, name=None):
                self.files.append((path, name))

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.artifacts = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                self.artifacts.append((artifact, aliases))

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.side_effect = lambda name, type=None, metadata=None: FakeArtifact(
            name, type=type, metadata=metadata
        )

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            best_dir = Path(tmpdir) / "best"
            best_dir.mkdir()
            program_path = best_dir / "best_program.py"
            info_path = best_dir / "best_program_info.json"
            program_path.write_text("print('hi')\n")
            info_path.write_text("{}\n")
            relative_program_path = os.path.relpath(program_path, start=os.getcwd())

            logger.log_best_program_artifact(relative_program_path, metadata={"iteration": 3})
            logger.log_metrics({"iteration": 4, "best_fitness_so_far": 0.9}, step=4)

        self.assertTrue(logger.enabled)
        self.assertEqual(len(fake_run.artifacts), 1)
        self.assertEqual(fake_run.logged[-1][0]["iteration"], 4)

    def test_artifact_failure_does_not_disable_metrics_logging(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeArtifact:
            def add_file(self, path, name=None):
                raise RuntimeError("artifact add failed")

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.logged = []
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                self.logged.append((dict(payload), step))

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = FakeArtifact()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload={},
                metadata={},
                step_metric="iteration",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            program_path = Path(tmpdir) / "best_program.py"
            program_path.write_text("print('hi')\n")
            logger.log_best_program_artifact(str(program_path), metadata={"iteration": 2})

        logger.log_metrics({"iteration": 3, "best_fitness_so_far": 0.7}, step=3)

        self.assertTrue(logger.enabled)
        self.assertEqual(fake_run.logged[-1][0]["iteration"], 3)

    def test_name_template_includes_model_name(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                return None

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        config = Config()
        config.llm.primary_model = "gemini-2.5-pro"
        config.wandb = WandbConfig(enabled=True, name="vocab_scaling_law-{model}")

        logger = create_wandb_logger(config.wandb, tempfile.mkdtemp())

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="single-task",
                config_payload=config,
                metadata={},
                step_metric="iteration",
            )

        self.assertEqual(fake_wandb.init.call_args.kwargs["name"], "vocab_scaling_law-gemini-2.5-pro")

    def test_multitask_init_defines_per_task_metrics_without_invalid_glob(self):
        class FakeSummary(dict):
            def update(self, data):
                super().update(data)

        class FakeRun:
            def __init__(self):
                self.id = "fake-run"
                self.summary = FakeSummary()

            def log(self, payload, step=None):
                return None

            def finish(self):
                return None

            def log_code(self, root=None):
                return None

            def log_artifact(self, artifact, aliases=None):
                return None

        fake_run = FakeRun()
        fake_wandb = MagicMock()
        fake_wandb.init.return_value = fake_run
        fake_wandb.plot.line_series.return_value = {"ok": True}
        fake_wandb.Artifact.return_value = MagicMock()

        logger = create_wandb_logger(WandbConfig(enabled=True), tempfile.mkdtemp())
        config_payload = {
            "multitask": {
                "tasks": [
                    {"name": "data_constrained_scaling_law"},
                    {"name": "vocab_scaling_law"},
                ]
            }
        }

        with patch(
            "openevolve.utils.wandb_logger.importlib.import_module",
            return_value=fake_wandb,
        ):
            logger.init_run(
                run_mode="multitask",
                config_payload=config_payload,
                metadata={},
                step_metric="multitask/global_iteration",
            )

        define_metric_calls = [call.args[0] for call in fake_wandb.define_metric.call_args_list]
        self.assertIn("task/data_constrained_scaling_law/best_fitness", define_metric_calls)
        self.assertIn("task/vocab_scaling_law/best_fitness", define_metric_calls)
        self.assertNotIn("task/*/best_fitness", define_metric_calls)
        self.assertTrue(logger.enabled)


class TestSingleTaskWandbIntegration(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        self.program_path = self.tmp_path / "program.py"
        self.program_path.write_text("def solve():\n    return 1\n")
        self.evaluator_path = self.tmp_path / "evaluator.py"
        self.evaluator_path.write_text(
            "def evaluate(path):\n"
            "    return {'combined_score': 1.0, 'accuracy': 0.5}\n"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_wandb_init_called_once_per_run(self):
        mock_logger = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 1.0})

        async def run_test():
            with patch("openevolve.controller.create_wandb_logger", return_value=mock_logger):
                with patch("openevolve.controller.Evaluator", return_value=mock_evaluator):
                    with patch("openevolve.controller.ProcessParallelController") as mock_parallel_cls:
                        mock_parallel = MagicMock()
                        mock_parallel.run_evolution = AsyncMock(return_value=None)
                        mock_parallel.start.return_value = None
                        mock_parallel.stop.return_value = None
                        mock_parallel.shutdown_event.is_set.return_value = False
                        mock_parallel_cls.return_value = mock_parallel

                        controller = OpenEvolve(
                            initial_program_path=str(self.program_path),
                            evaluation_file=str(self.evaluator_path),
                            config=Config(),
                            output_dir=str(self.tmp_path / "output"),
                        )
                        await controller.run(iterations=0)

        asyncio.run(run_test())

        mock_logger.init_run.assert_called_once()
        mock_logger.finish.assert_called_once()

    def test_single_task_iteration_event_logs_expected_metrics_and_artifact(self):
        controller = OpenEvolve(
            initial_program_path=str(self.program_path),
            evaluation_file=str(self.evaluator_path),
            config=Config(),
            output_dir=str(self.tmp_path / "output_event"),
        )
        controller.wandb_logger = MagicMock()

        parent = Program(
            id="parent",
            code="def solve():\n    return 1\n",
            metrics={"combined_score": 0.2, "accuracy": 0.2},
            iteration_found=0,
        )
        child = Program(
            id="child",
            code="def solve():\n    return 2\n",
            metrics={"combined_score": 0.8, "accuracy": 0.9},
            parent_id="parent",
            iteration_found=1,
        )
        controller.database.add(parent, iteration=0)
        controller.database.add(child, iteration=1)
        controller.database.best_program_id = child.id
        controller._best_fitness_so_far = 0.2

        controller._handle_parallel_iteration_event(
            {
                "event": "success",
                "iteration": 1,
                "child_program": child,
                "parent_program": parent,
                "iteration_time_sec": 2.5,
                "generation_time_sec": 0.7,
                "evaluation_time_sec": 1.1,
                "is_new_best": True,
            }
        )

        log_call = controller.wandb_logger.log_metrics.call_args
        logged_metrics = log_call.args[0]
        self.assertEqual(logged_metrics["iteration"], 1)
        self.assertEqual(logged_metrics["evaluation_success"], 1)
        self.assertEqual(logged_metrics["current_fitness"], 0.8)
        self.assertEqual(logged_metrics["best_fitness_so_far"], 0.8)
        self.assertAlmostEqual(logged_metrics["delta_best_fitness"], 0.6)
        self.assertEqual(logged_metrics["evaluation_time_sec"], 1.1)
        self.assertEqual(logged_metrics["generation_time_sec"], 0.7)
        self.assertEqual(logged_metrics["combined_score"], 0.8)
        self.assertEqual(logged_metrics["accuracy"], 0.9)

        controller.wandb_logger.log_best_program_artifact.assert_called_once()


class TestMultitaskWandbIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        (self.tmp_path / "init.py").write_text("def solve():\n    return 1\n")
        (self.tmp_path / "evaluate.py").write_text(
            "def evaluate(path):\n"
            "    code = open(path, 'r').read()\n"
            "    return {'combined_score': 1.0 if 'return 2' in code else 0.1, 'accuracy': 0.7}\n"
        )
        config_path = self.tmp_path / "multitask.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "multitask": {
                        "output_dir": str(self.tmp_path / "outputs"),
                        "max_global_iterations": 1,
                        "checkpoint_interval": 1,
                        "foreign_inspirations": {"enabled": True, "every_n_task_iterations": 1},
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "init.py",
                                "evaluation_file": "evaluate.py",
                                "related_tasks": [{"source_task": "task_b"}],
                            },
                            {
                                "name": "task_b",
                                "initial_program": "init.py",
                                "evaluation_file": "evaluate.py",
                            },
                        ],
                    }
                }
            )
        )
        self.multitask_config = load_multitask_config(config_path)

    def tearDown(self):
        self.tempdir.cleanup()

    async def test_multitask_step_logs_namespaced_metrics(self):
        controller = MultiTaskOpenEvolve(self.multitask_config)
        controller.wandb_logger = MagicMock()

        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)
            controller._task_best_fitness[task_state.task_name] = controller._get_task_best_fitness(
                task_state
            )

        task_a = controller.task_by_name["task_a"]
        controller._scheduler_counts["task_a"] = 1

        child = Program(
            id="task-a-child",
            code="def solve():\n    return 2\n",
            metrics={"combined_score": 1.0, "accuracy": 0.95},
            parent_id=task_a.database.best_program_id,
            iteration_found=1,
        )
        task_a.database.add(child, iteration=1, target_island=0)
        task_a.database.best_program_id = child.id

        result = TaskIterationResult(
            task_name="task_a",
            local_iteration=1,
            success=True,
            child_program=child,
            generation_time_sec=0.4,
            evaluation_time_sec=1.2,
            iteration_time_sec=1.8,
            foreign_inspiration_sources=["task_b"],
        )

        controller._log_multitask_step(global_iteration=1, task_state=task_a, result=result)

        logged_metrics = controller.wandb_logger.log_metrics.call_args.args[0]
        self.assertEqual(logged_metrics["multitask/global_iteration"], 1)
        self.assertEqual(logged_metrics["multitask/selected_task"], "task_a")
        self.assertEqual(logged_metrics["task_name"], "task_a")
        self.assertEqual(logged_metrics["task_local_iteration"], 1)
        self.assertEqual(logged_metrics["multitask/foreign_inspirations_used"], 1)
        self.assertEqual(logged_metrics["multitask/num_foreign_inspirations"], 1)
        self.assertEqual(logged_metrics["multitask/foreign_inspiration_sources"], "task_b")
        self.assertEqual(logged_metrics["task/task_a/current_fitness"], 1.0)
        self.assertEqual(logged_metrics["task/task_a/best_fitness"], 1.0)
        self.assertEqual(logged_metrics["task/task_a/combined_score"], 1.0)
        self.assertEqual(logged_metrics["task/task_a/accuracy"], 0.95)

        controller.wandb_logger.log_best_program_artifact.assert_called_once()

    async def test_multitask_online_bandit_step_logs_reward_metrics(self):
        config_path = self.tmp_path / "multitask_online_bandit.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "multitask": {
                        "output_dir": str(self.tmp_path / "outputs_online_bandit"),
                        "max_global_iterations": 1,
                        "checkpoint_interval": 1,
                        "foreign_inspirations": {
                            "enabled": True,
                            "trigger_mode": "online_bandit",
                            "warmup_task_iterations": 0,
                            "stagnation_patience": 1,
                            "transfer_cooldown": 0,
                            "max_related_tasks": 1,
                            "reward_mode": "rich",
                            "reward_window": 2,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "init.py",
                                "evaluation_file": "evaluate.py",
                                "related_tasks": [{"source_task": "task_b"}],
                            },
                            {
                                "name": "task_b",
                                "initial_program": "init.py",
                                "evaluation_file": "evaluate.py",
                            },
                        ],
                    }
                }
            )
        )
        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        controller.wandb_logger = MagicMock()

        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)
            controller._task_best_fitness[task_state.task_name] = controller._get_task_best_fitness(
                task_state
            )

        task_a = controller.task_by_name["task_a"]
        task_a.recent_child_fitness_history = [0.9, 0.1, 0.2]
        child = Program(
            id="task-a-bandit-child",
            code="def solve():\n    return 2\n",
            metrics={"combined_score": 0.18, "accuracy": 0.95},
            parent_id=task_a.database.best_program_id,
            iteration_found=1,
        )
        task_a.database.add(child, iteration=1, target_island=0)
        task_a.database.best_program_id = child.id
        controller._task_best_fitness["task_a"] = 0.18

        result = TaskIterationResult(
            task_name="task_a",
            local_iteration=1,
            success=True,
            child_program=child,
            foreign_inspiration_sources=["task_b"],
            foreign_transfer_trigger_reason="online_bandit",
            chosen_transfer_arm="task_b",
        )

        controller._log_multitask_step(global_iteration=1, task_state=task_a, result=result)

        logged_metrics = controller.wandb_logger.log_metrics.call_args.args[0]
        self.assertEqual(logged_metrics["multitask/bandit_reward_mode"], "rich")
        self.assertAlmostEqual(logged_metrics["multitask/bandit_reward_child_fitness"], 0.18)
        self.assertAlmostEqual(logged_metrics["multitask/bandit_reward_baseline"], 0.15)
        self.assertEqual(logged_metrics["task/task_a/bandit_reward_mode"], "rich")
        self.assertAlmostEqual(logged_metrics["task/task_a/bandit_reward_child_fitness"], 0.18)
        self.assertAlmostEqual(logged_metrics["task/task_a/bandit_reward_baseline"], 0.15)
        self.assertEqual(logged_metrics["task/task_a/foreign_transfer_reward"], 1)

    async def test_multitask_run_initializes_wandb_once(self):
        mock_logger = MagicMock()

        with patch("openevolve.multitask.controller.create_wandb_logger", return_value=mock_logger):
            controller = MultiTaskOpenEvolve(self.multitask_config)

        for task_state in controller.tasks:
            task_state.llm_ensemble.generate_with_context = AsyncMock(
                return_value=(
                    "<<<<<<< SEARCH\n"
                    "    return 1\n"
                    "=======\n"
                    "    return 2\n"
                    ">>>>>>> REPLACE"
                )
            )

        await controller.run(max_global_iterations=1)

        mock_logger.init_run.assert_called_once()
        mock_logger.finish.assert_called_once()
