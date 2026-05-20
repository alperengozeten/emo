"""Tests for the multitask controller."""

import json
import shutil
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import yaml

from openevolve.config import Config
from openevolve.database import Program
from openevolve.multitask.config import TaskConfig, load_multitask_config
from openevolve.multitask.controller import (
    MultiTaskOpenEvolve,
    ParallelWaveMultiTaskOpenEvolve,
    TaskIterationResult,
    create_multitask_controller,
)
from openevolve.prompt.sampler import PromptSampler


class TestMultiTaskOpenEvolve(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)

        (self.tmp_path / "init.py").write_text("def solve():\n    return 1\n")
        (self.tmp_path / "evaluate.py").write_text(
            "\n".join(
                [
                    "import os",
                    "",
                    "def evaluate(path):",
                    "    code = open(path, 'r').read()",
                    "    expected = os.environ['EXPECTED_RETURN']",
                    "    score = 1.0 if f'return {expected}' in code else 0.0",
                    "    return {'combined_score': score}",
                ]
            )
        )
        (self.tmp_path / "evaluate_with_artifacts.py").write_text(
            "\n".join(
                [
                    "import os",
                    "",
                    "from openevolve.evaluation_result import EvaluationResult",
                    "",
                    "def evaluate(path):",
                    "    code = open(path, 'r').read()",
                    "    expected = os.environ['EXPECTED_RETURN']",
                    "    score = 1.0 if f'return {expected}' in code else 0.0",
                    "    return EvaluationResult(",
                    "        metrics={'combined_score': score},",
                    "        artifacts={",
                    "            'large_log': f'artifact_for_{expected}_' + ('x' * (40 * 1024)),",
                    "        },",
                    "    )",
                ]
            )
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_config(self, config_dict):
        config_path = self.tmp_path / "multitask.yaml"
        config_path.write_text(yaml.safe_dump(config_dict))
        return config_path

    def _seed_task_programs(self, task_state, count=5):
        parent = task_state.database.get_best_program()
        self.assertIsNotNone(parent)

        for index in range(count):
            program = Program(
                id=f"{task_state.task_name}-seed-{index}",
                code=(
                    "def solve():\n"
                    f"    return {index + 10}\n"
                    + (f"# seed {index}\n" * (index + 1))
                ),
                language="python",
                parent_id=parent.id,
                generation=parent.generation + 1,
                iteration_found=index + 1,
                metrics={"combined_score": float(count - index)},
                metadata={
                    "changes": f"seed {index}",
                    "parent_metrics": parent.metrics,
                },
            )
            task_state.database.add(program, iteration=index + 1, target_island=0)

    async def test_controller_factory_keeps_sequential_mode_as_default(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_factory"),
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = create_multitask_controller(load_multitask_config(config_path))

        self.assertIs(type(controller), MultiTaskOpenEvolve)
        self.assertNotIsInstance(controller, ParallelWaveMultiTaskOpenEvolve)

    def test_relative_root_log_dir_is_rebased_under_multitask_output_dir(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_root_logs"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "base_config": {
                        "log_dir": "root_logs",
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))

        expected_log_dir = str((self.tmp_path / "outputs_root_logs" / "root_logs").resolve())
        self.assertEqual(controller.root_log_dir, expected_log_dir)
        self.assertEqual(controller.base_config.log_dir, "root_logs")
        self.assertTrue(Path(expected_log_dir).exists())
        self.assertTrue(list(Path(expected_log_dir).glob("openevolve_multitask_*.log")))

    def test_relative_task_output_paths_are_rebased_per_task(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_paths"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "base_config": {
                        "log_dir": "shared_logs",
                        "database": {"artifacts_base_path": "shared_artifacts"},
                        "evolution_trace": {
                            "enabled": True,
                            "output_path": "trace.jsonl",
                        },
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))

        task_a = controller.task_by_name["task_a"]
        task_b = controller.task_by_name["task_b"]

        self.assertEqual(
            task_a.config.log_dir,
            str((Path(task_a.output_dir) / "shared_logs").resolve()),
        )
        self.assertEqual(
            task_b.config.log_dir,
            str((Path(task_b.output_dir) / "shared_logs").resolve()),
        )
        self.assertTrue(
            list(Path(task_a.config.log_dir).glob("openevolve_task_task_a_*.log"))
        )
        self.assertTrue(
            list(Path(task_b.config.log_dir).glob("openevolve_task_task_b_*.log"))
        )
        self.assertEqual(
            task_a.config.database.artifacts_base_path,
            str((Path(task_a.output_dir) / "shared_artifacts").resolve()),
        )
        self.assertEqual(
            task_b.config.database.artifacts_base_path,
            str((Path(task_b.output_dir) / "shared_artifacts").resolve()),
        )
        self.assertEqual(
            str(task_a.evolution_tracer.output_path),
            str((Path(task_a.output_dir) / "trace.jsonl").resolve()),
        )
        self.assertEqual(
            str(task_b.evolution_tracer.output_path),
            str((Path(task_b.output_dir) / "trace.jsonl").resolve()),
        )
        self.assertNotEqual(task_a.config.log_dir, task_b.config.log_dir)
        self.assertNotEqual(
            task_a.config.database.artifacts_base_path,
            task_b.config.database.artifacts_base_path,
        )
        self.assertNotEqual(
            str(task_a.evolution_tracer.output_path),
            str(task_b.evolution_tracer.output_path),
        )

        controller._close_tracers()

    def test_duplicate_absolute_task_output_paths_are_rejected(self):
        shared_log_dir = str((self.tmp_path / "shared_logs_abs").resolve())
        shared_artifacts_dir = str((self.tmp_path / "shared_artifacts_abs").resolve())
        shared_trace_path = str((self.tmp_path / "shared_trace.jsonl").resolve())

        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_collision"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "base_config": {
                        "log_dir": shared_log_dir,
                        "database": {"artifacts_base_path": shared_artifacts_dir},
                        "evolution_trace": {
                            "enabled": True,
                            "output_path": shared_trace_path,
                        },
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "Multitask output path collision"):
            MultiTaskOpenEvolve(load_multitask_config(config_path))

    async def test_run_keeps_task_state_isolated_with_shared_evaluator(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 2,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))

        for task_state in controller.tasks:
            target_value = task_state.env["EXPECTED_RETURN"]
            task_state.llm_ensemble.generate_with_context = AsyncMock(
                return_value=(
                    "<<<<<<< SEARCH\n"
                    "    return 1\n"
                    "=======\n"
                    f"    return {target_value}\n"
                    ">>>>>>> REPLACE"
                )
            )

        best_programs = await controller.run()

        self.assertEqual(best_programs["task_a"].metrics["combined_score"], 1.0)
        self.assertEqual(best_programs["task_b"].metrics["combined_score"], 1.0)
        self.assertIn("return 2", best_programs["task_a"].code)
        self.assertIn("return 3", best_programs["task_b"].code)

        root_checkpoint = Path(controller.output_dir) / "checkpoints" / "checkpoint_2"
        self.assertTrue(root_checkpoint.exists())
        for task_state in controller.tasks:
            task_checkpoint = root_checkpoint / "tasks" / task_state.task_name
            self.assertTrue(task_checkpoint.exists())
            self.assertTrue((Path(task_state.output_dir) / "best").exists())

    async def test_foreign_inspiration_selection_respects_order_and_limits(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_order"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "every_n_task_iterations": 1,
                        "warmup_task_iterations": 0,
                        "max_related_tasks": 1,
                        "top_programs_per_related_task": 1,
                        "include_optional_relation_text": True,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [
                                {
                                    "source_task": "task_b",
                                    "prompt_context": "Use task_b first.",
                                },
                                {
                                    "source_task": "task_c",
                                    "prompt_context": "Use task_c second.",
                                },
                            ],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_c",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        selected = controller._select_foreign_inspirations(task_a)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_task"], "task_b")
        self.assertEqual(selected[0]["prompt_context"], "Use task_b first.")
        self.assertTrue(selected[0]["include_scores"])
        self.assertEqual(len(selected[0]["programs"]), 1)

    async def test_foreign_inspiration_selection_can_disable_scores(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_no_scores"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "every_n_task_iterations": 1,
                        "warmup_task_iterations": 0,
                        "include_scores": False,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [
                                {
                                    "source_task": "task_b",
                                    "prompt_context": "Use task_b ideas.",
                                }
                            ],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        selected = controller._select_foreign_inspirations(task_a)

        self.assertEqual(len(selected), 1)
        self.assertFalse(selected[0]["include_scores"])

    async def test_foreign_inspiration_schedule_uses_task_local_iterations(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_schedule"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "every_n_task_iterations": 3,
                        "warmup_task_iterations": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]

        task_a.local_iteration = 0
        self.assertFalse(controller._should_include_foreign_inspirations(task_a))

        task_a.local_iteration = 1
        self.assertTrue(controller._should_include_foreign_inspirations(task_a))

        task_a.local_iteration = 2
        self.assertFalse(controller._should_include_foreign_inspirations(task_a))

        task_a.local_iteration = 4
        self.assertTrue(controller._should_include_foreign_inspirations(task_a))

    async def test_prepare_task_iteration_request_keeps_base_prompt_budget_without_overrides(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_base_prompt_budget"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "every_n_task_iterations": 1,
                        "warmup_task_iterations": 0,
                    },
                    "base_config": {
                        "prompt": {
                            "num_top_programs": 3,
                            "num_diverse_programs": 2,
                            "num_local_inspirations": 2,
                        }
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        self._seed_task_programs(task_a)

        request = controller._prepare_task_iteration_request(task_a)

        self.assertEqual(len(request.previous_programs), 3)
        self.assertEqual(len(request.top_programs), 5)
        self.assertEqual(len(request.inspirations), 2)
        self.assertEqual(
            [source["source_task"] for source in request.foreign_inspirations],
            ["task_b"],
        )
        self.assertIsNone(request.effective_prompt_config)

    async def test_prompt_overrides_do_not_apply_without_actual_foreign_inspirations(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_no_foreign_prompt_override"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "every_n_task_iterations": 1,
                        "warmup_task_iterations": 0,
                        "prompt_overrides": {
                            "num_top_programs": 1,
                            "num_diverse_programs": 0,
                            "num_local_inspirations": 1,
                        },
                    },
                    "base_config": {
                        "prompt": {
                            "num_top_programs": 3,
                            "num_diverse_programs": 2,
                            "num_local_inspirations": 2,
                        }
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        await controller._ensure_initial_program(task_a)
        self._seed_task_programs(task_a)
        task_a.llm_ensemble.generate_with_context = AsyncMock(return_value=None)

        captured = {}

        def build_prompt_spy(prompt_sampler_self, *args, **kwargs):
            captured["sampler_num_top_programs"] = prompt_sampler_self.config.num_top_programs
            captured["sampler_num_diverse_programs"] = (
                prompt_sampler_self.config.num_diverse_programs
            )
            captured["num_previous_programs"] = len(kwargs["previous_programs"])
            captured["num_top_programs"] = len(kwargs["top_programs"])
            captured["num_local_inspirations"] = len(kwargs["inspirations"])
            captured["num_foreign_inspirations"] = len(kwargs["foreign_inspirations"])
            return {"system": "fake-system", "user": "fake-user"}

        with patch.object(PromptSampler, "build_prompt", new=build_prompt_spy):
            result = await controller._run_task_iteration(task_a)

        self.assertFalse(result.success)
        self.assertEqual(captured["sampler_num_top_programs"], 3)
        self.assertEqual(captured["sampler_num_diverse_programs"], 2)
        self.assertEqual(captured["num_previous_programs"], 3)
        self.assertEqual(captured["num_top_programs"], 5)
        self.assertEqual(captured["num_local_inspirations"], 2)
        self.assertEqual(captured["num_foreign_inspirations"], 0)

    async def test_sequential_prompt_overrides_apply_for_transfer_iteration_only(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_transfer_prompt_override"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "every_n_task_iterations": 1,
                        "warmup_task_iterations": 0,
                        "prompt_overrides": {
                            "num_top_programs": 1,
                            "num_diverse_programs": 0,
                            "num_local_inspirations": 1,
                        },
                    },
                    "base_config": {
                        "prompt": {
                            "num_top_programs": 3,
                            "num_diverse_programs": 2,
                            "num_local_inspirations": 2,
                        }
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        self._seed_task_programs(task_a)
        task_a.llm_ensemble.generate_with_context = AsyncMock(return_value=None)

        captured = {}

        def build_prompt_spy(prompt_sampler_self, *args, **kwargs):
            captured["sampler_num_top_programs"] = prompt_sampler_self.config.num_top_programs
            captured["sampler_num_diverse_programs"] = (
                prompt_sampler_self.config.num_diverse_programs
            )
            captured["num_previous_programs"] = len(kwargs["previous_programs"])
            captured["num_top_programs"] = len(kwargs["top_programs"])
            captured["num_local_inspirations"] = len(kwargs["inspirations"])
            captured["num_foreign_inspirations"] = len(kwargs["foreign_inspirations"])
            return {"system": "fake-system", "user": "fake-user"}

        with patch.object(PromptSampler, "build_prompt", new=build_prompt_spy):
            result = await controller._run_task_iteration(task_a)

        self.assertFalse(result.success)
        self.assertEqual(result.foreign_inspiration_sources, ["task_b"])
        self.assertEqual(captured["sampler_num_top_programs"], 1)
        self.assertEqual(captured["sampler_num_diverse_programs"], 0)
        self.assertEqual(captured["num_previous_programs"], 1)
        self.assertEqual(captured["num_top_programs"], 1)
        self.assertEqual(captured["num_local_inspirations"], 1)
        self.assertEqual(captured["num_foreign_inspirations"], 1)
        self.assertEqual(task_a.config.prompt.num_top_programs, 3)
        self.assertEqual(task_a.config.prompt.num_diverse_programs, 2)
        self.assertEqual(task_a.config.prompt.num_local_inspirations, 2)

    async def test_stagnation_trigger_mode_respects_warmup_patience_and_cooldown(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_stagnation_trigger"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "stagnation",
                        "warmup_task_iterations": 2,
                        "stagnation_patience": 2,
                        "transfer_cooldown": 2,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]

        task_a.local_iteration = 1
        task_a.no_improve_steps = 5
        self.assertIsNone(controller._get_foreign_transfer_trigger(task_a))

        task_a.local_iteration = 2
        task_a.no_improve_steps = 1
        self.assertIsNone(controller._get_foreign_transfer_trigger(task_a))

        task_a.no_improve_steps = 2
        self.assertEqual(controller._get_foreign_transfer_trigger(task_a), "stagnation")

        task_a.last_transfer_iteration = 1
        self.assertIsNone(controller._get_foreign_transfer_trigger(task_a))

        task_a.local_iteration = 3
        self.assertEqual(controller._get_foreign_transfer_trigger(task_a), "stagnation")

    async def test_online_bandit_none_is_a_valid_arm(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_none"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                        "top_programs_per_related_task": 1,
                        "min_pulls_per_arm": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.local_iteration = 1
        task_a.no_improve_steps = 1
        task_a.transfer_bandit_pulls = {"NONE": 2, "task_b": 2}

        with patch.object(
            controller,
            "_sample_task_betavariate",
            side_effect=[0.9, 0.1],
        ):
            decision = controller._get_foreign_transfer_decision(task_a)

        self.assertEqual(decision.chosen_transfer_arm, "NONE")
        self.assertEqual(decision.trigger_reason, "online_bandit")
        self.assertEqual(decision.foreign_inspirations, [])

    async def test_online_bandit_forced_exploration_runs_before_thompson_sampling(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_forced"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                        "top_programs_per_related_task": 1,
                        "min_pulls_per_arm": 2,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [
                                {"source_task": "task_b"},
                                {"source_task": "task_c"},
                            ],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                        {
                            "name": "task_c",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "4"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.local_iteration = 1
        task_a.no_improve_steps = 1
        task_a.transfer_bandit_pulls = {"NONE": 2, "task_b": 0, "task_c": 2}

        with patch.object(
            controller,
            "_sample_task_betavariate",
            side_effect=AssertionError("Thompson sampling should not run during forced exploration"),
        ):
            decision = controller._get_foreign_transfer_decision(task_a)

        self.assertEqual(decision.chosen_transfer_arm, "task_b")
        self.assertEqual(
            [source["source_task"] for source in decision.foreign_inspirations],
            ["task_b"],
        )

    async def test_online_bandit_thompson_sampling_choice_uses_mocked_samples(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_thompson"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                        "top_programs_per_related_task": 1,
                        "min_pulls_per_arm": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [
                                {"source_task": "task_b"},
                                {"source_task": "task_c"},
                            ],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                        {
                            "name": "task_c",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "4"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.local_iteration = 1
        task_a.no_improve_steps = 1
        task_a.transfer_bandit_pulls = {"NONE": 1, "task_b": 1, "task_c": 1}

        with patch.object(
            controller,
            "_sample_task_betavariate",
            side_effect=[0.1, 0.8, 0.3],
        ) as sample_mock:
            decision = controller._get_foreign_transfer_decision(task_a)

        self.assertEqual(sample_mock.call_count, 3)
        self.assertEqual(decision.chosen_transfer_arm, "task_b")
        self.assertEqual(
            [source["source_task"] for source in decision.foreign_inspirations],
            ["task_b"],
        )

    async def test_online_bandit_skips_posterior_update_when_selected_source_has_no_payload(
        self,
    ):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_empty_payload"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                        "top_programs_per_related_task": 1,
                        "min_pulls_per_arm": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.local_iteration = 1
        task_a.no_improve_steps = 1
        task_a.transfer_bandit_pulls = {"NONE": 1, "task_b": 1}

        with (
            patch.object(
                controller,
                "_sample_task_betavariate",
                side_effect=[0.1, 0.9],
            ),
            patch.object(controller, "_collect_foreign_inspirations", return_value=[]),
        ):
            decision = controller._get_foreign_transfer_decision(task_a)

        self.assertEqual(decision.trigger_reason, "online_bandit")
        self.assertIsNone(decision.chosen_transfer_arm)
        self.assertEqual(decision.foreign_inspirations, [])

        before_alpha = dict(task_a.transfer_bandit_alpha)
        before_beta = dict(task_a.transfer_bandit_beta)
        before_pulls = dict(task_a.transfer_bandit_pulls)

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.2,
            current_best=0.35,
            local_iteration=2,
            foreign_transfer_used=False,
            chosen_transfer_arm=decision.chosen_transfer_arm,
        )

        self.assertIsNone(progress_update.reward_for_chosen_arm)
        self.assertEqual(task_a.transfer_bandit_alpha, before_alpha)
        self.assertEqual(task_a.transfer_bandit_beta, before_beta)
        self.assertEqual(task_a.transfer_bandit_pulls, before_pulls)
        self.assertIsNone(task_a.last_transfer_iteration)

    async def test_online_bandit_reward_updates_only_the_chosen_arm(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_reward"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [
                                {"source_task": "task_b"},
                                {"source_task": "task_c"},
                            ],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                        {
                            "name": "task_c",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "4"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.transfer_bandit_alpha = {"NONE": 7.0, "task_b": 2.0, "task_c": 1.0}
        task_a.transfer_bandit_beta = {"NONE": 8.0, "task_b": 3.0, "task_c": 1.0}
        task_a.transfer_bandit_pulls = {"NONE": 5, "task_b": 4, "task_c": 0}

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.2,
            current_best=0.35,
            local_iteration=4,
            foreign_transfer_used=True,
            chosen_transfer_arm="task_c",
        )

        self.assertEqual(progress_update.reward_mode, "sparse")
        self.assertEqual(progress_update.reward_for_chosen_arm, 1)
        self.assertEqual(task_a.transfer_bandit_alpha["task_c"], 2.0)
        self.assertEqual(task_a.transfer_bandit_beta["task_c"], 1.0)
        self.assertEqual(task_a.transfer_bandit_pulls["task_c"], 1)
        self.assertEqual(task_a.transfer_bandit_alpha["NONE"], 7.0)
        self.assertEqual(task_a.transfer_bandit_beta["task_b"], 3.0)
        self.assertEqual(task_a.last_transfer_iteration, 4)

    async def test_online_bandit_rich_reward_uses_recent_prior_child_fitness_median(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_rich_reward"),
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
                        "reward_margin": 0.0,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.recent_child_fitness_history = [0.9, 0.1, 0.2]
        child = Program(
            id="task-a-rich-child",
            code="def solve():\n    return 2\n",
            metrics={"combined_score": 0.18},
            parent_id="parent",
            iteration_found=4,
        )

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.5,
            current_best=0.5,
            local_iteration=4,
            foreign_transfer_used=True,
            chosen_transfer_arm="task_b",
            child_program=child,
        )

        self.assertEqual(progress_update.reward_mode, "rich")
        self.assertEqual(progress_update.reward_for_chosen_arm, 1)
        self.assertAlmostEqual(progress_update.child_fitness_for_reward, 0.18)
        self.assertAlmostEqual(progress_update.reward_baseline_fitness, 0.15)
        self.assertEqual(task_a.transfer_bandit_alpha["task_b"], 2.0)
        self.assertEqual(task_a.transfer_bandit_beta["task_b"], 1.0)
        self.assertEqual(task_a.transfer_bandit_pulls["task_b"], 1)
        self.assertEqual(task_a.last_transfer_iteration, 4)
        self.assertEqual(task_a.recent_child_fitness_history, [0.9, 0.1, 0.2, 0.18])

    async def test_online_bandit_rich_reward_falls_back_to_sparse_without_history(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_rich_fallback"),
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
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        child = Program(
            id="task-a-rich-fallback-child",
            code="def solve():\n    return 2\n",
            metrics={"combined_score": 0.25},
            parent_id="parent",
            iteration_found=2,
        )

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.2,
            current_best=0.35,
            local_iteration=2,
            foreign_transfer_used=True,
            chosen_transfer_arm="task_b",
            child_program=child,
        )

        self.assertEqual(progress_update.reward_mode, "rich")
        self.assertEqual(progress_update.reward_for_chosen_arm, 1)
        self.assertAlmostEqual(progress_update.child_fitness_for_reward, 0.25)
        self.assertIsNone(progress_update.reward_baseline_fitness)
        self.assertEqual(task_a.transfer_bandit_alpha["task_b"], 2.0)
        self.assertEqual(task_a.transfer_bandit_beta["task_b"], 1.0)
        self.assertEqual(task_a.transfer_bandit_pulls["task_b"], 1)
        self.assertEqual(task_a.recent_child_fitness_history, [0.25])

    async def test_online_bandit_no_improvement_gives_zero_reward_for_none_arm(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_zero_reward"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.35,
            current_best=0.35,
            local_iteration=4,
            foreign_transfer_used=False,
            chosen_transfer_arm="NONE",
        )

        self.assertEqual(progress_update.reward_for_chosen_arm, 0)
        self.assertEqual(task_a.transfer_bandit_alpha["NONE"], 1.0)
        self.assertEqual(task_a.transfer_bandit_beta["NONE"], 2.0)
        self.assertEqual(task_a.transfer_bandit_pulls["NONE"], 1)
        self.assertIsNone(task_a.last_transfer_iteration)

    async def test_online_bandit_committed_failure_gives_zero_reward_for_chosen_arm(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_failure"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        task_b = controller.task_by_name["task_b"]
        await controller._ensure_initial_program(task_a)
        await controller._ensure_initial_program(task_b)
        controller._initialize_task_progress_state(from_checkpoint=False)

        task_a.local_iteration = 1
        controller._log_multitask_step(
            global_iteration=1,
            task_state=task_a,
            result=TaskIterationResult(
                task_name="task_a",
                local_iteration=1,
                success=False,
                failure_reason="synthetic failure",
                foreign_inspiration_sources=["task_b"],
                foreign_transfer_trigger_reason="online_bandit",
                chosen_transfer_arm="task_b",
            ),
        )

        self.assertEqual(task_a.transfer_bandit_alpha["task_b"], 1.0)
        self.assertEqual(task_a.transfer_bandit_beta["task_b"], 2.0)
        self.assertEqual(task_a.transfer_bandit_pulls["task_b"], 1)
        self.assertEqual(task_a.last_transfer_iteration, 1)
        self.assertEqual(task_a.recent_child_fitness_history, [])

    async def test_online_bandit_rich_failure_gives_zero_reward_and_keeps_history_unchanged(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_rich_failure"),
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
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        controller._ensure_transfer_bandit_state(task_a)
        task_a.recent_child_fitness_history = [0.3]

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.35,
            current_best=0.35,
            local_iteration=3,
            foreign_transfer_used=True,
            chosen_transfer_arm="task_b",
        )

        self.assertEqual(progress_update.reward_mode, "rich")
        self.assertEqual(progress_update.reward_for_chosen_arm, 0)
        self.assertIsNone(progress_update.child_fitness_for_reward)
        self.assertAlmostEqual(progress_update.reward_baseline_fitness, 0.3)
        self.assertEqual(task_a.transfer_bandit_alpha["task_b"], 1.0)
        self.assertEqual(task_a.transfer_bandit_beta["task_b"], 2.0)
        self.assertEqual(task_a.transfer_bandit_pulls["task_b"], 1)
        self.assertEqual(task_a.recent_child_fitness_history, [0.3])

    async def test_improvement_resets_no_improve_steps_to_zero(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_progress_state"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "stagnation",
                        "min_best_fitness_improvement": 1.0e-4,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        task_a.no_improve_steps = 3

        progress_update = controller._update_task_progress_state(
            task_state=task_a,
            previous_best=0.2,
            current_best=0.35,
            local_iteration=4,
            foreign_transfer_used=False,
            chosen_transfer_arm=None,
        )

        self.assertGreater(progress_update.delta_best, 0.0)
        self.assertEqual(task_a.no_improve_steps, 0)
        self.assertEqual(task_a.last_improvement_iteration, 4)
        self.assertIsNone(task_a.last_transfer_iteration)

    async def test_committed_failure_increments_no_improve_steps(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_failure_progress"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "stagnation",
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        task_a = controller.task_by_name["task_a"]
        await controller._ensure_initial_program(task_a)
        controller._initialize_task_progress_state(from_checkpoint=False)

        task_a.local_iteration = 1
        controller._log_multitask_step(
            global_iteration=1,
            task_state=task_a,
            result=TaskIterationResult(
                task_name="task_a",
                local_iteration=1,
                success=False,
                failure_reason="synthetic failure",
            ),
        )

        self.assertEqual(task_a.no_improve_steps, 1)
        self.assertEqual(task_a.last_improvement_iteration, 0)
        self.assertIsNone(task_a.last_transfer_iteration)

    async def test_sequential_checkpoint_roundtrip_preserves_stagnation_state(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_stagnation_checkpoint"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "stagnation",
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        controller.completed_global_iterations = 8
        controller.next_task_index = 1
        task_a = controller.task_by_name["task_a"]
        task_b = controller.task_by_name["task_b"]
        task_a.local_iteration = 4
        task_a.no_improve_steps = 3
        task_a.last_improvement_iteration = 1
        task_a.last_transfer_iteration = 2
        task_b.local_iteration = 4
        task_b.no_improve_steps = 0
        task_b.last_improvement_iteration = 4
        task_b.last_transfer_iteration = None

        checkpoint_path = controller._save_checkpoint(8)

        resumed_controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        resumed_controller._load_checkpoint(checkpoint_path)

        resumed_task_a = resumed_controller.task_by_name["task_a"]
        resumed_task_b = resumed_controller.task_by_name["task_b"]
        self.assertEqual(resumed_controller.completed_global_iterations, 8)
        self.assertEqual(resumed_controller.next_task_index, 1)
        self.assertEqual(resumed_task_a.local_iteration, 4)
        self.assertEqual(resumed_task_a.no_improve_steps, 3)
        self.assertEqual(resumed_task_a.last_improvement_iteration, 1)
        self.assertEqual(resumed_task_a.last_transfer_iteration, 2)
        self.assertEqual(resumed_task_b.local_iteration, 4)
        self.assertEqual(resumed_task_b.no_improve_steps, 0)
        self.assertEqual(resumed_task_b.last_improvement_iteration, 4)
        self.assertIsNone(resumed_task_b.last_transfer_iteration)

    async def test_sequential_checkpoint_roundtrip_preserves_online_bandit_state(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_online_bandit_checkpoint"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 1,
                        "transfer_cooldown": 0,
                        "max_related_tasks": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            await controller._ensure_initial_program(task_state)

        controller.completed_global_iterations = 4
        controller.next_task_index = 1
        task_a = controller.task_by_name["task_a"]
        task_b = controller.task_by_name["task_b"]
        task_a.local_iteration = 2
        task_a.no_improve_steps = 2
        task_a.last_improvement_iteration = 1
        task_a.last_transfer_iteration = 2
        task_a.transfer_bandit_alpha = {"NONE": 3.0, "task_b": 4.0}
        task_a.transfer_bandit_beta = {"NONE": 5.0, "task_b": 6.0}
        task_a.transfer_bandit_pulls = {"NONE": 7, "task_b": 8}
        task_a.recent_child_fitness_history = [0.25, 0.5, 0.75]
        task_b.local_iteration = 2

        checkpoint_path = controller._save_checkpoint(4)

        resumed_controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        resumed_controller._load_checkpoint(checkpoint_path)

        resumed_task_a = resumed_controller.task_by_name["task_a"]
        self.assertEqual(resumed_task_a.transfer_bandit_alpha, {"NONE": 3.0, "task_b": 4.0})
        self.assertEqual(resumed_task_a.transfer_bandit_beta, {"NONE": 5.0, "task_b": 6.0})
        self.assertEqual(resumed_task_a.transfer_bandit_pulls, {"NONE": 7, "task_b": 8})
        self.assertEqual(resumed_task_a.recent_child_fitness_history, [0.25, 0.5, 0.75])

    async def test_checkpoint_is_self_contained_and_relocatable(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_portable"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 2,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            target_value = task_state.env["EXPECTED_RETURN"]
            task_state.llm_ensemble.generate_with_context = AsyncMock(
                return_value=(
                    "<<<<<<< SEARCH\n"
                    "    return 1\n"
                    "=======\n"
                    f"    return {target_value}\n"
                    ">>>>>>> REPLACE"
                )
            )

        await controller.run()

        root_checkpoint = Path(controller.output_dir) / "checkpoints" / "checkpoint_2"
        state_path = root_checkpoint / "multitask_state.json"
        snapshot_path = root_checkpoint / "multitask_config_snapshot.json"
        self.assertTrue(state_path.exists())
        self.assertTrue(snapshot_path.exists())

        state = json.loads(state_path.read_text())
        snapshot = json.loads(snapshot_path.read_text())
        self.assertEqual(state["task_checkpoints"]["task_a"], "tasks/task_a")
        self.assertEqual(state["task_checkpoints"]["task_b"], "tasks/task_b")
        self.assertIn("resume_validation", snapshot)
        self.assertIn("hash", snapshot["resume_validation"])
        self.assertIn("payload", snapshot["resume_validation"])

        relocated_checkpoint = self.tmp_path / "relocated_checkpoint_2"
        shutil.copytree(root_checkpoint, relocated_checkpoint)
        shutil.rmtree(Path(controller.output_dir))

        resumed_controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        resumed_controller._load_checkpoint(str(relocated_checkpoint))

        self.assertEqual(resumed_controller.completed_global_iterations, 2)
        self.assertEqual(resumed_controller.task_by_name["task_a"].local_iteration, 1)
        self.assertEqual(resumed_controller.task_by_name["task_b"].local_iteration, 1)
        self.assertTrue(
            resumed_controller.task_by_name["task_a"].checkpoint_metadata["path"].endswith(
                "tasks/task_a"
            )
        )

    async def test_checkpoint_preserves_large_artifacts_after_relocation(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_artifacts"),
                    "max_global_iterations": 1,
                    "checkpoint_interval": 1,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate_with_artifacts.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        }
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        controller.task_by_name["task_a"].llm_ensemble.generate_with_context = AsyncMock(
            return_value=(
                "<<<<<<< SEARCH\n"
                "    return 1\n"
                "=======\n"
                "    return 2\n"
                ">>>>>>> REPLACE"
            )
        )

        with patch.dict("os.environ", {"ENABLE_ARTIFACTS": "true"}, clear=False):
            await controller.run()

        root_checkpoint = Path(controller.output_dir) / "checkpoints" / "checkpoint_1"
        best_program = controller.task_by_name["task_a"].database.get_best_program()
        self.assertIsNotNone(best_program)
        checkpoint_artifact_dir = (
            root_checkpoint / "tasks" / "task_a" / "artifacts" / best_program.id
        )
        self.assertTrue(checkpoint_artifact_dir.exists())

        relocated_checkpoint = self.tmp_path / "relocated_checkpoint_with_artifacts"
        shutil.copytree(root_checkpoint, relocated_checkpoint)
        shutil.rmtree(Path(controller.output_dir))

        resumed_controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        resumed_controller._load_checkpoint(str(relocated_checkpoint))

        resumed_best_program = resumed_controller.task_by_name["task_a"].database.get_best_program()
        self.assertIsNotNone(resumed_best_program)
        self.assertTrue(Path(resumed_best_program.artifact_dir).exists())
        self.assertTrue(
            str(Path(resumed_best_program.artifact_dir).resolve()).startswith(
                str((relocated_checkpoint / "tasks" / "task_a").resolve())
            )
        )

        resumed_artifacts = resumed_controller.task_by_name["task_a"].database.get_artifacts(
            resumed_best_program.id
        )
        self.assertIn("large_log", resumed_artifacts)
        self.assertTrue(resumed_artifacts["large_log"].startswith("artifact_for_2_"))

    async def test_resume_rejects_mismatched_checkpoint_config_by_default(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_mismatch"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 2,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            target_value = task_state.env["EXPECTED_RETURN"]
            task_state.llm_ensemble.generate_with_context = AsyncMock(
                return_value=(
                    "<<<<<<< SEARCH\n"
                    "    return 1\n"
                    "=======\n"
                    f"    return {target_value}\n"
                    ">>>>>>> REPLACE"
                )
            )

        await controller.run()

        root_checkpoint = Path(controller.output_dir) / "checkpoints" / "checkpoint_2"
        (self.tmp_path / "evaluate_alt.py").write_text(
            (self.tmp_path / "evaluate.py").read_text()
        )

        changed_config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_mismatch"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 2,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate_alt.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        resumed_controller = MultiTaskOpenEvolve(load_multitask_config(changed_config_path))
        with self.assertRaisesRegex(
            ValueError,
            "task 'task_b' evaluation_file changed",
        ):
            resumed_controller._load_checkpoint(str(root_checkpoint))

    async def test_force_resume_allows_mismatched_checkpoint_config(self):
        config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_force_resume"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 2,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        controller = MultiTaskOpenEvolve(load_multitask_config(config_path))
        for task_state in controller.tasks:
            target_value = task_state.env["EXPECTED_RETURN"]
            task_state.llm_ensemble.generate_with_context = AsyncMock(
                return_value=(
                    "<<<<<<< SEARCH\n"
                    "    return 1\n"
                    "=======\n"
                    f"    return {target_value}\n"
                    ">>>>>>> REPLACE"
                )
            )

        await controller.run()

        root_checkpoint = Path(controller.output_dir) / "checkpoints" / "checkpoint_2"
        (self.tmp_path / "evaluate_alt.py").write_text(
            (self.tmp_path / "evaluate.py").read_text()
        )

        changed_config_path = self._write_config(
            {
                "multitask": {
                    "output_dir": str(self.tmp_path / "outputs_force_resume"),
                    "max_global_iterations": 2,
                    "checkpoint_interval": 2,
                    "foreign_inspirations": {"enabled": False},
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate.py",
                            "env": {"EXPECTED_RETURN": "2"},
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": "evaluate_alt.py",
                            "env": {"EXPECTED_RETURN": "3"},
                        },
                    ],
                }
            }
        )

        resumed_controller = MultiTaskOpenEvolve(load_multitask_config(changed_config_path))
        resumed_controller._load_checkpoint(str(root_checkpoint), force_resume=True)

        self.assertEqual(resumed_controller.completed_global_iterations, 2)
        self.assertEqual(resumed_controller.task_by_name["task_a"].local_iteration, 1)
        self.assertEqual(resumed_controller.task_by_name["task_b"].local_iteration, 1)


class _DummyPromptSampler:
    def __init__(self, _prompt_config):
        pass

    def set_templates(self, _template_name):
        pass


class TestMultiTaskManualModeInitialization(unittest.TestCase):
    def test_manual_mode_queue_is_initialized_before_llm_clients(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "init.py").write_text("def solve():\n    return 1\n")
            (tmp_path / "eval.py").write_text(
                "def evaluate(path):\n    return {'combined_score': 1.0}\n"
            )

            controller = MultiTaskOpenEvolve.__new__(MultiTaskOpenEvolve)
            controller.output_dir = str(tmp_path / "run")
            controller.base_config = Config.from_dict(
                {
                    "llm": {
                        "manual_mode": True,
                        "primary_model": "manual-model",
                    }
                }
            )
            controller.multitask_config = SimpleNamespace(config_dir=str(tmp_path))
            controller._configure_task_output_paths = Mock()
            controller._ensure_task_log_handler = Mock()
            controller._configure_task_random_seed = Mock()
            controller._activate_task_logging = lambda _task_name: nullcontext()
            controller._create_evolution_tracer = Mock(return_value=None)

            task_config = TaskConfig(
                name="task_a",
                initial_program=str(tmp_path / "init.py"),
                evaluation_file=str(tmp_path / "eval.py"),
                output_subdir="task_a",
            )
            expected_queue_dir = str(
                (tmp_path / "run" / "task_a" / "manual_tasks_queue").resolve()
            )

            def build_ensemble(models_cfg):
                queue_dirs = {
                    getattr(model_cfg, "_manual_queue_dir", None) for model_cfg in models_cfg
                }
                self.assertEqual(queue_dirs, {expected_queue_dir})
                return SimpleNamespace(models_cfg=models_cfg)

            with (
                patch(
                    "openevolve.multitask.controller.PromptSampler",
                    _DummyPromptSampler,
                ),
                patch(
                    "openevolve.multitask.controller.LLMEnsemble",
                    side_effect=build_ensemble,
                ) as ensemble_ctor,
                patch(
                    "openevolve.multitask.controller.ProgramDatabase",
                    return_value=SimpleNamespace(),
                ) as database_ctor,
                patch(
                    "openevolve.multitask.controller.Evaluator",
                    return_value=SimpleNamespace(),
                ) as evaluator_ctor,
            ):
                task_state = controller._create_task_state(task_config)

            self.assertEqual(ensemble_ctor.call_count, 2)
            database_ctor.assert_called_once()
            evaluator_ctor.assert_called_once()
            self.assertEqual(
                task_state.config.llm.models[0]._manual_queue_dir,
                expected_queue_dir,
            )
            self.assertTrue(Path(expected_queue_dir).is_dir())


if __name__ == "__main__":
    unittest.main()
