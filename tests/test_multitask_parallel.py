"""Tests for synchronized-wave parallel multitask execution."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple
from unittest import mock

import yaml

from openevolve.database import Program
from openevolve.multitask.config import load_multitask_config
from openevolve.multitask.controller import (
    MultiTaskOpenEvolve,
    ParallelWaveMultiTaskOpenEvolve,
    TaskIterationResult,
    create_multitask_controller,
)
from openevolve.multitask import parallel_worker
from openevolve.multitask.parallel_worker import (
    InitialProgramEvaluationResult,
    InitialProgramEvaluationRequest,
    TaskIterationWorkerResult,
    TaskIterationRequest,
    WorkerRngState,
)


class _EnvEchoLLM:
    def __init__(self, model_cfg):
        self.model = model_cfg.name or "env-echo-llm"

    async def generate_with_context(self, _system_message, _messages, **_kwargs):
        expected = os.environ["EXPECTED_RETURN"]
        return (
            "<<<<<<< SEARCH\n"
            "    return 1\n"
            "=======\n"
            f"    return {expected}\n"
            ">>>>>>> REPLACE"
        )

    async def generate(self, prompt, **kwargs):
        return await self.generate_with_context("", [{"role": "user", "content": prompt}], **kwargs)


def build_env_echo_llm(model_cfg):
    return _EnvEchoLLM(model_cfg)


class _FakeWaveHarness:
    def __init__(
        self,
        *,
        task_values: Dict[str, str],
        normal_failures: Optional[Iterable[Tuple[str, int]]] = None,
        infrastructure_failures: Optional[Iterable[Tuple[str, int]]] = None,
        include_large_artifacts: bool = False,
    ):
        self.task_values = dict(task_values)
        self.normal_failures = set(normal_failures or [])
        self.infrastructure_failures = set(infrastructure_failures or [])
        self.include_large_artifacts = include_large_artifacts
        self.initial_requests = {}
        self.iteration_requests_by_task = {task_name: [] for task_name in self.task_values}

    def build_initial_result(self, task_name, request):
        self.initial_requests[task_name] = request
        return InitialProgramEvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={},
            rng_state=request.rng_state,
        )

    def build_iteration_result(self, task_name, request):
        self.iteration_requests_by_task[task_name].append(request)
        wave = request.local_iteration

        if (task_name, wave) in self.infrastructure_failures:
            raise RuntimeError(f"infrastructure boom for {task_name} wave {wave}")

        foreign_sources = [
            source["source_task"]
            for source in request.foreign_inspirations
            if source.get("source_task")
        ]

        if (task_name, wave) in self.normal_failures:
            return TaskIterationWorkerResult(
                task_name=task_name,
                local_iteration=wave,
                success=False,
                target_island=request.target_island,
                rng_state=request.rng_state,
                foreign_inspiration_sources=foreign_sources,
                failure_reason=f"normal failure for {task_name} wave {wave}",
                generation_time_sec=0.01,
                evaluation_time_sec=0.0,
                iteration_time_sec=0.01,
                prompt={"system": "fake", "user": "fake"},
                llm_response=None,
                artifacts=None,
                chosen_transfer_arm=request.chosen_transfer_arm,
            )

        child_program_id = f"{task_name}-wave-{wave}"
        child_program_dict = {
            "id": child_program_id,
            "code": (
                f"def solve():\n"
                f"    return {self.task_values[task_name]}\n"
                f"# {task_name}-wave-{wave}\n"
            ),
            "changes_description": "",
            "language": "python",
            "parent_id": request.parent_program["id"],
            "generation": request.parent_program["generation"] + 1,
            "metrics": {
                "combined_score": float(wave),
                "wave": float(wave),
            },
            "iteration_found": wave,
            "metadata": {
                "changes": f"fake wave {wave}",
                "parent_metrics": request.parent_program.get("metrics", {}),
            },
        }

        artifacts = None
        if self.include_large_artifacts:
            artifacts = {
                "large_log": f"artifact_for_{task_name}_{wave}_" + ("x" * (40 * 1024)),
            }

        return TaskIterationWorkerResult(
            task_name=task_name,
            local_iteration=wave,
            success=True,
            target_island=request.target_island,
            rng_state=request.rng_state,
            foreign_inspiration_sources=foreign_sources,
            child_program_dict=child_program_dict,
            parent_id=request.parent_program["id"],
            generation_time_sec=0.01,
            evaluation_time_sec=0.01,
            iteration_time_sec=0.02,
            prompt={"system": "fake-system", "user": f"wave {wave}"},
            llm_response="fake-response",
            artifacts=artifacts,
            chosen_transfer_arm=request.chosen_transfer_arm,
        )


class _FakeDedicatedTaskWorker:
    def __init__(self, task_name: str, harness: _FakeWaveHarness):
        self.task_name = task_name
        self.harness = harness

    def start(self):
        return None

    def stop(self):
        return None

    def submit_initial_program(self, request):
        future = Future()
        try:
            future.set_result(self.harness.build_initial_result(self.task_name, request))
        except Exception as exc:  # pragma: no cover - safety belt for the fake
            future.set_exception(exc)
        return future

    def submit_iteration(self, request):
        future = Future()
        try:
            future.set_result(self.harness.build_iteration_result(self.task_name, request))
        except Exception as exc:
            future.set_exception(exc)
        return future


class _KeywordOnlyInitError(Exception):
    def __init__(self, message, *, response, body):
        super().__init__(message)
        self.response = response
        self.body = body


class TestParallelWaveMultiTaskOpenEvolve(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tempdir.name)
        (self.tmp_path / "init.py").write_text("def solve():\n    return 1\n")
        (self.tmp_path / "stub_eval.py").write_text(
            "def evaluate(path):\n    return {'combined_score': 0.0}\n"
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_config(self, config_dict):
        config_path = self.tmp_path / "multitask_parallel.yaml"
        config_path.write_text(yaml.safe_dump(config_dict))
        return config_path

    def _base_config(self):
        return {
            "random_seed": 123,
            "diff_based_evolution": True,
            "prompt": {
                "num_top_programs": 1,
                "num_diverse_programs": 0,
            },
            "llm": {
                "models": [
                    {
                        "name": "fake-primary",
                        "init_client": build_env_echo_llm,
                    }
                ],
                "evaluator_models": [
                    {
                        "name": "fake-evaluator",
                        "init_client": build_env_echo_llm,
                    }
                ],
            },
            "database": {
                "population_size": 8,
                "archive_size": 8,
                "num_islands": 1,
                "feature_dimensions": ["combined_score"],
            },
            "evaluator": {
                "parallel_evaluations": 1,
                "timeout": 5,
                "max_retries": 0,
                "use_llm_feedback": False,
            },
        }

    def _load_parallel_config(
        self,
        *,
        output_name: str,
        max_waves: int = 2,
        checkpoint_every_waves: int = 1,
        foreign_inspirations: Optional[Dict[str, object]] = None,
        evaluation_file: Optional[str] = None,
    ):
        config_path = self._write_config(
            {
                "multitask": {
                    "execution_mode": "parallel_synchronized_waves",
                    "output_dir": str(self.tmp_path / output_name),
                    "max_waves": max_waves,
                    "checkpoint_every_waves": checkpoint_every_waves,
                    "foreign_inspirations": foreign_inspirations
                    or {
                        "enabled": True,
                        "every_n_task_iterations": 1,
                        "warmup_task_iterations": 0,
                        "max_related_tasks": 1,
                        "top_programs_per_related_task": 1,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "init.py",
                            "evaluation_file": evaluation_file or "stub_eval.py",
                            "env": {
                                "EXPECTED_RETURN": "2",
                                "EVAL_TASK_NAME": "task_a",
                            },
                            "related_tasks": [{"source_task": "task_b"}],
                        },
                        {
                            "name": "task_b",
                            "initial_program": "init.py",
                            "evaluation_file": evaluation_file or "stub_eval.py",
                            "env": {
                                "EXPECTED_RETURN": "3",
                                "EVAL_TASK_NAME": "task_b",
                            },
                            "related_tasks": [{"source_task": "task_a"}],
                        },
                    ],
                }
            }
        )
        multitask_config = load_multitask_config(config_path)
        multitask_config.base_config = self._base_config()
        return multitask_config

    def _build_fake_controller(self, harness: _FakeWaveHarness, **config_kwargs):
        controller = create_multitask_controller(self._load_parallel_config(**config_kwargs))
        self.assertIsInstance(controller, ParallelWaveMultiTaskOpenEvolve)
        controller._create_task_worker = (
            lambda task_state, _h=harness: _FakeDedicatedTaskWorker(task_state.task_name, _h)
        )
        return controller

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

    async def test_factory_selects_parallel_controller(self):
        controller = create_multitask_controller(
            self._load_parallel_config(output_name="outputs_factory", max_waves=1)
        )
        self.assertIsInstance(controller, ParallelWaveMultiTaskOpenEvolve)
        self.assertIs(type(controller), ParallelWaveMultiTaskOpenEvolve)

    async def test_parallel_mode_keeps_tasks_in_lockstep_after_n_waves(self):
        harness = _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"})
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_lockstep",
            max_waves=2,
        )

        best_programs = await controller.run(max_waves=2)

        self.assertEqual(controller.completed_waves, 2)
        self.assertEqual(controller.completed_global_iterations, 4)
        self.assertEqual(controller.task_by_name["task_a"].local_iteration, 2)
        self.assertEqual(controller.task_by_name["task_b"].local_iteration, 2)
        self.assertEqual(controller._scheduler_counts["task_a"], 2)
        self.assertEqual(controller._scheduler_counts["task_b"], 2)
        self.assertIn("task_a-wave-2", best_programs["task_a"].code)
        self.assertIn("task_b-wave-2", best_programs["task_b"].code)

    async def test_foreign_inspirations_use_pre_wave_state_and_hide_same_wave_children(self):
        harness = _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"})
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_foreign",
            max_waves=2,
        )

        await controller.run(max_waves=2)

        task_a_wave_1 = harness.iteration_requests_by_task["task_a"][0]
        task_a_wave_2 = harness.iteration_requests_by_task["task_a"][1]

        self.assertNotIn("task_b-wave-1", json.dumps(task_a_wave_1.foreign_inspirations))
        self.assertIn("task_b-wave-1", json.dumps(task_a_wave_2.foreign_inspirations))
        self.assertNotIn("task_b-wave-2", json.dumps(task_a_wave_2.foreign_inspirations))

    async def test_stagnation_trigger_uses_pre_wave_state(self):
        harness = _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"})
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_stagnation_pre_wave",
            max_waves=1,
            foreign_inspirations={
                "enabled": True,
                "trigger_mode": "stagnation",
                "warmup_task_iterations": 0,
                "stagnation_patience": 1,
                "transfer_cooldown": 0,
                "max_related_tasks": 1,
                "top_programs_per_related_task": 1,
            },
        )

        controller._start_task_workers()
        try:
            for task_state in controller.tasks:
                await controller._ensure_initial_program(task_state)
            controller._initialize_task_progress_state(from_checkpoint=False)

            for task_state in controller.tasks:
                task_state.local_iteration = 1
                task_state.no_improve_steps = 1

            requests = {
                task_state.task_name: controller._prepare_task_iteration_request(task_state)
                for task_state in controller.tasks
            }

            controller.task_by_name["task_a"].no_improve_steps = 0
            controller.task_by_name["task_b"].no_improve_steps = 0

            self.assertEqual(requests["task_a"].foreign_transfer_trigger_reason, "stagnation")
            self.assertEqual(requests["task_b"].foreign_transfer_trigger_reason, "stagnation")
            self.assertEqual(
                [source["source_task"] for source in requests["task_a"].foreign_inspirations],
                ["task_b"],
            )
            self.assertEqual(
                [source["source_task"] for source in requests["task_b"].foreign_inspirations],
                ["task_a"],
            )
        finally:
            controller._stop_task_workers()

    async def test_online_bandit_arm_choice_uses_pre_wave_frozen_state(self):
        harness = _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"})
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_online_bandit_pre_wave",
            max_waves=1,
            foreign_inspirations={
                "enabled": True,
                "trigger_mode": "online_bandit",
                "warmup_task_iterations": 0,
                "stagnation_patience": 1,
                "transfer_cooldown": 0,
                "max_related_tasks": 1,
                "top_programs_per_related_task": 1,
                "min_pulls_per_arm": 2,
            },
        )

        controller._start_task_workers()
        try:
            for task_state in controller.tasks:
                await controller._ensure_initial_program(task_state)
            controller._initialize_task_progress_state(from_checkpoint=False)

            task_a = controller.task_by_name["task_a"]
            controller._ensure_transfer_bandit_state(task_a)
            task_a.local_iteration = 1
            task_a.no_improve_steps = 1
            task_a.transfer_bandit_pulls = {"NONE": 2, "task_b": 0}

            frozen_transfer_states = controller._snapshot_task_transfer_states()

            task_a.no_improve_steps = 0
            task_a.transfer_bandit_pulls = {"NONE": 0, "task_b": 2}

            request = controller._prepare_task_iteration_request(
                task_a,
                frozen_transfer_states=frozen_transfer_states,
            )

            self.assertEqual(request.chosen_transfer_arm, "task_b")
            self.assertEqual(request.foreign_transfer_trigger_reason, "online_bandit")
            self.assertEqual(
                [source["source_task"] for source in request.foreign_inspirations],
                ["task_b"],
            )
        finally:
            controller._stop_task_workers()

    async def test_parallel_rich_reward_uses_prior_history_for_same_wave_baselines(self):
        harness = _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"})
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_online_bandit_rich_wave",
            max_waves=1,
            foreign_inspirations={
                "enabled": True,
                "trigger_mode": "online_bandit",
                "warmup_task_iterations": 0,
                "stagnation_patience": 1,
                "transfer_cooldown": 0,
                "max_related_tasks": 1,
                "top_programs_per_related_task": 1,
                "reward_mode": "rich",
                "reward_window": 2,
            },
        )
        controller.wandb_logger = mock.MagicMock()

        controller._start_task_workers()
        try:
            for task_state in controller.tasks:
                await controller._ensure_initial_program(task_state)
            controller._initialize_task_progress_state(from_checkpoint=False)

            controller.completed_waves = 1
            controller.completed_global_iterations = 2

            task_a = controller.task_by_name["task_a"]
            task_b = controller.task_by_name["task_b"]
            task_a.local_iteration = 1
            task_b.local_iteration = 1
            task_a.recent_child_fitness_history = [0.9, 0.1, 0.2]
            task_b.recent_child_fitness_history = [0.8, 0.2, 0.3]

            result_a = TaskIterationResult(
                task_name="task_a",
                local_iteration=1,
                success=True,
                child_program=Program(
                    id="task-a-wave-rich",
                    code="def solve():\n    return 2\n",
                    metrics={"combined_score": 0.18},
                    parent_id="parent",
                    iteration_found=1,
                ),
                foreign_inspiration_sources=["task_b"],
                foreign_transfer_trigger_reason="online_bandit",
                chosen_transfer_arm="task_b",
            )
            result_b = TaskIterationResult(
                task_name="task_b",
                local_iteration=1,
                success=True,
                child_program=Program(
                    id="task-b-wave-rich",
                    code="def solve():\n    return 3\n",
                    metrics={"combined_score": 0.28},
                    parent_id="parent",
                    iteration_found=1,
                ),
                foreign_inspiration_sources=["task_a"],
                foreign_transfer_trigger_reason="online_bandit",
                chosen_transfer_arm="task_a",
            )

            controller._log_parallel_wave(
                wave_index=1,
                committed_results=[result_a, result_b],
            )
        finally:
            controller._stop_task_workers()

        logged_metrics = controller.wandb_logger.log_metrics.call_args.args[0]
        self.assertEqual(logged_metrics["task/task_a/bandit_reward_mode"], "rich")
        self.assertAlmostEqual(logged_metrics["task/task_a/bandit_reward_baseline"], 0.15)
        self.assertAlmostEqual(logged_metrics["task/task_a/bandit_reward_child_fitness"], 0.18)
        self.assertEqual(logged_metrics["task/task_a/foreign_transfer_reward"], 1)
        self.assertEqual(logged_metrics["task/task_b/bandit_reward_mode"], "rich")
        self.assertAlmostEqual(logged_metrics["task/task_b/bandit_reward_baseline"], 0.25)
        self.assertAlmostEqual(logged_metrics["task/task_b/bandit_reward_child_fitness"], 0.28)
        self.assertEqual(logged_metrics["task/task_b/foreign_transfer_reward"], 1)
        self.assertEqual(task_a.recent_child_fitness_history, [0.9, 0.1, 0.2, 0.18])
        self.assertEqual(task_b.recent_child_fitness_history, [0.8, 0.2, 0.3, 0.28])

    async def test_prepare_task_iteration_request_applies_transfer_prompt_overrides(self):
        multitask_config = self._load_parallel_config(
            output_name="outputs_parallel_prompt_override",
            max_waves=1,
            foreign_inspirations={
                "enabled": True,
                "every_n_task_iterations": 1,
                "warmup_task_iterations": 0,
                "max_related_tasks": 1,
                "top_programs_per_related_task": 1,
                "prompt_overrides": {
                    "num_top_programs": 1,
                    "num_diverse_programs": 0,
                    "num_local_inspirations": 1,
                },
            },
        )
        multitask_config.base_config["prompt"]["num_top_programs"] = 3
        multitask_config.base_config["prompt"]["num_diverse_programs"] = 2
        multitask_config.base_config["prompt"]["num_local_inspirations"] = 2

        controller = create_multitask_controller(multitask_config)
        self.assertIsInstance(controller, ParallelWaveMultiTaskOpenEvolve)
        controller._create_task_worker = (
            lambda task_state: _FakeDedicatedTaskWorker(
                task_state.task_name,
                _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"}),
            )
        )

        controller._start_task_workers()
        try:
            for task_state in controller.tasks:
                await controller._ensure_initial_program(task_state)

            task_a = controller.task_by_name["task_a"]
            self._seed_task_programs(task_a)

            request = controller._prepare_task_iteration_request(task_a)

            self.assertEqual(
                [source["source_task"] for source in request.foreign_inspirations],
                ["task_b"],
            )
            self.assertEqual(len(request.previous_programs), 1)
            self.assertEqual(len(request.top_programs), 1)
            self.assertEqual(len(request.inspirations), 1)
            self.assertEqual(request.effective_prompt_config["num_top_programs"], 1)
            self.assertEqual(request.effective_prompt_config["num_diverse_programs"], 0)
            self.assertEqual(request.effective_prompt_config["num_local_inspirations"], 1)
            self.assertEqual(task_a.config.prompt.num_top_programs, 3)
            self.assertEqual(task_a.config.prompt.num_diverse_programs, 2)
            self.assertEqual(task_a.config.prompt.num_local_inspirations, 2)
        finally:
            controller._stop_task_workers()

    async def test_normal_task_failure_still_advances_synchronized_wave(self):
        harness = _FakeWaveHarness(
            task_values={"task_a": "2", "task_b": "3"},
            normal_failures={("task_b", 1)},
        )
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_normal_failure",
            max_waves=1,
        )

        best_programs = await controller.run(max_waves=1)

        self.assertEqual(controller.completed_waves, 1)
        self.assertEqual(controller.completed_global_iterations, 2)
        self.assertEqual(controller.task_by_name["task_a"].local_iteration, 1)
        self.assertEqual(controller.task_by_name["task_b"].local_iteration, 1)
        self.assertEqual(controller.task_by_name["task_a"].no_improve_steps, 0)
        self.assertEqual(controller.task_by_name["task_b"].no_improve_steps, 1)
        self.assertEqual(len(controller.task_by_name["task_a"].database.programs), 2)
        self.assertEqual(len(controller.task_by_name["task_b"].database.programs), 1)
        self.assertIsNotNone(best_programs["task_a"])
        self.assertIsNotNone(best_programs["task_b"])

    async def test_infrastructure_failure_aborts_wave_without_partial_commit(self):
        harness = _FakeWaveHarness(
            task_values={"task_a": "2", "task_b": "3"},
            infrastructure_failures={("task_b", 1)},
        )
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_infra_failure",
            max_waves=1,
        )

        with self.assertRaisesRegex(RuntimeError, "aborted before commit"):
            await controller.run(max_waves=1)

        self.assertEqual(controller.completed_waves, 0)
        self.assertEqual(controller.completed_global_iterations, 0)
        self.assertEqual(controller.task_by_name["task_a"].local_iteration, 0)
        self.assertEqual(controller.task_by_name["task_b"].local_iteration, 0)
        self.assertEqual(len(controller.task_by_name["task_a"].database.programs), 1)
        self.assertEqual(len(controller.task_by_name["task_b"].database.programs), 1)
        self.assertFalse((Path(controller.output_dir) / "checkpoints").exists())

    async def test_aborted_wave_does_not_update_online_bandit_stats(self):
        harness = _FakeWaveHarness(
            task_values={"task_a": "2", "task_b": "3"},
            normal_failures={("task_a", 1), ("task_b", 1)},
            infrastructure_failures={("task_b", 2)},
        )
        controller = self._build_fake_controller(
            harness,
            output_name="outputs_online_bandit_abort",
            max_waves=2,
            foreign_inspirations={
                "enabled": True,
                "trigger_mode": "online_bandit",
                "warmup_task_iterations": 0,
                "stagnation_patience": 1,
                "transfer_cooldown": 0,
                "max_related_tasks": 1,
                "top_programs_per_related_task": 1,
            },
        )

        with self.assertRaisesRegex(RuntimeError, "aborted before commit"):
            await controller.run(max_waves=2)

        task_a = controller.task_by_name["task_a"]
        task_b = controller.task_by_name["task_b"]
        self.assertEqual(controller.completed_waves, 1)
        self.assertEqual(task_a.transfer_bandit_pulls["NONE"], 0)
        self.assertEqual(task_a.transfer_bandit_pulls["task_b"], 0)
        self.assertEqual(task_b.transfer_bandit_pulls["NONE"], 0)
        self.assertEqual(task_b.transfer_bandit_pulls["task_a"], 0)

    async def test_parallel_checkpoint_roundtrip_preserves_stagnation_state(self):
        foreign_inspirations = {
            "enabled": True,
            "trigger_mode": "stagnation",
            "warmup_task_iterations": 0,
            "stagnation_patience": 1,
            "transfer_cooldown": 0,
            "max_related_tasks": 1,
            "top_programs_per_related_task": 1,
        }
        controller = self._build_fake_controller(
            _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"}),
            output_name="outputs_stagnation_checkpoint",
            max_waves=2,
            checkpoint_every_waves=1,
            foreign_inspirations=foreign_inspirations,
        )

        controller._start_task_workers()
        try:
            for task_state in controller.tasks:
                await controller._ensure_initial_program(task_state)

            controller.completed_waves = 2
            controller.completed_global_iterations = 4
            task_a = controller.task_by_name["task_a"]
            task_b = controller.task_by_name["task_b"]
            task_a.local_iteration = 2
            task_a.no_improve_steps = 2
            task_a.last_improvement_iteration = 1
            task_a.last_transfer_iteration = 2
            task_b.local_iteration = 2
            task_b.no_improve_steps = 0
            task_b.last_improvement_iteration = 2
            task_b.last_transfer_iteration = None

            checkpoint_path = controller._save_checkpoint(2)
        finally:
            controller._stop_task_workers()

        resumed_controller = self._build_fake_controller(
            _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"}),
            output_name="outputs_stagnation_checkpoint",
            max_waves=3,
            checkpoint_every_waves=1,
            foreign_inspirations=foreign_inspirations,
        )
        resumed_controller._load_checkpoint(checkpoint_path)

        resumed_task_a = resumed_controller.task_by_name["task_a"]
        resumed_task_b = resumed_controller.task_by_name["task_b"]
        self.assertEqual(resumed_controller.completed_waves, 2)
        self.assertEqual(resumed_controller.completed_global_iterations, 4)
        self.assertEqual(resumed_task_a.local_iteration, 2)
        self.assertEqual(resumed_task_a.no_improve_steps, 2)
        self.assertEqual(resumed_task_a.last_improvement_iteration, 1)
        self.assertEqual(resumed_task_a.last_transfer_iteration, 2)
        self.assertEqual(resumed_task_b.local_iteration, 2)
        self.assertEqual(resumed_task_b.no_improve_steps, 0)
        self.assertEqual(resumed_task_b.last_improvement_iteration, 2)
        self.assertIsNone(resumed_task_b.last_transfer_iteration)

    async def test_parallel_checkpoint_roundtrip_preserves_online_bandit_state(self):
        foreign_inspirations = {
            "enabled": True,
            "trigger_mode": "online_bandit",
            "warmup_task_iterations": 0,
            "stagnation_patience": 1,
            "transfer_cooldown": 0,
            "max_related_tasks": 1,
            "top_programs_per_related_task": 1,
        }
        controller = self._build_fake_controller(
            _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"}),
            output_name="outputs_online_bandit_checkpoint",
            max_waves=2,
            checkpoint_every_waves=1,
            foreign_inspirations=foreign_inspirations,
        )

        controller._start_task_workers()
        try:
            for task_state in controller.tasks:
                await controller._ensure_initial_program(task_state)

            controller.completed_waves = 2
            controller.completed_global_iterations = 4
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

            checkpoint_path = controller._save_checkpoint(2)
        finally:
            controller._stop_task_workers()

        resumed_controller = self._build_fake_controller(
            _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"}),
            output_name="outputs_online_bandit_checkpoint",
            max_waves=3,
            checkpoint_every_waves=1,
            foreign_inspirations=foreign_inspirations,
        )
        resumed_controller._load_checkpoint(checkpoint_path)

        resumed_task_a = resumed_controller.task_by_name["task_a"]
        self.assertEqual(resumed_task_a.transfer_bandit_alpha, {"NONE": 3.0, "task_b": 4.0})
        self.assertEqual(resumed_task_a.transfer_bandit_beta, {"NONE": 5.0, "task_b": 6.0})
        self.assertEqual(resumed_task_a.transfer_bandit_pulls, {"NONE": 7, "task_b": 8})
        self.assertEqual(resumed_task_a.recent_child_fitness_history, [0.25, 0.5, 0.75])

    async def test_parallel_checkpoint_resume_and_artifact_portability(self):
        first_harness = _FakeWaveHarness(
            task_values={"task_a": "2", "task_b": "3"},
            include_large_artifacts=True,
        )
        controller = self._build_fake_controller(
            first_harness,
            output_name="outputs_checkpoint_resume",
            max_waves=1,
            checkpoint_every_waves=1,
        )

        await controller.run(max_waves=1)

        checkpoint_root = Path(controller.output_dir) / "checkpoints" / "checkpoint_wave_0001"
        relocated_checkpoint = self.tmp_path / "relocated_parallel_checkpoint"
        shutil.copytree(checkpoint_root, relocated_checkpoint)
        shutil.rmtree(Path(controller.output_dir))

        second_harness = _FakeWaveHarness(task_values={"task_a": "2", "task_b": "3"})
        resumed_controller = self._build_fake_controller(
            second_harness,
            output_name="outputs_checkpoint_resume",
            max_waves=2,
            checkpoint_every_waves=1,
        )
        resumed_controller._load_checkpoint(str(relocated_checkpoint))

        resumed_best_program = resumed_controller.task_by_name["task_a"].database.get_best_program()
        self.assertIsNotNone(resumed_best_program)
        self.assertTrue(Path(resumed_best_program.artifact_dir).exists())
        self.assertTrue(
            str(Path(resumed_best_program.artifact_dir).resolve()).startswith(
                str((relocated_checkpoint / "tasks" / "task_a").resolve())
            )
        )
        self.assertIn(
            "large_log",
            resumed_controller.task_by_name["task_a"].database.get_artifacts(
                resumed_best_program.id
            ),
        )
        self.assertEqual(resumed_controller.completed_waves, 1)
        self.assertEqual(resumed_controller.completed_global_iterations, 2)
        self.assertEqual(resumed_controller.task_by_name["task_a"].local_iteration, 1)
        self.assertEqual(resumed_controller.task_by_name["task_b"].local_iteration, 1)

        await resumed_controller.run(checkpoint_path=str(relocated_checkpoint), max_waves=2)

        self.assertEqual(resumed_controller.completed_waves, 2)
        self.assertEqual(resumed_controller.completed_global_iterations, 4)
        self.assertEqual(resumed_controller.task_by_name["task_a"].local_iteration, 2)
        self.assertEqual(resumed_controller.task_by_name["task_b"].local_iteration, 2)

    async def test_parallel_mode_isolates_shared_evaluator_env_in_dedicated_workers(self):
        helper_file = self.tmp_path / "helper_env.py"
        helper_file.write_text(
            "\n".join(
                [
                    "import os",
                    "",
                    "IMPORTED_EXPECTED_RETURN = os.environ['EXPECTED_RETURN']",
                    "IMPORTED_TASK_NAME = os.environ['EVAL_TASK_NAME']",
                ]
            )
        )
        evaluation_file = self.tmp_path / "evaluate_env_isolated.py"
        evaluation_file.write_text(
            "\n".join(
                [
                    "from helper_env import IMPORTED_EXPECTED_RETURN, IMPORTED_TASK_NAME",
                    "",
                    "def evaluate(path):",
                    "    code = open(path, 'r').read()",
                    "    score = 1.0 if f'return {IMPORTED_EXPECTED_RETURN}' in code else 0.0",
                    "    return {",
                    "        'combined_score': score,",
                    "        'helper_expected_return': IMPORTED_EXPECTED_RETURN,",
                    "        'helper_task_name': IMPORTED_TASK_NAME,",
                    "    }",
                ]
            )
        )

        controller = create_multitask_controller(
            self._load_parallel_config(
                output_name="outputs_env_isolation",
                max_waves=1,
                foreign_inspirations={"enabled": False},
                evaluation_file=str(evaluation_file),
            )
        )
        self.assertIsInstance(controller, ParallelWaveMultiTaskOpenEvolve)

        best_programs = await controller.run(max_waves=1)

        self.assertEqual(best_programs["task_a"].metrics["combined_score"], 1.0)
        self.assertEqual(best_programs["task_b"].metrics["combined_score"], 1.0)
        self.assertEqual(best_programs["task_a"].metrics["helper_expected_return"], "2")
        self.assertEqual(best_programs["task_b"].metrics["helper_expected_return"], "3")
        self.assertEqual(best_programs["task_a"].metrics["helper_task_name"], "task_a")
        self.assertEqual(best_programs["task_b"].metrics["helper_task_name"], "task_b")
        self.assertIn("return 2", best_programs["task_a"].code)
        self.assertIn("return 3", best_programs["task_b"].code)

        for task_name in ("task_a", "task_b"):
            task_log_dir = Path(controller.task_by_name[task_name].config.log_dir)
            worker_logs = list(task_log_dir.glob(f"openevolve_parallel_worker_{task_name}_*.log"))
            self.assertEqual(len(worker_logs), 1)
            worker_log_text = worker_logs[0].read_text()
            self.assertIn(f"[{task_name}]", worker_log_text)
            self.assertIn("Evaluated program", worker_log_text)

    def test_run_task_iteration_wraps_unpicklable_generation_error(self):
        request = TaskIterationRequest(
            task_name="task_a",
            local_iteration=1,
            target_island=0,
            parent_program=Program(
                id="parent",
                code="def solve():\n    return 1\n",
                language="python",
                metrics={"combined_score": 0.0},
                generation=0,
                iteration_found=0,
                metadata={},
            ).to_dict(),
            inspirations=[],
            previous_programs=[],
            top_programs=[],
            parent_artifacts={},
            foreign_inspirations=[],
            feature_dimensions=["combined_score"],
            rng_state=WorkerRngState(None, None, None, None),
        )

        class _PromptSampler:
            def build_prompt(self, **_kwargs):
                return {"system": "fake-system", "user": "fake-user"}

        class _LLM:
            async def generate_with_context(self, *_args, **_kwargs):
                raise _KeywordOnlyInitError("boom", response=object(), body={})

        with (
            mock.patch.object(parallel_worker, "_restore_rng_state", return_value=None),
            mock.patch.object(
                parallel_worker,
                "_worker_config",
                SimpleNamespace(
                    prompt=SimpleNamespace(
                        num_top_programs=3,
                        num_diverse_programs=2,
                        programs_as_changes_description=False,
                        initial_changes_description="",
                        diff_summary_max_line_len=80,
                        diff_summary_max_lines=20,
                    ),
                    diff_based_evolution=False,
                    language="python",
                    max_code_length=1000,
                ),
            ),
            mock.patch.object(parallel_worker, "_worker_prompt_sampler", _PromptSampler()),
            mock.patch.object(parallel_worker, "_worker_evaluator", object()),
            mock.patch.object(parallel_worker, "_worker_llm_ensemble", _LLM()),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "LLM generation failed: _KeywordOnlyInitError: boom",
            ):
                parallel_worker.run_task_iteration(request)

    def test_run_task_iteration_uses_effective_prompt_config_for_rendering(self):
        request = TaskIterationRequest(
            task_name="task_a",
            local_iteration=1,
            target_island=0,
            parent_program=Program(
                id="parent",
                code="def solve():\n    return 1\n",
                language="python",
                metrics={"combined_score": 0.0},
                generation=0,
                iteration_found=0,
                metadata={},
            ).to_dict(),
            inspirations=[],
            previous_programs=[
                {
                    "id": "prev",
                    "code": "def solve():\n    return 2\n",
                    "metrics": {"combined_score": 0.9},
                }
            ],
            top_programs=[
                {
                    "id": "top1",
                    "code": "def solve():\n    return 2\n",
                    "metrics": {"combined_score": 0.9},
                },
                {
                    "id": "top2",
                    "code": "def solve():\n    return 3\n",
                    "metrics": {"combined_score": 0.8},
                },
            ],
            parent_artifacts={},
            foreign_inspirations=[
                {
                    "source_task": "task_b",
                    "programs": [
                        {
                            "id": "foreign1",
                            "code": "def foreign():\n    return 4\n",
                            "metrics": {"combined_score": 0.7},
                        }
                    ],
                }
            ],
            feature_dimensions=["combined_score"],
            rng_state=WorkerRngState(None, None, None, None),
            effective_prompt_config={
                "num_top_programs": 1,
                "num_diverse_programs": 0,
                "num_local_inspirations": 1,
            },
        )

        captured = {}

        def build_prompt_spy(prompt_sampler_self, *args, **kwargs):
            captured["sampler_num_top_programs"] = prompt_sampler_self.config.num_top_programs
            captured["sampler_num_diverse_programs"] = (
                prompt_sampler_self.config.num_diverse_programs
            )
            captured["num_top_programs"] = len(kwargs["top_programs"])
            captured["num_previous_programs"] = len(kwargs["previous_programs"])
            return {"system": "fake-system", "user": "fake-user"}

        class _LLM:
            async def generate_with_context(self, *_args, **_kwargs):
                return None

        with (
            mock.patch.object(parallel_worker, "_restore_rng_state", return_value=None),
            mock.patch.object(
                parallel_worker,
                "_worker_config",
                SimpleNamespace(
                    prompt=SimpleNamespace(
                        num_top_programs=3,
                        num_diverse_programs=2,
                        programs_as_changes_description=False,
                        initial_changes_description="",
                        diff_summary_max_line_len=80,
                        diff_summary_max_lines=20,
                    ),
                    diff_based_evolution=False,
                    language="python",
                    max_code_length=1000,
                ),
            ),
            mock.patch.object(parallel_worker, "_worker_prompt_sampler", object()),
            mock.patch.object(parallel_worker, "_worker_evaluator", object()),
            mock.patch.object(parallel_worker, "_worker_llm_ensemble", _LLM()),
            mock.patch.object(parallel_worker.PromptSampler, "build_prompt", new=build_prompt_spy),
        ):
            result = parallel_worker.run_task_iteration(request)

        self.assertFalse(result.success)
        self.assertEqual(captured["sampler_num_top_programs"], 1)
        self.assertEqual(captured["sampler_num_diverse_programs"], 0)
        self.assertEqual(captured["num_previous_programs"], 1)
        self.assertEqual(captured["num_top_programs"], 2)

    def test_run_initial_program_evaluation_wraps_unpicklable_errors(self):
        request = InitialProgramEvaluationRequest(
            program_id="init-program",
            code="def solve():\n    return 1\n",
            rng_state=WorkerRngState(None, None, None, None),
        )

        class _Evaluator:
            async def evaluate_program(self, *_args, **_kwargs):
                raise _KeywordOnlyInitError("boom", response=object(), body={})

        with (
            mock.patch.object(parallel_worker, "_restore_rng_state", return_value=None),
            mock.patch.object(parallel_worker, "_worker_evaluator", _Evaluator()),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "initial program evaluation failed: _KeywordOnlyInitError: boom",
            ):
                parallel_worker.run_initial_program_evaluation(request)


if __name__ == "__main__":
    unittest.main()
