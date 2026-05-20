"""Tests for multitask configuration loading and per-task config derivation."""

import tempfile
import unittest
from pathlib import Path

import yaml

from openevolve.config import Config
from openevolve.multitask.config import (
    derive_task_config,
    load_base_task_config,
    load_multitask_config,
)


class TestMultitaskConfig(unittest.TestCase):
    def test_load_and_derive_task_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "init.py").write_text("def solve():\n    return 1\n")
            (tmp_path / "eval.py").write_text(
                "def evaluate(path):\n    return {'combined_score': 1.0}\n"
            )
            (tmp_path / "templates").mkdir()

            config_path = tmp_path / "multitask.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "multitask": {
                            "output_dir": "results/run_a",
                            "max_global_iterations": 12,
                            "checkpoint_interval": 3,
                            "base_config": {
                                "random_seed": 7,
                                "prompt": {"template_dir": "templates"},
                            },
                            "tasks": [
                                {
                                    "name": "task_a",
                                    "initial_program": "init.py",
                                    "evaluation_file": "eval.py",
                                    "related_tasks": [{"source_task": "task_b"}],
                                },
                                {
                                    "name": "task_b",
                                    "initial_program": "init.py",
                                    "evaluation_file": "eval.py",
                                    "config_overrides": {
                                        "random_seed": 99,
                                        "prompt": {"system_message": "task-specific"},
                                    },
                                },
                            ],
                        }
                    }
                )
            )

            multitask_config = load_multitask_config(config_path)
            self.assertTrue(Path(multitask_config.output_dir).is_absolute())
            self.assertEqual(multitask_config.tasks[0].output_subdir, "task_a")
            self.assertTrue(Path(multitask_config.tasks[0].initial_program).is_absolute())
            self.assertTrue(Path(multitask_config.tasks[0].evaluation_file).is_absolute())

            base_config = load_base_task_config(multitask_config)
            task_a_config = derive_task_config(
                base_config,
                multitask_config.tasks[0].config_overrides,
                multitask_config.config_dir,
            )
            task_b_config = derive_task_config(
                base_config,
                multitask_config.tasks[1].config_overrides,
                multitask_config.config_dir,
            )

            self.assertEqual(task_a_config.random_seed, 7)
            self.assertEqual(task_b_config.random_seed, 99)
            self.assertEqual(task_b_config.prompt.system_message, "task-specific")
            self.assertTrue(Path(task_a_config.prompt.template_dir).is_absolute())

    def test_task_random_seed_override_updates_database_seed(self):
        base_config = Config.from_dict(
            {
                "random_seed": 42,
                "database": {"random_seed": 42},
            }
        )

        task_config = derive_task_config(
            base_config=base_config,
            overrides={"random_seed": 99},
            config_dir=".",
        )

        self.assertEqual(task_config.random_seed, 99)
        self.assertEqual(task_config.database.random_seed, 99)

    def test_explicit_task_database_random_seed_override_wins(self):
        base_config = Config.from_dict(
            {
                "random_seed": 42,
                "database": {"random_seed": 42},
            }
        )

        task_config = derive_task_config(
            base_config=base_config,
            overrides={
                "random_seed": 99,
                "database": {"random_seed": 7},
            },
            config_dir=".",
        )

        self.assertEqual(task_config.random_seed, 99)
        self.assertEqual(task_config.database.random_seed, 7)


class TestMultitaskConfigValidation(unittest.TestCase):
    def test_default_execution_mode_is_sequential_round_robin(self):
        multitask_config = load_multitask_config_from_dict(
            {
                "multitask": {
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "a.py",
                            "evaluation_file": "eval.py",
                        }
                    ]
                }
            }
        )

        self.assertEqual(multitask_config.execution_mode, "sequential_round_robin")
        self.assertEqual(multitask_config.max_global_iterations, 100)
        self.assertEqual(multitask_config.checkpoint_interval, 10)

    def test_foreign_inspirations_default_trigger_mode_is_periodic(self):
        multitask_config = load_multitask_config_from_dict(
            {
                "multitask": {
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "a.py",
                            "evaluation_file": "eval.py",
                        }
                    ]
                }
            }
        )

        foreign_inspirations = multitask_config.foreign_inspirations
        self.assertEqual(foreign_inspirations.trigger_mode, "periodic")
        self.assertEqual(foreign_inspirations.stagnation_patience, 6)
        self.assertEqual(foreign_inspirations.transfer_cooldown, 4)
        self.assertEqual(foreign_inspirations.min_best_fitness_improvement, 1.0e-4)
        self.assertEqual(foreign_inspirations.min_improvement, 1.0e-4)
        self.assertEqual(foreign_inspirations.min_pulls_per_arm, 2)
        self.assertEqual(foreign_inspirations.bandit_decay, 1.0)
        self.assertEqual(foreign_inspirations.reward_mode, "sparse")
        self.assertEqual(foreign_inspirations.reward_window, 5)
        self.assertEqual(foreign_inspirations.reward_margin, 0.0)
        self.assertIsNone(foreign_inspirations.prompt_overrides)

    def test_foreign_inspiration_prompt_overrides_load(self):
        multitask_config = load_multitask_config_from_dict(
            {
                "multitask": {
                    "foreign_inspirations": {
                        "prompt_overrides": {
                            "num_top_programs": 1,
                            "num_diverse_programs": 0,
                            "num_local_inspirations": 2,
                        }
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "a.py",
                            "evaluation_file": "eval.py",
                        }
                    ],
                }
            }
        )

        prompt_overrides = multitask_config.foreign_inspirations.prompt_overrides
        self.assertIsNotNone(prompt_overrides)
        self.assertEqual(prompt_overrides.num_top_programs, 1)
        self.assertEqual(prompt_overrides.num_diverse_programs, 0)
        self.assertEqual(prompt_overrides.num_local_inspirations, 2)

    def test_foreign_inspiration_prompt_overrides_validate_non_negative_ints(self):
        with self.assertRaisesRegex(ValueError, "prompt_overrides.num_top_programs"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "prompt_overrides": {"num_top_programs": -1}
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "prompt_overrides.num_local_inspirations"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "prompt_overrides": {"num_local_inspirations": 1.5}
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

    def test_legacy_min_improvement_alias_still_loads(self):
        multitask_config = load_multitask_config_from_dict(
            {
                "multitask": {
                    "foreign_inspirations": {
                        "require_stagnation": True,
                        "stagnation_patience": 3,
                        "min_improvement": 0.25,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "a.py",
                            "evaluation_file": "eval.py",
                        }
                    ],
                }
            }
        )

        foreign_inspirations = multitask_config.foreign_inspirations
        self.assertEqual(foreign_inspirations.trigger_mode, "stagnation")
        self.assertEqual(foreign_inspirations.stagnation_patience, 3)
        self.assertEqual(foreign_inspirations.min_best_fitness_improvement, 0.25)
        self.assertEqual(foreign_inspirations.min_improvement, 0.25)

    def test_stagnation_trigger_mode_validates_required_fields(self):
        with self.assertRaisesRegex(ValueError, "trigger_mode"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {"trigger_mode": "unknown"},
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "stagnation_patience"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "stagnation",
                            "stagnation_patience": 0,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "transfer_cooldown"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "stagnation",
                            "transfer_cooldown": -1,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "min_best_fitness_improvement"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "stagnation",
                            "min_best_fitness_improvement": -1.0,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

    def test_online_bandit_trigger_mode_validates_required_fields(self):
        multitask_config = load_multitask_config_from_dict(
            {
                "multitask": {
                    "foreign_inspirations": {
                        "enabled": True,
                        "trigger_mode": "online_bandit",
                        "warmup_task_iterations": 0,
                        "stagnation_patience": 2,
                        "transfer_cooldown": 0,
                        "min_best_fitness_improvement": 1.0e-4,
                        "max_related_tasks": 1,
                        "min_pulls_per_arm": 3,
                        "bandit_decay": 0.75,
                        "reward_mode": "rich",
                        "reward_window": 7,
                        "reward_margin": 0.05,
                    },
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "a.py",
                            "evaluation_file": "eval.py",
                        }
                    ],
                }
            }
        )

        foreign_inspirations = multitask_config.foreign_inspirations
        self.assertEqual(foreign_inspirations.trigger_mode, "online_bandit")
        self.assertEqual(foreign_inspirations.min_pulls_per_arm, 3)
        self.assertEqual(foreign_inspirations.bandit_decay, 0.75)
        self.assertEqual(foreign_inspirations.reward_mode, "rich")
        self.assertEqual(foreign_inspirations.reward_window, 7)
        self.assertEqual(foreign_inspirations.reward_margin, 0.05)

        with self.assertRaisesRegex(ValueError, "max_related_tasks"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "online_bandit",
                            "max_related_tasks": 2,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "min_pulls_per_arm"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "online_bandit",
                            "max_related_tasks": 1,
                            "min_pulls_per_arm": 0,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "bandit_decay"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "online_bandit",
                            "max_related_tasks": 1,
                            "bandit_decay": 0.0,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "reward_mode"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "trigger_mode": "online_bandit",
                            "max_related_tasks": 1,
                            "reward_mode": "dense",
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "reward_window"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "reward_window": 0,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "reward_margin"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "foreign_inspirations": {
                            "reward_margin": -0.1,
                        },
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

    def test_parallel_mode_requires_wave_fields(self):
        with self.assertRaisesRegex(ValueError, "max_waves"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "execution_mode": "parallel_synchronized_waves",
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

        with self.assertRaisesRegex(ValueError, "checkpoint_every_waves"):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "execution_mode": "parallel_synchronized_waves",
                        "max_waves": 3,
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                            }
                        ],
                    }
                }
            )

    def test_parallel_mode_loads_explicit_wave_settings(self):
        multitask_config = load_multitask_config_from_dict(
            {
                "multitask": {
                    "execution_mode": "parallel_synchronized_waves",
                    "max_waves": 5,
                    "checkpoint_every_waves": 1,
                    "tasks": [
                        {
                            "name": "task_a",
                            "initial_program": "a.py",
                            "evaluation_file": "eval.py",
                        }
                    ],
                }
            }
        )

        self.assertEqual(multitask_config.execution_mode, "parallel_synchronized_waves")
        self.assertEqual(multitask_config.max_waves, 5)
        self.assertEqual(multitask_config.checkpoint_every_waves, 1)

    def test_unknown_related_task_reference_raises(self):
        with self.assertRaises(ValueError):
            load_multitask_config_from_dict(
                {
                    "multitask": {
                        "tasks": [
                            {
                                "name": "task_a",
                                "initial_program": "a.py",
                                "evaluation_file": "eval.py",
                                "related_tasks": [{"source_task": "task_missing"}],
                            }
                        ]
                    }
                }
            )


def load_multitask_config_from_dict(config_dict):
    """Small helper to test validation without writing fixtures to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "a.py").write_text("def solve():\n    return 1\n")
        (tmp_path / "eval.py").write_text(
            "def evaluate(path):\n    return {'combined_score': 1.0}\n"
        )
        config_path = tmp_path / "multitask.yaml"
        config_path.write_text(yaml.safe_dump(config_dict))
        return load_multitask_config(config_path)
