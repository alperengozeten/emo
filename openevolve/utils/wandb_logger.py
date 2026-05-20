"""Optional Weights & Biases logging helpers for OpenEvolve."""

from __future__ import annotations

import importlib
import logging
import math
import os
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_SKIP = object()
_MAXIMIZE_METRIC_TOKENS = (
    "accuracy",
    "auc",
    "bleu",
    "correlation",
    "coverage",
    "fitness",
    "f1",
    "gain",
    "iou",
    "kdt",
    "pass",
    "precision",
    "r2",
    "recall",
    "return",
    "reward",
    "rouge",
    "score",
    "similarity",
)
_MINIMIZE_METRIC_TOKENS = (
    "cost",
    "distance",
    "error",
    "latency",
    "loss",
    "mape",
    "mae",
    "mse",
    "nmae",
    "nll",
    "nmse",
    "perplex",
    "rmse",
)
_EXCLUDED_BEST_TRACKING_METRICS = {
    "best_fitness",
    "best_fitness_so_far",
    "best_task_fitness_max",
    "complexity",
    "current_fitness",
    "diversity",
    "evaluation_success",
    "generation",
    "global_iteration",
    "iteration",
    "num_active_tasks",
    "num_foreign_inspirations",
    "parent_id",
    "program_id",
    "scheduler_count",
    "selected_task",
    "task_local_iteration",
    "task_name",
}
_EXCLUDED_BEST_TRACKING_SUBSTRINGS = (
    "foreign_inspiration",
    "timeout",
)
_VISUALIZATION_CLAMP_BOUNDS = {
    "r2": (-1.0, 1.0),
}
_VISUALIZATION_OUTLIER_RATIO = 1_000.0


def _is_truthy_env(value: Optional[str]) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY_VALUES


def _sanitize_for_wandb(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if callable(value):
        return _SKIP

    if is_dataclass(value):
        serialized: Dict[str, Any] = {}
        for field_info in fields(value):
            if field_info.name.startswith("_"):
                continue
            nested_value = _sanitize_for_wandb(getattr(value, field_info.name))
            if nested_value is not _SKIP:
                serialized[field_info.name] = nested_value
        return serialized

    if isinstance(value, dict):
        serialized = {}
        for key, nested_value in value.items():
            safe_value = _sanitize_for_wandb(nested_value)
            if safe_value is not _SKIP:
                serialized[str(key)] = safe_value
        return serialized

    if isinstance(value, (list, tuple, set)):
        serialized_list: List[Any] = []
        for item in value:
            safe_item = _sanitize_for_wandb(item)
            if safe_item is not _SKIP:
                serialized_list.append(safe_item)
        return serialized_list

    return str(value)


def flatten_scalars(metrics: Dict[str, Any], prefix: Optional[str] = None) -> Dict[str, Any]:
    """Flatten scalar values in a metrics dictionary for logging."""
    flat: Dict[str, Any] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            key = str(key)
        target_key = f"{prefix}/{key}" if prefix else key
        if value is None or isinstance(value, (str, int, float, bool)):
            flat[target_key] = value
            continue
        if isinstance(value, Path):
            flat[target_key] = str(value)
    return flat


class WandbRunLogger:
    """Small wrapper around wandb with safe no-op behavior."""

    def __init__(self, config: Any, output_dir: str):
        self.config = config
        self.output_dir = str(Path(output_dir).resolve())
        self.enabled = False
        self.run = None
        self._wandb = None
        self._metric_namespace: Optional[str] = None
        self._step_metric: Optional[str] = None
        self._log_step = 0
        self._best_metric_history: Dict[str, List[tuple[int, float]]] = {}
        self._tracked_metric_history: Dict[str, List[tuple[int, float]]] = {}
        self._best_metric_state: Dict[str, float] = {}
        self._best_metric_summary: Dict[str, str] = {}
        self._artifact_base_name = "openevolve"

    def init_run(
        self,
        *,
        run_mode: str,
        config_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        runtime = self._resolve_runtime_settings()
        if not runtime["enabled"]:
            return

        resolved_step_metric = self._namespaced_key(step_metric, namespace=runtime["namespace"])

        try:
            wandb = importlib.import_module("wandb")
        except ImportError:
            logger.info("W&B requested but 'wandb' is not installed. Continuing without tracking.")
            return

        init_kwargs = {
            "project": runtime["project"],
            "entity": runtime["entity"],
            "id": runtime["run_id"],
            "name": self._resolve_runtime_string(runtime["name"], config_payload),
            "group": self._resolve_runtime_string(runtime["group"], config_payload),
            "job_type": self._resolve_runtime_string(runtime["job_type"], config_payload) or run_mode,
            "tags": self._resolve_runtime_tags(runtime["tags"], config_payload),
            "notes": runtime["notes"],
            "mode": runtime["mode"],
            "resume": runtime["resume"],
            "allow_val_change": runtime["allow_val_change"],
            "dir": self.output_dir,
            "config": _sanitize_for_wandb(config_payload or {}),
        }
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

        try:
            self._wandb = wandb
            self.run = wandb.init(**init_kwargs)
            if self.run is None:
                return
            self.enabled = True
            self._metric_namespace = runtime["namespace"]
            self._step_metric = resolved_step_metric
            self._log_step = 0
            self._artifact_base_name = self._sanitize_artifact_name(
                f"openevolve-{self.run.id}-{run_mode}"
            )

            if metadata:
                self.update_summary({"run_metadata": metadata})

            self._define_metrics(
                run_mode=run_mode,
                step_metric=resolved_step_metric,
                config_payload=config_payload,
            )

            if runtime["log_code"]:
                self.run.log_code(root=os.getcwd())
        except Exception as exc:
            logger.warning("Failed to initialize W&B tracking: %s", exc)
            self._disable()

    def log_metrics(self, metrics: Dict[str, Any], *, step: Optional[int] = None) -> None:
        if not self.enabled or not self.run:
            return

        payload = self._namespace_payload(flatten_scalars(metrics))
        if step is not None and self._step_metric and self._step_metric not in payload:
            payload[self._step_metric] = step
        payload = self._augment_best_so_far_metrics(payload)

        try:
            self._log_step += 1
            self.run.log(payload)
            self._record_best_metric_history(payload, step)
        except Exception as exc:
            logger.warning("Failed to log W&B metrics: %s", exc)
            self._disable()

    def update_summary(self, summary: Dict[str, Any]) -> None:
        if not self.enabled or not self.run:
            return

        try:
            safe_summary = _sanitize_for_wandb(summary)
            if isinstance(safe_summary, dict):
                self.run.summary.update(self._namespace_summary(safe_summary))
        except Exception as exc:
            logger.warning("Failed to update W&B summary: %s", exc)
            self._disable()

    def log_best_program_artifact(
        self,
        program_path: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        task_name: Optional[str] = None,
    ) -> None:
        if not self.enabled or not self.run or not self._should_log_best_program_artifact():
            return

        path = Path(program_path)
        if not path.exists():
            return

        artifact_name = self._artifact_name("best-program", task_name=task_name)
        aliases = ["latest"]
        iteration = (metadata or {}).get("iteration")
        if iteration is not None:
            aliases.append(f"iteration-{iteration}")
        self._log_artifact(
            artifact_name=artifact_name,
            artifact_type="program",
            files=self._best_program_files(path),
            metadata=metadata,
            aliases=aliases,
        )

    def log_checkpoint_artifact(
        self, checkpoint_path: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if not self.enabled or not self.run or not self._should_log_checkpoint_artifact():
            return

        checkpoint_root = Path(checkpoint_path)
        if not checkpoint_root.exists():
            return

        aliases = ["latest"]
        step_value = (metadata or {}).get("iteration") or (metadata or {}).get("global_iteration")
        if step_value is not None:
            aliases.append(f"step-{step_value}")

        self._log_artifact(
            artifact_name=self._artifact_name("checkpoint-metadata"),
            artifact_type="checkpoint",
            files=self._checkpoint_files(checkpoint_root),
            metadata=metadata,
            aliases=aliases,
        )

    def log_file_artifact(
        self,
        file_path: str,
        *,
        artifact_name: str,
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or not self.run:
            return

        file_root = Path(file_path)
        if not file_root.exists():
            return

        self._log_artifact(
            artifact_name=self._artifact_name(artifact_name),
            artifact_type=artifact_type,
            files=[file_root],
            metadata=metadata,
            aliases=["latest"],
        )

    def finish(self) -> None:
        if not self.enabled or not self.run:
            return

        try:
            self._log_best_metric_plots()
            self.run.finish()
        except Exception as exc:
            logger.warning("Failed to finish W&B run cleanly: %s", exc)
        finally:
            self._disable()

    def _define_metrics(
        self,
        *,
        run_mode: str,
        step_metric: Optional[str],
        config_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.run or not self._wandb:
            return

        if step_metric:
            self._wandb.define_metric(step_metric)

        if run_mode == "single-task":
            if step_metric:
                self._wandb.define_metric(self._metric_pattern("*"), step_metric=step_metric)
            current_fitness_key = self._metric_key("current_fitness")
            best_fitness_key = self._metric_key("best_fitness_so_far")
            self._wandb.define_metric(current_fitness_key, summary="last")
            self._wandb.define_metric(best_fitness_key, summary="max")
            self._best_metric_summary[best_fitness_key] = "max"
        elif run_mode == "multitask":
            if step_metric:
                self._wandb.define_metric(self._metric_pattern("multitask/*"), step_metric=step_metric)
                self._wandb.define_metric(self._metric_pattern("task/*"), step_metric=step_metric)
            self._wandb.define_metric(self._metric_key("multitask/current_fitness"), summary="last")
            self._wandb.define_metric(
                self._metric_key("multitask/best_task_fitness_max"),
                summary="max",
            )
            for task_name in self._extract_task_names(config_payload):
                task_prefix = f"task/{task_name}"
                self._wandb.define_metric(
                    self._metric_key(f"{task_prefix}/current_fitness"),
                    summary="last",
                )
                self._wandb.define_metric(
                    self._metric_key(f"{task_prefix}/best_fitness"),
                    summary="max",
                )

    def _record_best_metric_history(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
        history_step = step
        if self._step_metric:
            metric_step = metrics.get(self._step_metric)
            if isinstance(metric_step, (int, float)) and not isinstance(metric_step, bool):
                history_step = int(metric_step)

        if history_step is None:
            return

        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            if (
                key == "best_fitness_so_far"
                or key.endswith("/best_fitness")
                or key.endswith("/best_fitness_so_far")
            ):
                self._best_metric_history.setdefault(key, []).append((history_step, numeric_value))
            metric_summary = self._tracked_metric_summary(key)
            if metric_summary:
                self._tracked_metric_history.setdefault(key, []).append((history_step, numeric_value))

    def _log_best_metric_plots(self) -> None:
        if not self.enabled or not self.run or not self._wandb:
            return

        for metric_name, history in self._best_metric_history.items():
            if len(history) < 2:
                continue

            sorted_history = sorted(history, key=lambda item: item[0])
            xs = [step for step, _ in sorted_history]
            ys = [value for _, value in sorted_history]
            plot_key = f"plots/{metric_name.replace('/', '_')}"
            try:
                self.run.log(
                    {
                        plot_key: self._wandb.plot.line_series(
                            xs=xs,
                            ys=[ys],
                            keys=[metric_name],
                            title=f"{metric_name} over time",
                            xname=self._step_metric or "step",
                        )
                    }
                )
            except Exception:
                logger.debug("Skipping custom W&B plot for %s", metric_name, exc_info=True)

        for metric_name, history in self._tracked_metric_history.items():
            if len(history) < 2:
                continue

            summary = self._tracked_metric_summary(metric_name)
            if summary is None:
                continue

            best_metric_name = self._best_series_name(metric_name)
            sorted_history = sorted(history, key=lambda item: item[0])
            xs = [step for step, _ in sorted_history]
            ys = self._best_plot_values(
                metric_name,
                [value for _, value in sorted_history],
                summary,
            )
            plot_key = f"plots/{best_metric_name.replace('/', '_')}"
            try:
                self.run.log(
                    {
                        plot_key: self._wandb.plot.line_series(
                            xs=xs,
                            ys=[ys],
                            keys=[best_metric_name],
                            title=f"{best_metric_name} over time",
                            xname=self._step_metric or "step",
                        )
                    }
                )
            except Exception:
                logger.debug("Skipping custom W&B plot for %s", best_metric_name, exc_info=True)

    def _log_artifact(
        self,
        *,
        artifact_name: str,
        artifact_type: str,
        files: Iterable[Path],
        metadata: Optional[Dict[str, Any]],
        aliases: Optional[List[str]] = None,
    ) -> None:
        if not self.enabled or not self.run or not self._wandb:
            return

        file_list = [path.resolve() for path in files if path.exists()]
        if not file_list:
            return

        try:
            artifact = self._wandb.Artifact(
                artifact_name,
                type=artifact_type,
                metadata=_sanitize_for_wandb(metadata or {}),
            )
            base_dir = (
                file_list[0].parent
                if len(file_list) == 1 and file_list[0].is_file()
                else self._common_parent(file_list)
            )
            for path in file_list:
                try:
                    relative_name = str(path.relative_to(base_dir)) if base_dir else path.name
                except ValueError:
                    relative_name = path.name
                artifact.add_file(str(path), name=relative_name)
            self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        except Exception as exc:
            logger.warning("Failed to log W&B artifact '%s': %s", artifact_name, exc)

    def _best_program_files(self, program_path: Path) -> List[Path]:
        files = [program_path]
        info_path = program_path.with_name("best_program_info.json")
        if info_path.exists():
            files.append(info_path)
        return files

    def _checkpoint_files(self, checkpoint_root: Path) -> List[Path]:
        files: List[Path] = []
        files.extend(sorted(checkpoint_root.glob("*.json")))
        files.extend(sorted(checkpoint_root.glob("best_program*")))
        files.extend(sorted(checkpoint_root.glob("tasks/*/task_state.json")))
        files.extend(sorted(checkpoint_root.glob("tasks/*/best_program_info.json")))
        files.extend(sorted(checkpoint_root.glob("tasks/*/best_program.*")))
        return files

    def _artifact_name(self, suffix: str, *, task_name: Optional[str] = None) -> str:
        if self._metric_namespace:
            suffix = f"{self._metric_namespace.replace('/', '-')}-{suffix}"
        if task_name:
            suffix = f"{task_name}-{suffix}"
        return self._sanitize_artifact_name(f"{self._artifact_base_name}-{suffix}")

    def _metric_key(self, key: str) -> str:
        return self._namespaced_key(key) or key

    def _metric_pattern(self, pattern: str) -> str:
        return self._namespaced_key(pattern) or pattern

    def _namespaced_key(self, key: Optional[str], *, namespace: Optional[str] = None) -> Optional[str]:
        if not key:
            return key
        resolved_namespace = self._metric_namespace if namespace is None else namespace
        if not resolved_namespace:
            return key
        return f"{resolved_namespace}/{key}"

    def _namespace_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._metric_namespace:
            return dict(payload)
        return {
            self._namespaced_key(key) or key: value
            for key, value in payload.items()
        }

    def _namespace_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        if not self._metric_namespace:
            return dict(summary)
        return {
            self._namespaced_key(str(key)) or str(key): value
            for key, value in summary.items()
        }

    def _template_replacements(self, config_payload: Any) -> Dict[str, str]:
        model_name = self._extract_model_name(config_payload)
        edit_mode = self._extract_edit_mode(config_payload)
        foreign_scores = self._extract_foreign_scores_mode(config_payload)
        prompt_budget = self._extract_prompt_budget_mode(config_payload)
        trigger_mode = self._extract_trigger_mode(config_payload)
        reward_mode = self._extract_reward_mode(config_payload)
        task_names = self._extract_task_names(config_payload)
        replacements = {
            "model": self._sanitize_run_name_component(model_name),
            "primary_model": self._sanitize_run_name_component(model_name),
            "trigger_mode": self._sanitize_run_name_component(trigger_mode),
            "reward_mode": self._sanitize_run_name_component(reward_mode),
            "edit_mode": self._sanitize_run_name_component(edit_mode),
            "foreign_scores": self._sanitize_run_name_component(foreign_scores),
            "prompt_budget": self._sanitize_run_name_component(prompt_budget),
        }
        if task_names:
            sanitized_tasks = [
                sanitized
                for sanitized in (
                    self._sanitize_run_name_component(task_name) for task_name in task_names
                )
                if sanitized
            ]
            if sanitized_tasks:
                replacements["task_names"] = "-".join(sanitized_tasks)
                replacements["task_count"] = str(len(sanitized_tasks))
        return replacements

    def _resolve_runtime_string(
        self,
        configured_value: Optional[str],
        config_payload: Any,
    ) -> Optional[str]:
        if not configured_value or "{" not in configured_value:
            return configured_value

        resolved_value = configured_value
        for key, value in self._template_replacements(config_payload).items():
            if value:
                resolved_value = resolved_value.replace(f"{{{key}}}", value)
        return resolved_value

    def _resolve_runtime_tags(
        self,
        configured_tags: List[str],
        config_payload: Any,
    ) -> List[str]:
        return [
            resolved
            for resolved in (
                self._resolve_runtime_string(tag, config_payload) for tag in (configured_tags or [])
            )
            if resolved
        ]

    def _extract_model_name(self, config_payload: Any) -> Optional[str]:
        if config_payload is None:
            return None

        llm_cfg = None
        if isinstance(config_payload, dict):
            llm_cfg = config_payload.get("llm")
            if llm_cfg is None and config_payload.get("base_config") is not None:
                return self._extract_model_name(config_payload.get("base_config"))
        else:
            llm_cfg = getattr(config_payload, "llm", None)
            base_config = getattr(config_payload, "base_config", None)
            if llm_cfg is None and base_config is not None:
                return self._extract_model_name(base_config)

        if llm_cfg is None:
            return None

        primary_model = (
            llm_cfg.get("primary_model")
            if isinstance(llm_cfg, dict)
            else getattr(llm_cfg, "primary_model", None)
        )
        if primary_model:
            return str(primary_model)

        models = llm_cfg.get("models") if isinstance(llm_cfg, dict) else getattr(llm_cfg, "models", None)
        if models:
            first_model = models[0]
            if isinstance(first_model, dict):
                name = first_model.get("name")
            else:
                name = getattr(first_model, "name", None)
            if name:
                return str(name)

        return None

    def _extract_edit_mode(self, config_payload: Any) -> Optional[str]:
        if config_payload is None:
            return None

        if isinstance(config_payload, dict):
            if "diff_based_evolution" in config_payload:
                return "diff" if bool(config_payload.get("diff_based_evolution")) else "full"
            if config_payload.get("base_config") is not None:
                return self._extract_edit_mode(config_payload.get("base_config"))
            if config_payload.get("multitask") is not None:
                return self._extract_edit_mode(config_payload.get("multitask"))
            return None

        diff_based_evolution = getattr(config_payload, "diff_based_evolution", None)
        if diff_based_evolution is not None:
            return "diff" if bool(diff_based_evolution) else "full"

        base_config = getattr(config_payload, "base_config", None)
        if base_config is not None:
            return self._extract_edit_mode(base_config)

        multitask_config = getattr(config_payload, "multitask", None)
        if multitask_config is not None:
            return self._extract_edit_mode(multitask_config)

        return None

    def _extract_trigger_mode(self, config_payload: Any) -> Optional[str]:
        if config_payload is None:
            return None

        if isinstance(config_payload, dict):
            foreign_inspirations = config_payload.get("foreign_inspirations")
            if foreign_inspirations is not None:
                if isinstance(foreign_inspirations, dict):
                    return str(foreign_inspirations.get("trigger_mode") or "periodic")
                trigger_mode = getattr(foreign_inspirations, "trigger_mode", None)
                return str(trigger_mode or "periodic")

            if config_payload.get("multitask") is not None:
                return self._extract_trigger_mode(config_payload.get("multitask"))
            return None

        foreign_inspirations = getattr(config_payload, "foreign_inspirations", None)
        if foreign_inspirations is not None:
            trigger_mode = getattr(foreign_inspirations, "trigger_mode", None)
            return str(trigger_mode or "periodic")

        multitask_config = getattr(config_payload, "multitask", None)
        if multitask_config is not None:
            return self._extract_trigger_mode(multitask_config)

        return None

    def _extract_foreign_scores_mode(self, config_payload: Any) -> Optional[str]:
        if config_payload is None:
            return None

        if isinstance(config_payload, dict):
            foreign_inspirations = config_payload.get("foreign_inspirations")
            if foreign_inspirations is not None:
                include_scores = True
                if isinstance(foreign_inspirations, dict):
                    include_scores = bool(foreign_inspirations.get("include_scores", True))
                else:
                    include_scores = bool(getattr(foreign_inspirations, "include_scores", True))
                return "scores" if include_scores else "noscores"

            if config_payload.get("multitask") is not None:
                return self._extract_foreign_scores_mode(config_payload.get("multitask"))
            return None

        foreign_inspirations = getattr(config_payload, "foreign_inspirations", None)
        if foreign_inspirations is not None:
            include_scores = bool(getattr(foreign_inspirations, "include_scores", True))
            return "scores" if include_scores else "noscores"

        multitask_config = getattr(config_payload, "multitask", None)
        if multitask_config is not None:
            return self._extract_foreign_scores_mode(multitask_config)

        return None

    def _extract_reward_mode(self, config_payload: Any) -> Optional[str]:
        if config_payload is None:
            return None

        if isinstance(config_payload, dict):
            foreign_inspirations = config_payload.get("foreign_inspirations")
            if foreign_inspirations is not None:
                if isinstance(foreign_inspirations, dict):
                    return str(foreign_inspirations.get("reward_mode") or "sparse")
                reward_mode = getattr(foreign_inspirations, "reward_mode", None)
                return str(reward_mode or "sparse")

            if config_payload.get("multitask") is not None:
                return self._extract_reward_mode(config_payload.get("multitask"))
            return None

        foreign_inspirations = getattr(config_payload, "foreign_inspirations", None)
        if foreign_inspirations is not None:
            reward_mode = getattr(foreign_inspirations, "reward_mode", None)
            return str(reward_mode or "sparse")

        multitask_config = getattr(config_payload, "multitask", None)
        if multitask_config is not None:
            return self._extract_reward_mode(multitask_config)

        return None

    def _extract_prompt_budget_mode(self, config_payload: Any) -> Optional[str]:
        if config_payload is None:
            return None

        if isinstance(config_payload, dict):
            foreign_inspirations = config_payload.get("foreign_inspirations")
            if foreign_inspirations is not None:
                if isinstance(foreign_inspirations, dict):
                    prompt_overrides = foreign_inspirations.get("prompt_overrides")
                else:
                    prompt_overrides = getattr(foreign_inspirations, "prompt_overrides", None)
                return "xferbudget" if prompt_overrides is not None else "basebudget"

            if config_payload.get("multitask") is not None:
                return self._extract_prompt_budget_mode(config_payload.get("multitask"))
            return None

        foreign_inspirations = getattr(config_payload, "foreign_inspirations", None)
        if foreign_inspirations is not None:
            prompt_overrides = getattr(foreign_inspirations, "prompt_overrides", None)
            return "xferbudget" if prompt_overrides is not None else "basebudget"

        multitask_config = getattr(config_payload, "multitask", None)
        if multitask_config is not None:
            return self._extract_prompt_budget_mode(multitask_config)

        return None

    def _extract_task_names(self, config_payload: Any) -> List[str]:
        if config_payload is None:
            return []

        multitask_cfg = None
        if isinstance(config_payload, dict):
            multitask_cfg = config_payload.get("multitask", config_payload)
        else:
            multitask_cfg = getattr(config_payload, "multitask", config_payload)

        tasks = (
            multitask_cfg.get("tasks")
            if isinstance(multitask_cfg, dict)
            else getattr(multitask_cfg, "tasks", None)
        )
        if not tasks:
            return []

        task_names: List[str] = []
        for task in tasks:
            if isinstance(task, dict):
                task_name = task.get("task_name") or task.get("name")
            else:
                task_name = getattr(task, "task_name", None) or getattr(task, "name", None)
            if task_name:
                task_names.append(str(task_name))
        return task_names

    def _sanitize_run_name_component(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")

    def _resolve_runtime_settings(self) -> Dict[str, Any]:
        config = self.config
        enabled = bool(getattr(config, "enabled", False) or _is_truthy_env(os.getenv("OPENEVOLVE_WANDB")))
        mode = os.getenv("WANDB_MODE") or getattr(config, "mode", None)
        if mode == "disabled":
            enabled = False

        return {
            "enabled": enabled,
            "project": os.getenv("WANDB_PROJECT") or getattr(config, "project", None),
            "entity": os.getenv("WANDB_ENTITY") or getattr(config, "entity", None),
            "run_id": getattr(config, "run_id", None),
            "resume": getattr(config, "resume", None),
            "allow_val_change": getattr(config, "allow_val_change", None),
            "name": getattr(config, "name", None),
            "group": getattr(config, "group", None),
            "job_type": getattr(config, "job_type", None),
            "tags": list(getattr(config, "tags", []) or []),
            "notes": getattr(config, "notes", None),
            "mode": mode,
            "namespace": getattr(config, "namespace", None),
            "log_code": bool(getattr(config, "log_code", False)),
        }

    def _should_log_best_program_artifact(self) -> bool:
        return bool(getattr(self.config, "log_best_program_artifact", True))

    def _should_log_checkpoint_artifact(self) -> bool:
        return bool(getattr(self.config, "log_checkpoint_artifact", False))

    def _sanitize_artifact_name(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "openevolve"

    def _common_parent(self, paths: List[Path]) -> Optional[Path]:
        if not paths:
            return None
        common_path = Path(os.path.commonpath([str(path.resolve()) for path in paths]))
        return common_path if common_path.exists() else None

    def _augment_best_so_far_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        augmented = dict(payload)
        for metric_name, value in payload.items():
            summary = self._tracked_metric_summary(metric_name)
            if summary is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue

            numeric_value = self._normalize_best_tracking_value(metric_name, numeric_value)

            best_metric_name = self._best_series_name(metric_name)
            previous_best = self._best_metric_state.get(best_metric_name)
            if previous_best is None:
                next_best = numeric_value
            elif summary == "min":
                next_best = min(previous_best, numeric_value)
            else:
                next_best = max(previous_best, numeric_value)

            self._best_metric_state[best_metric_name] = next_best
            augmented.setdefault(best_metric_name, next_best)
            self._define_best_metric_summary(best_metric_name, summary)
        return augmented

    def _tracked_metric_summary(self, metric_name: str) -> Optional[str]:
        leaf_name = metric_name.rsplit("/", 1)[-1].lower()
        if not leaf_name:
            return None
        if leaf_name in _EXCLUDED_BEST_TRACKING_METRICS:
            return None
        if leaf_name.startswith("best_") or leaf_name.endswith("_so_far"):
            return None
        if leaf_name.startswith("num_"):
            return None
        if metric_name == self._step_metric or leaf_name == self._step_metric:
            return None
        if leaf_name.endswith("_sec") or leaf_name.endswith("_seconds"):
            return None
        if "iteration" in leaf_name:
            return None
        if any(token in leaf_name for token in _EXCLUDED_BEST_TRACKING_SUBSTRINGS):
            return None
        if any(token in leaf_name for token in _MINIMIZE_METRIC_TOKENS):
            return "min"
        if any(token in leaf_name for token in _MAXIMIZE_METRIC_TOKENS):
            return "max"
        return "max"

    def _best_series_name(self, metric_name: str) -> str:
        if "/" not in metric_name:
            return f"best_{metric_name}_so_far"
        prefix, leaf_name = metric_name.rsplit("/", 1)
        return f"{prefix}/best_{leaf_name}_so_far"

    def _define_best_metric_summary(self, metric_name: str, summary: str) -> None:
        if self._best_metric_summary.get(metric_name) == summary:
            return
        self._best_metric_summary[metric_name] = summary
        if not self.run or not self._wandb:
            return
        try:
            self._wandb.define_metric(metric_name, summary=summary)
        except Exception:
            logger.debug("Skipping dynamic W&B metric definition for %s", metric_name, exc_info=True)

    def _running_best_values(self, values: List[float], summary: str) -> List[float]:
        running: List[float] = []
        best_value: Optional[float] = None
        for value in values:
            if best_value is None:
                best_value = value
            elif summary == "min":
                best_value = min(best_value, value)
            else:
                best_value = max(best_value, value)
            running.append(best_value)
        return running

    def _best_plot_values(self, metric_name: str, values: List[float], summary: str) -> List[float]:
        normalized_values = [
            self._normalize_best_tracking_value(metric_name, value)
            for value in values
        ]
        running_values = self._running_best_values(normalized_values, summary)
        if summary != "min":
            return running_values
        return self._clip_min_plot_outlier(running_values)

    def _clip_min_plot_outlier(self, values: List[float]) -> List[float]:
        distinct_values = sorted(
            {value for value in values if math.isfinite(value)},
            reverse=True,
        )
        if len(distinct_values) < 2:
            return values

        largest_value = distinct_values[0]
        next_largest_value = distinct_values[1]
        reference_scale = max(abs(next_largest_value), 1e-12)
        if largest_value <= next_largest_value:
            return values
        if abs(largest_value) / reference_scale < _VISUALIZATION_OUTLIER_RATIO:
            return values

        return [
            min(value, next_largest_value) if math.isfinite(value) else value
            for value in values
        ]

    def _normalize_best_tracking_value(self, metric_name: str, value: float) -> float:
        leaf_name = metric_name.rsplit("/", 1)[-1].lower()
        bounds = _VISUALIZATION_CLAMP_BOUNDS.get(leaf_name)
        if bounds is None:
            return value

        lower, upper = bounds
        return max(lower, min(upper, value))

    def _disable(self) -> None:
        self.enabled = False
        self.run = None
        self._wandb = None
        self._metric_namespace = None
        self._step_metric = None
        self._log_step = 0
        self._best_metric_history = {}
        self._tracked_metric_history = {}
        self._best_metric_state = {}
        self._best_metric_summary = {}


def create_wandb_logger(config: Any, output_dir: str) -> WandbRunLogger:
    return WandbRunLogger(config=config, output_dir=output_dir)
