"""OpenEvolve package exports."""

from __future__ import annotations

from importlib import import_module

from openevolve._version import __version__

__all__ = [
    "Config",
    "OpenEvolve",
    "__version__",
    "run_evolution",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
]


def __getattr__(name: str):
    if name == "Config":
        return import_module("openevolve.config").Config
    if name == "OpenEvolve":
        return import_module("openevolve.controller").OpenEvolve
    if name in {"run_evolution", "evolve_function", "evolve_algorithm", "evolve_code", "EvolutionResult"}:
        return getattr(import_module("openevolve.api"), name)
    raise AttributeError(f"module 'openevolve' has no attribute {name!r}")
