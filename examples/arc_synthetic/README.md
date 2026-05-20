# ARC Synthetic Example

This example is a deterministic ARC-style prototype for multitask transfer experiments.
It does not require the external ARC dataset. Instead it uses three fixed synthetic
subtasks that share the same program interface:

- `rotate_90_cw`
- `shift_right_wrap`
- `flip_horizontal`

All three tasks use the same candidate program shape:

```python
def transform_grid_attempt_1(grid):
    ...

def transform_grid_attempt_2(grid):
    ...
```

The shared evaluator selects the task through `ARC_SYNTHETIC_TASK_ID`, scores the
visible train examples with `combined_score`, and also logs a hidden held-out score
as `heldout_score`.

## Single-Task Run

Run one synthetic ARC task directly from the repository root:

```bash
ARC_SYNTHETIC_TASK_ID=rotate_90_cw \
PYTHONPATH=. python openevolve-run.py \
  examples/arc_synthetic/initial_program.py \
  examples/arc_synthetic/evaluator.py \
  --config examples/arc_synthetic/config_rotate_90_cw.yaml \
  --output examples/arc_synthetic/results/rotate_90_cw
```

You can swap in the matching config for the other tasks:

- `examples/arc_synthetic/config_shift_right_wrap.yaml`
- `examples/arc_synthetic/config_flip_horizontal.yaml`

## Multitask Run

The multitask prototype is wired through the standard multitask CLI and file layout:

```bash
PYTHONPATH=. python -m openevolve.multitask.cli \
  multi_task_evolve/multitask_arc_synthetic_rotate_shift_flip.yaml
```

That config lives under `multi_task_evolve/` and references the shared
`examples/arc_synthetic/` evaluator and initial program using the same relative-path
convention as the existing multitask examples.
