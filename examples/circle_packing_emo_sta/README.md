# Circle Packing EMO-STA

This is the multi-task shared-then-adapt version of circle packing. It is
separate from the original standalone examples in
`examples/circle_packing/` and `examples/circle_packing_with_artifacts/`, which
remain fixed `n=26` examples.

The EMO-STA family uses four public tasks that all share the same problem
structure: pack circles of varying radii into the unit square `[0,1] x [0,1]`
without overlaps while maximizing the sum of radii.

The public EMO-STA task IDs are:

- `cp_n20`
- `cp_n22`
- `cp_n24`
- `cp_n26`

The family also has evaluation-only unseen-`n` holdouts:

- `cp_n21`
- `cp_n23`
- `cp_n25`

These holdouts are not used in shared training, checkpoint spawning,
task-specific adaptation, or direct single-task baselines. They are only used
after those phases finish to measure transfer to nearby unseen `n` values.

All four tasks use the same generic candidate interface:

```python
def construct_packing(n: int):
    ...
    return centers, radii, sum_radii

def run_packing(n: int):
    return construct_packing(n)
```

The shared EMO-STA phase evaluates the same evolving program on all four tasks
and optimizes the average normalized score across them. The spawn step then
warmstarts four task-specific checkpoints from the shared checkpoint population,
and each task-specific adaptation run adapts to one exact `n`.

The evaluator chooses tasks through `CIRCLE_PACKING_TASK_ID`:

- `CIRCLE_PACKING_TASK_ID=all` evaluates the shared family objective
- `CIRCLE_PACKING_TASK_ID=cp_n20|cp_n22|cp_n24|cp_n26` evaluates one seen task
- `CIRCLE_PACKING_TASK_ID=cp_n21|cp_n23|cp_n25` evaluates one holdout task

Quick smoke checks:

```bash
CIRCLE_PACKING_TASK_ID=all python examples/circle_packing_emo_sta/evaluator.py \
  examples/circle_packing_emo_sta/initial_program.py
```

```bash
CIRCLE_PACKING_TASK_ID=cp_n26 python examples/circle_packing_emo_sta/evaluator.py \
  examples/circle_packing_emo_sta/initial_program.py
```

```bash
CIRCLE_PACKING_TASK_ID=cp_n21 python examples/circle_packing_emo_sta/evaluator.py \
  examples/circle_packing_emo_sta/initial_program.py
```

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml \
  --shared-iterations 2 \
  --adaptation-iterations 2 \
  --baseline-iterations 2
```

Use `--skip-holdouts` to skip the post-hoc holdout evaluation phase.
