# Circle Packing Rectangle EMO-STA

This is the multi-task shared-then-adapt version of circle packing in
rectangles of perimeter 4. It is separate from the original square
circle-packing examples in `examples/circle_packing/`,
`examples/circle_packing_with_artifacts/`, and the square EMO-STA family in
`examples/circle_packing_emo_sta/`.

This family uses four public tasks that all share the same program interface
and evaluator family. Each task packs varying-radius circles into a rectangle
with:

- width `alpha`
- height `2 - alpha`
- perimeter `4`
- `0 < alpha <= 1`

The public EMO-STA task IDs are:

- `cp_rect_n20`
- `cp_rect_n21`
- `cp_rect_n22`
- `cp_rect_n23`

All four tasks use the same generic candidate interface:

```python
def construct_packing(n: int):
    ...
    return centers, radii, alpha, sum_radii

def run_packing(n: int):
    return construct_packing(n)
```

The shared EMO-STA phase evaluates the same evolving program on all four tasks
and optimizes the average normalized score across them. The spawn step then
warmstarts four task-specific checkpoints from the shared checkpoint
population, and each task-specific adaptation run adapts to one exact
value of `n`.

The evaluator chooses tasks through `CIRCLE_PACKING_RECTANGLE_TASK_ID`:

- `CIRCLE_PACKING_RECTANGLE_TASK_ID=all` evaluates the shared family objective
- `CIRCLE_PACKING_RECTANGLE_TASK_ID=cp_rect_n20|cp_rect_n21|cp_rect_n22|cp_rect_n23`
  evaluates one task

Pinned public scoring anchors:

- `n=20`: `2.305`
- `n=21`: `2.365`
- `n=22`: `2.425`
- `n=23`: `2.484`

These are fixed normalization anchors for reproducibility. They are not clipped
at `1.0`, so a valid packing may score above `1.0`. Newer work has reported a
slightly higher `n=21` value around `2.3658`, but this EMO-STA evaluator keeps
the pinned anchor at `2.365`.

The evaluator validates:

- `alpha`
- rectangle containment inside `[0, alpha] x [0, 2 - alpha]`
- pairwise non-overlap
- output shapes and finiteness

Post-hoc OOD evaluation is available for frozen finished EMO-STA runs on nearby
unseen values of `n`:

- `cp_rect_n19`: `target_sum_radii = 2.241`
- `cp_rect_n24`: `target_sum_radii = 2.535`
- `cp_rect_n25`: `target_sum_radii = 2.592`

These OOD tasks are available only through explicit evaluator selection or
`multi_task_shared_then_adapt/scripts/run_posthoc_ood_evaluation.py`. They are not
part of `CIRCLE_PACKING_RECTANGLE_TASK_ID=all`, which remains the four training
tasks listed above. The OOD anchors are pinned for reproducible normalized
scoring, and the default post-hoc utility evaluates both backward and forward
nearby-`n` tasks. The default post-hoc OOD set remains `cp_rect_n19` and
`cp_rect_n24`; include `cp_rect_n25` with `--ood-task-ids` to evaluate it.

Quick smoke checks:

```bash
CIRCLE_PACKING_RECTANGLE_TASK_ID=all python examples/circle_packing_rectangle_emo_sta/evaluator.py \
  examples/circle_packing_rectangle_emo_sta/initial_program.py
```

```bash
CIRCLE_PACKING_RECTANGLE_TASK_ID=cp_rect_n21 python examples/circle_packing_rectangle_emo_sta/evaluator.py \
  examples/circle_packing_rectangle_emo_sta/initial_program.py
```

```bash
CIRCLE_PACKING_RECTANGLE_TASK_ID=cp_rect_n24 python examples/circle_packing_rectangle_emo_sta/evaluator.py \
  examples/circle_packing_rectangle_emo_sta/initial_program.py
```

```bash
CIRCLE_PACKING_RECTANGLE_TASK_ID=cp_rect_n25 python examples/circle_packing_rectangle_emo_sta/evaluator.py \
  examples/circle_packing_rectangle_emo_sta/initial_program.py
```

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml \
  --shared-iterations 4 \
  --adaptation-iterations 2 \
  --baseline-iterations 3
```
