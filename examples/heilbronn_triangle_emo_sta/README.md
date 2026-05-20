# Heilbronn Triangle EMO-STA

This is the multi-task shared-then-adapt version of the Heilbronn triangle
problem for a unit-area triangle. It is separate from other EMO-STA families and
keeps the same shared-run, task-spawn, task-adaptation, and direct-baseline
workflow used elsewhere in `multi_task_shared_then_adapt`.

The family uses four public tasks that differ only by the number of points:

- `heil_tri_n9`
- `heil_tri_n10`
- `heil_tri_n11`
- `heil_tri_n12`

All tasks use the same canonical unit-area triangle:

- `A = (0.0, 0.0)`
- `B = (2.0, 0.0)`
- `C = (0.0, 1.0)`

The evolving program keeps one generic interface across all phases:

```python
def construct_points(n: int):
    ...
    return points, min_area

def run_heilbronn(n: int):
    return construct_points(n)
```

The shared EMO-STA phase evaluates the same evolving program on all four tasks
and optimizes the average normalized score across them. The spawn step then
warmstarts four task-specific checkpoints from the shared checkpoint
population, and each task-specific adaptation run adapts to one exact
value of `n`.

The evaluator computes the exact minimum triangle area over all triples of
returned points, validates containment inside the canonical triangle, and uses
public pinned normalization anchors for reproducible scoring.

Post-hoc OOD evaluation is available for frozen finished EMO-STA runs on nearby
unseen values of `n`:

- `heil_tri_n8`: `target_min_area = 0.06778914101959856`
- `heil_tri_n13`: `target_min_area = 0.02456425934867466`
- `heil_tri_n14`: `target_min_area = 0.02377577301721215`

These OOD tasks are available only through explicit evaluator selection or
`multi_task_shared_then_adapt/scripts/run_posthoc_ood_evaluation.py`. They are not
part of `HEILBRONN_TRIANGLE_TASK_ID=all`, which remains the four training tasks
listed above. The OOD anchors are pinned for reproducible normalized scoring,
and the default post-hoc utility evaluates both backward and forward nearby-`n`
tasks. The default post-hoc OOD set remains `heil_tri_n8` and `heil_tri_n13`;
include `heil_tri_n14` with `--ood-task-ids` to evaluate it.

Quick smoke checks:

```bash
HEILBRONN_TRIANGLE_TASK_ID=all python examples/heilbronn_triangle_emo_sta/evaluator.py \
  examples/heilbronn_triangle_emo_sta/initial_program.py
```

```bash
HEILBRONN_TRIANGLE_TASK_ID=heil_tri_n11 python examples/heilbronn_triangle_emo_sta/evaluator.py \
  examples/heilbronn_triangle_emo_sta/initial_program.py
```

```bash
HEILBRONN_TRIANGLE_TASK_ID=heil_tri_n13 python examples/heilbronn_triangle_emo_sta/evaluator.py \
  examples/heilbronn_triangle_emo_sta/initial_program.py
```

```bash
HEILBRONN_TRIANGLE_TASK_ID=heil_tri_n14 python examples/heilbronn_triangle_emo_sta/evaluator.py \
  examples/heilbronn_triangle_emo_sta/initial_program.py
```

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml \
  --shared-iterations 2 \
  --adaptation-iterations 2 \
  --baseline-iterations 2
```
