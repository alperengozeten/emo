# Function Minimization for EMO-STA

This directory contains the `multi_task_shared_then_adapt` version of
function minimization. It is separate from the standalone
[examples/function_minimization](../function_minimization) example, which
remains a single-objective benchmark.

The EMO-STA family uses four public 2D objective-function tasks:

- `fm_sincosxy_2d`
- `fm_ackley_2d`
- `fm_rastrigin_2d`
- `fm_rosenbrock_2d`

All four tasks share one evolving program representation: a generic
derivative-free search algorithm that receives `objective_fn` and `bounds`
from the evaluator. Shared mode scores one candidate on all four landscapes and
optimizes the average score. Task-specific adaptation then warmstarts from the
shared checkpoint population and continues on one exact landscape at a time.

Use this directory through the EMO-STA manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml
```

Outputs are written under:

```text
multi_task_shared_then_adapt/results/function_minimization/<run_name>/
```
