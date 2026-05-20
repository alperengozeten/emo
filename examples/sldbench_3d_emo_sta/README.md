# SLDBench 3D EMO-STA

This directory is the EMO-STA-specific version of a narrow SLDBench subset. It
is separate from the original standalone [`examples/sldbench`](../sldbench)
example so the shared-then-adapt workflow can stay cleanly distinct from
the original single-task setup.

This family includes exactly two public 3D scalar-output tasks:

- `vocab_scaling_law`
- `data_constrained_scaling_law`

Both tasks are canonicalized before they reach the candidate program. The
candidate always sees:

```text
data_points.shape == (N, 3)
columns == [model_size_like, diversity_like, total_data_like]
```

The shared phase optimizes average score across both tasks. Spawned task-local
adaptation runs warmstart from the shared checkpoint population, but group-local
coefficients are always fitted locally during evaluation. EMO-STA shares the law
code and fitter, not fitted coefficients across tasks or groups.

Use the EMO-STA manifest at:

```text
multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml
```

Synthetic smoke-test mode is available for offline tests and local development:

```bash
SLDBENCH_3D_USE_SYNTHETIC_FIXTURE=1 SLDBENCH_3D_TASK_ID=all \
python examples/sldbench_3d_emo_sta/evaluator.py \
  examples/sldbench_3d_emo_sta/initial_program.py
```

For real dataset runs, `SLDBENCH_CACHE_DIR` can be pointed at a writable Hugging
Face cache directory when the default cache location is unavailable or
read-only.
