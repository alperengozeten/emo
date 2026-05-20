# Rust Adaptive Sort EMO-STA

This is the multi-task shared-then-adapt version of the Rust adaptive sort
example. It is intentionally separate from
[examples/rust_adaptive_sort](../rust_adaptive_sort), which remains the original
standalone single-evaluator example.

This EMO-STA family uses four public deterministic task regimes:

- `ras_random`
- `ras_nearly_sorted`
- `ras_reverse_sorted`
- `ras_duplicates`

Why it fits EMO-STA:

- all tasks share one evolving artifact: `adaptive_sort`
- the shared phase optimizes the average score across the four regimes
- spawned task-local checkpoints warmstart directly from the shared archive
- task-specific runs can then adapt to one exact regime

The benchmark is deterministic and compile-once/run-many:

- the Python evaluator creates a temporary Cargo project
- the candidate code is copied into `src/lib.rs`
- `cargo build --release` runs once per candidate
- the compiled benchmark binary is then executed once per selected task

`partially_sorted` is intentionally not part of the initial shared family. It
can be added later as a post-hoc holdout/generalization check without changing
the core EMO-STA workflow.

## Files

- `initial_program.rs`: starting generic adaptive sorting baseline
- `evaluator.py`: synchronous EMO-STA evaluator
- `config.yaml`: OpenEvolve config for the Rust EMO-STA family
- `requirements.txt`: Python-side notes and system requirements
- `benchmark_runner/`: deterministic Rust benchmark harness template

## Smoke Tests

```bash
RUST_ADAPTIVE_SORT_TASK_ID=all python examples/rust_adaptive_sort_emo_sta/evaluator.py \
  examples/rust_adaptive_sort_emo_sta/initial_program.rs
```

```bash
RUST_ADAPTIVE_SORT_TASK_ID=ras_random python examples/rust_adaptive_sort_emo_sta/evaluator.py \
  examples/rust_adaptive_sort_emo_sta/initial_program.rs
```

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml \
  --shared-iterations 4 \
  --adaptation-iterations 2 \
  --baseline-iterations 3
```
