# Signal Processing for EMO-STA

This directory contains the `multi_task_shared_then_adapt` version of
signal processing. It is separate from the standalone
[examples/signal_processing](../signal_processing) example, which remains
usable as its own single-example benchmark.

The EMO-STA family uses four public signal-processing tasks:

- `sp_trend_sine_500_n02`
- `sp_multifreq_600_n03`
- `sp_chirp_700_n04`
- `sp_step_800_n05`

All four tasks share one evolving program representation: a generic causal 1D
filtering algorithm with the direct-input interface:

```python
def process_signal(noisy_signal, window_size=20):
    ...
    return filtered_signal

def run_signal_processing(noisy_signal, window_size=20):
    ...
    return {"filtered_signal": filtered_signal}
```

The evaluator passes the actual noisy signal into the candidate. The evolving
program does not receive the clean signal, task ID, or formula name, and it is
not supposed to regenerate the benchmark data internally.

Shared mode evaluates one candidate across all four public signal types and
optimizes the average score. The spawn step then warmstarts one task-specific
checkpoint per task from the shared checkpoint population, and adaptation runs
continue from those spawned archives.

Use this directory through the EMO-STA manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml
```

Outputs are written under:

```text
multi_task_shared_then_adapt/results/signal_processing/<run_name>/
```
