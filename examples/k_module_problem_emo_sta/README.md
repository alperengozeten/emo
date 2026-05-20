# K-Module Problem for EMO-STA

This directory contains the `multi_task_shared_then_adapt` K-module family. It
is separate from the standalone
[examples/k_module_problem](../k_module_problem) example.

This family is designed so shared optimization is still useful, but
the shared consensus is not already equal to any one hidden task. It uses 6
modules with 6 opaque options per module, and the hidden tasks are arranged so
each task matches the shared consensus on exactly half of the modules.

The task IDs are intentionally opaque:

- `kmb_task_a`
- `kmb_task_b`
- `kmb_task_c`
- `kmb_task_d`

The target configurations remain hidden and are not serialized into public task
specs, README examples, output paths, or prompt-visible artifacts.

Use this directory through the EMO-STA manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml
```

Outputs are written under:

```text
multi_task_shared_then_adapt/results/k_module_problem_balanced/<run_name>/
```
