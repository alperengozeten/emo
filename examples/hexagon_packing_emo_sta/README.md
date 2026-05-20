# Hexagon Packing EMO-STA

This is the multi-task shared-then-adapt version of hexagon packing. It
is separate from the original standalone AlphaEvolve math-problem examples in
`examples/alphaevolve_math_problems/hexagon_packing/11` and
`examples/alphaevolve_math_problems/hexagon_packing/12`, which remain
unchanged fixed-`n` examples.

The EMO-STA family uses four public tasks that all share the same problem
structure: pack `n` unit regular hexagons inside a regular outer hexagon while
minimizing the outer side length.

The public EMO-STA task IDs are:

- `hex_pack_n10`
- `hex_pack_n11`
- `hex_pack_n12`
- `hex_pack_n13`

All tasks use the same generic candidate interface:

```python
def construct_hexagon_packing(n: int):
    ...
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def run_hexagon_packing(n: int):
    return construct_hexagon_packing(n)
```

Each row of `inner_hex_data` is `(center_x, center_y, angle_degrees)` for a
unit regular hexagon. `outer_hex_data` is
`(outer_center_x, outer_center_y, outer_angle_degrees)`. The evaluator checks
that all inner hexagons stay inside the outer hexagon and that pairwise overlap
is absent up to boundary-contact tolerance using polygon geometry.

The shared EMO-STA phase evaluates the same evolving program on all four tasks
and optimizes the average normalized score across them. The spawn step then
warmstarts four task-specific checkpoints from the shared checkpoint
population, and each task-specific adaptation run adapts to one exact
value of `n`.

The evaluator chooses tasks through `HEXAGON_PACKING_TASK_ID`:

- `HEXAGON_PACKING_TASK_ID=all` evaluates the shared family objective
- `HEXAGON_PACKING_TASK_ID=hex_pack_n10|hex_pack_n11|hex_pack_n12|hex_pack_n13`
  evaluates one task

Quick smoke checks:

```bash
HEXAGON_PACKING_TASK_ID=all python examples/hexagon_packing_emo_sta/evaluator.py \
  examples/hexagon_packing_emo_sta/initial_program.py
```

```bash
HEXAGON_PACKING_TASK_ID=hex_pack_n11 python examples/hexagon_packing_emo_sta/evaluator.py \
  examples/hexagon_packing_emo_sta/initial_program.py
```

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/hexagon_packing_emo_sta.yaml \
  --shared-iterations 2 \
  --adaptation-iterations 2 \
  --baseline-iterations 2
```
