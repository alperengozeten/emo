# EMO-STA: Multi-Task Shared-Then-Adapt

`multi_task_shared_then_adapt` is an archive-sharing workflow for related task families.
The original OpenEvolve project README is preserved as [README_openevolve.md](../../README_openevolve.md).

One shared run evolves against a shared scalar objective, then task-specific runs warmstart directly from the shared archive/checkpoint.

The EMO-STA documentation below covers the eight main-paper benchmark families:

- the K-module EMO-STA family in [examples/k_module_problem_emo_sta](../../examples/k_module_problem_emo_sta)
- function minimization in [examples/function_minimization_emo_sta](../../examples/function_minimization_emo_sta)
- Heilbronn triangle in [examples/heilbronn_triangle_emo_sta](../../examples/heilbronn_triangle_emo_sta)
- circle packing in [examples/circle_packing_emo_sta](../../examples/circle_packing_emo_sta)
- circle packing in rectangles of perimeter 4 in [examples/circle_packing_rectangle_emo_sta](../../examples/circle_packing_rectangle_emo_sta)
- signal processing in [examples/signal_processing_emo_sta](../../examples/signal_processing_emo_sta)
- SLDBench 3D scaling laws in [examples/sldbench_3d_emo_sta](../../examples/sldbench_3d_emo_sta)
- Rust adaptive sort in [examples/rust_adaptive_sort_emo_sta](../../examples/rust_adaptive_sort_emo_sta)

## When To Use EMO-STA

Use EMO-STA when the task family is tightly integratable:

- tasks share one program representation
- tasks share one evaluator family
- a shared checkpoint can be meaningfully rescored per task
- direct database reuse is desirable

Do not use EMO-STA when tasks are only loosely related or cannot share one program/evaluator interface.

## Supported Families

### K-Module Problem

`k_module_problem_balanced` is the hidden-task K-module EMO-STA family. It keeps
evaluation deterministic, cheap, and fully compatible with the same shared
evolution, checkpoint spawning, task-specific adaptation, and direct single-task
baseline workflow.

The family has a hidden-task structure:

- it uses 6 modules instead of 4
- it uses 6 opaque options per module
- the hidden tasks are arranged so the shared consensus is unique but is not equal to any one task
- each hidden task matches the shared consensus on exactly 3 of 6 modules
- task IDs remain opaque so hidden targets do not leak through paths or public metadata

The evaluator selects tasks through `K_MODULE_BALANCED_TASK_ID`.

- `K_MODULE_BALANCED_TASK_ID=all` runs the shared evaluator over all four hidden tasks and optimizes the average task score.
- `K_MODULE_BALANCED_TASK_ID=<task_id>` runs a single task-specific evaluator.

The family uses these opaque task IDs:

- `kmb_task_a`
- `kmb_task_b`
- `kmb_task_c`
- `kmb_task_d`

Use the K-module manifest:

```text
multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/k_module_problem_balanced/<run_name>/
```

### Function Minimization

Function minimization is a good fit for EMO-STA because all tasks share one Python
program representation, one evaluator family, one derivative-free search
interface, and closely related 2D continuous landscapes. The shared stage can
optimize average performance across several objective functions, then the
task-specific phases can adapt to one exact landscape at a time.

The evaluator selects tasks through `FUNCTION_MINIMIZATION_TASK_ID`.

- `FUNCTION_MINIMIZATION_TASK_ID=all` runs the shared evaluator over all four tasks and optimizes the average task score.
- `FUNCTION_MINIMIZATION_TASK_ID=<task_id>` runs a single task-specific evaluator.

Unlike hidden-task families, the function-minimization tasks are
fully public and descriptive:

- `fm_sincosxy_2d`
- `fm_ackley_2d`
- `fm_rastrigin_2d`
- `fm_rosenbrock_2d`

The exact public formulas are:

```python
f_sincosxy(x, y) = sin(x) * cos(y) + sin(x * y) + (x**2 + y**2) / 20
f_ackley(x, y) = -20 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2*pi*x) + cos(2*pi*y))) + e + 20
f_rastrigin(x, y) = 20 + x**2 + y**2 - 10 * (cos(2*pi*x) + cos(2*pi*y))
f_rosenbrock(x, y) = (1 - x)**2 + 100 * (y - x**2)**2
```

Use the manifest:

```text
multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml
```

### Heilbronn Triangle

The Heilbronn triangle family is a good fit for EMO-STA because all tasks share
one evolving Python artifact: a generic `construct_points(n)` /
`run_heilbronn(n)` program that places points inside one fixed canonical
unit-area triangle and tries to maximize the minimum area over all triples. The
shared phase can learn reusable geometric structure across nearby values of
`n`, and the spawned task-specific phases can then adapt that shared code to
one exact value of `n`.

This EMO-STA family implements the triangle variant only. All four public tasks
use the same canonical unit-area triangle:

- `A = (0.0, 0.0)`
- `B = (2.0, 0.0)`
- `C = (0.0, 1.0)`

The evaluator selects tasks through `HEILBRONN_TRIANGLE_TASK_ID`.

- `HEILBRONN_TRIANGLE_TASK_ID=all` runs the shared evaluator over all four public tasks and optimizes the average normalized task score.
- `HEILBRONN_TRIANGLE_TASK_ID=<task_id>` runs a single task-specific evaluator.

The Heilbronn-triangle EMO-STA family uses exactly these four public task IDs:

- `heil_tri_n9`
- `heil_tri_n10`
- `heil_tri_n11`
- `heil_tri_n12`

All tasks share the same objective class and differ only by `n_points` and the
public pinned normalization anchor:

- `heil_tri_n9`: `target_min_area = 0.0548469387755102`
- `heil_tri_n10`: `target_min_area = 0.04337673349889024`
- `heil_tri_n11`: `target_min_area = 0.03609267801015405`
- `heil_tri_n12`: `target_min_area = 0.03100478174352528`

These targets are pinned scoring anchors for reproducibility only. They are not
clipped at `1.0`, so a valid construction that exceeds an anchor can score
above `1.0`.

This EMO-STA family intentionally uses one generic program interface for the
shared phase, spawned task-specific adaptation phases, and direct baselines:

```python
def construct_points(n: int):
    ...
    return points, min_area

def run_heilbronn(n: int):
    return construct_points(n)
```

The evolving program should stay generic over `n`; task selection happens
through the evaluator env var rather than through per-task prompt overrides.
One generic base prompt/config is reused across shared, adaptation, and
baseline phases. The shared phase optimizes the average normalized score across
the four public tasks, and the spawned task-specific phases warmstart from the
shared checkpoint population before adapting to one exact task.

Use the manifest:

```text
multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/heilbronn_triangle/<run_name>/
```

Heilbronn triangle also supports explicit post-hoc OOD evaluation on frozen
finished programs for nearby unseen values of `n`:

- `heil_tri_n8`: `target_min_area = 0.06778914101959856`
- `heil_tri_n13`: `target_min_area = 0.02456425934867466`
- `heil_tri_n14`: `target_min_area = 0.02377577301721215`

These OOD tasks are not part of `HEILBRONN_TRIANGLE_TASK_ID=all`; `all` remains
the original four training tasks. The OOD anchors are pinned for reproducible
normalized scoring and are used only by explicit evaluator selection or the
post-hoc OOD utility. The default post-hoc OOD set remains `heil_tri_n8` and
`heil_tri_n13`; include `heil_tri_n14` with `--ood-task-ids` to evaluate it.

### Circle Packing

Circle packing is a good fit for EMO-STA because all tasks share one evolving
Python artifact: a generic `construct_packing(n)` / `run_packing(n)` program
that packs circles of varying radii into the unit square. The shared phase can
learn reusable geometric construction and local optimization logic across
several closely related `n` values, and the spawned task-specific phases can
then adapt that shared code to one exact `n`.

The evaluator selects tasks through `CIRCLE_PACKING_TASK_ID`.

- `CIRCLE_PACKING_TASK_ID=all` runs the shared evaluator over all four public tasks and optimizes the average normalized task score.
- `CIRCLE_PACKING_TASK_ID=<task_id>` runs a single task-specific evaluator.

The circle-packing EMO-STA family uses exactly these four public task IDs:

- `cp_n20`
- `cp_n22`
- `cp_n24`
- `cp_n26`

All tasks pack varying-radius circles inside the same container:

- container: unit square `[0,1] x [0,1]`
- objective: maximize the sum of radii
- constraints: all circles remain inside the square and do not overlap
- selection objective: `score == combined_score == target_ratio`

The public approximate normalization anchors are:

- `cp_n20`: `target_sum_radii = 2.301`
- `cp_n22`: `target_sum_radii = 2.420`
- `cp_n24`: `target_sum_radii = 2.530`
- `cp_n26`: `target_sum_radii = 2.635`

These targets are public scoring anchors only. They are not clipped at `1.0`,
so a valid packing that exceeds the anchor can score above `1.0`.

The family also includes evaluation-only unseen-`n` holdout tasks:

- `cp_n21`: `target_sum_radii = 2.362`
- `cp_n23`: `target_sum_radii = 2.478`
- `cp_n25`: `target_sum_radii = 2.587`

These holdouts are not part of the shared training objective, not used for
spawn selection, not used for task-specific adaptation, and not used for direct
single-task baselines. They are only used in a post-hoc evaluation layer that
measures whether the learned strategy transfers to nearby unseen values of `n`.

This EMO-STA family intentionally uses one generic program interface for the
shared phase, spawned task-specific adaptation phases, and direct baselines:

```python
def construct_packing(n: int):
    ...
    return centers, radii, sum_radii

def run_packing(n: int):
    return construct_packing(n)
```

The evolving program should stay generic over `n`; task selection happens
through the evaluator env var rather than through per-task prompt overrides.
One generic base prompt/config is reused across shared, adaptation, and
baseline phases.

By default the main EMO-STA runner performs this post-hoc holdout evaluation for
`circle_packing`. Use `--skip-holdouts` if you want to suppress that diagnostic
phase.

Use the manifest:

```text
multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/circle_packing/<run_name>/
```

### Circle Packing In Rectangles Of Perimeter 4

`circle_packing_rectangle` is a second circle-packing EMO-STA family that keeps
the same shared-then-adapt workflow structure while changing the container
geometry. It is a good fit for EMO-STA because all tasks still share one
evolving Python artifact: a generic `construct_packing(n)` / `run_packing(n)`
program that chooses a rectangle width `alpha`, packs varying-radius circles
inside that rectangle, and can reuse the same constructive and local
optimization logic across nearby `n` values.

The evaluator selects tasks through `CIRCLE_PACKING_RECTANGLE_TASK_ID`.

- `CIRCLE_PACKING_RECTANGLE_TASK_ID=all` runs the shared evaluator over all four public tasks and optimizes the average normalized task score.
- `CIRCLE_PACKING_RECTANGLE_TASK_ID=<task_id>` runs a single task-specific evaluator.

The rectangle circle-packing EMO-STA family uses exactly these four public task
IDs:

- `cp_rect_n20`
- `cp_rect_n21`
- `cp_rect_n22`
- `cp_rect_n23`

All tasks share the same exact formulation:

- rectangle width is the decision variable `alpha`
- rectangle height is `2 - alpha`
- perimeter is always `4`
- `0 < alpha <= 1`
- circles must lie fully inside `[0, alpha] x [0, 2 - alpha]`
- circles must be pairwise non-overlapping
- objective: maximize the sum of radii
- selection objective: `score == combined_score == target_ratio`

The pinned public normalization anchors are:

- `cp_rect_n20`: `target_sum_radii = 2.305`
- `cp_rect_n21`: `target_sum_radii = 2.365`
- `cp_rect_n22`: `target_sum_radii = 2.425`
- `cp_rect_n23`: `target_sum_radii = 2.484`

These targets are fixed scoring anchors for reproducibility. They are not
clipped at `1.0`, so a valid packing that exceeds the anchor can score above
`1.0`. Newer work has reported an `n=21` value around `2.3658`, but this EMO-STA
family intentionally keeps the pinned evaluator anchor at `2.365`.

This EMO-STA family intentionally uses one generic program interface for the
shared phase, spawned task-specific adaptation phases, and direct baselines:

```python
def construct_packing(n: int):
    ...
    return centers, radii, alpha, sum_radii

def run_packing(n: int):
    return construct_packing(n)
```

The evolving program should stay generic over `n`; task selection happens
through the evaluator env var rather than through per-task prompt overrides.
One generic base prompt/config is reused across shared, adaptation, and
baseline phases.

Use the manifest:

```text
multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/circle_packing_rectangle/<run_name>/
```

Rectangle circle packing also supports explicit post-hoc OOD evaluation on
frozen finished programs for nearby unseen values of `n`:

- `cp_rect_n19`: `target_sum_radii = 2.241`
- `cp_rect_n24`: `target_sum_radii = 2.535`
- `cp_rect_n25`: `target_sum_radii = 2.592`

These OOD tasks are not part of `CIRCLE_PACKING_RECTANGLE_TASK_ID=all`; `all`
remains the original four training tasks. The OOD anchors are pinned for
reproducible normalized scoring and are used only by explicit evaluator
selection or the post-hoc OOD utility. The default post-hoc OOD set remains
`cp_rect_n19` and `cp_rect_n24`; include `cp_rect_n25` with `--ood-task-ids` to
evaluate it.

## Post-Hoc OOD Evaluation

Finished static EMO-STA runs for `heilbronn_triangle` and
`circle_packing_rectangle` can be evaluated after the fact on unseen nearby-`n`
tasks without rerunning evolution. This utility evaluates the frozen best
programs selected by the completed in-distribution runs:

- the shared-phase best program
- each final adapted task-specific best program
- each direct single-task baseline best program

This is **post-hoc unseen nearby-n evaluation**. The prompts, configs, and
selection decisions remain those of the original in-distribution training task
set; OOD scores are diagnostic and are not used for program selection.
Current defaults preserve the original OOD sets. Additional explicit OOD tasks
such as `heil_tri_n14` and `cp_rect_n25` are evaluated only when listed with
`--ood-task-ids`.

Example commands:

```bash
python multi_task_shared_then_adapt/scripts/run_posthoc_ood_evaluation.py \
  --manifest multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml \
  --results-dir multi_task_shared_then_adapt/results/heilbronn_triangle/<run_name>
```

```bash
python multi_task_shared_then_adapt/scripts/run_posthoc_ood_evaluation.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml \
  --results-dir multi_task_shared_then_adapt/results/circle_packing_rectangle/<run_name>
```

The default output directory is `<results-dir>/posthoc_ood/`, containing:

- `ood_summary.json`
- `ood_summary.csv`

### Signal Processing

Signal processing is a good fit for EMO-STA because all tasks share one Python
program representation, one evaluator family, and one generic causal 1D
filtering interface. The shared stage can optimize average filtering quality
across several related signal families, then the spawned task-specific runs can
adapt to one exact signal family at a time.

The evaluator selects tasks through `SIGNAL_PROCESSING_TASK_ID`.

- `SIGNAL_PROCESSING_TASK_ID=all` runs the shared evaluator over all four public tasks and optimizes the average task score.
- `SIGNAL_PROCESSING_TASK_ID=<task_id>` runs a single task-specific evaluator.

The signal-processing EMO-STA family uses exactly these four public task IDs:

- `sp_trend_sine_500_n02`
- `sp_multifreq_600_n03`
- `sp_chirp_700_n04`
- `sp_step_800_n05`

All tasks use these common conventions:

- `t = np.linspace(0.0, 10.0, length)`
- `window_size = 20`
- Gaussian additive noise with mean `0.0` and standard deviation `noise_level`
- `np.random.default_rng(seed)` for deterministic noise generation
- `noisy_signal = clean_signal + rng.normal(0.0, noise_level, length)`

The exact public clean-signal formulas are:

```python
# sp_trend_sine_500_n02
clean = 2.0 * np.sin(2.0 * np.pi * 0.5 * t) + 0.1 * t

# sp_multifreq_600_n03
clean = (
    np.sin(2.0 * np.pi * 0.5 * t)
    + 0.5 * np.sin(2.0 * np.pi * 2.0 * t)
    + 0.2 * np.sin(2.0 * np.pi * 5.0 * t)
)

# sp_chirp_700_n04
clean = np.sin(2.0 * np.pi * (0.5 + 0.2 * t) * t)

# sp_step_800_n05
clean = np.concatenate(
    [
        np.ones(length // 3),
        2.0 * np.ones(length // 3),
        0.5 * np.ones(length - 2 * (length // 3)),
    ]
)
```

This EMO-STA family intentionally uses a direct-input filtering interface:

```python
def process_signal(noisy_signal, window_size=20):
    ...
    return filtered_signal

def run_signal_processing(noisy_signal, window_size=20):
    ...
    return {"filtered_signal": filtered_signal}
```

The evaluator passes the actual `noisy_signal` array into the candidate. The
evolving program does not receive the clean signal, task ID, or formula name,
and it should not regenerate benchmark data internally.

Use the manifest:

```text
multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/signal_processing/<run_name>/
```

### SLDBench 3D Scaling Laws

`sldbench_3d` is a narrow EMO-STA family built specifically around the cleanest
first SLDBench subset for shared-then-adapt: the two public 3D scalar
tasks

- `vocab_scaling_law`
- `data_constrained_scaling_law`

This narrow subset fits EMO-STA well because both tasks share the same evolving
artifact:

- `scaling_law_func(data_points, params)`
- `fit_scaling_law(data_points, loss_values)`

The shared phase evolves reusable law structure and fitting strategy across both
tasks, while the task-specific phases adapt that shared program to one concrete
task. The shared phase does not transfer fitted coefficients across tasks or
groups. Coefficients are always fitted locally per task and per group during
evaluation. This EMO-STA family lives in its own
`examples/sldbench_3d_emo_sta/` directory and keeps the original standalone
`examples/sldbench/` workflow unchanged.

The evaluator selects tasks through `SLDBENCH_3D_TASK_ID`.

- `SLDBENCH_3D_TASK_ID=all` runs the shared evaluator over both tasks and optimizes the average task score.
- `SLDBENCH_3D_TASK_ID=<task_id>` runs a single task-specific evaluator.

The EMO-STA SLDBench family canonicalizes both raw feature schemas into one
shared 3-axis interface:

```text
[model_size_like, diversity_like, total_data_like]
```

The canonical mapping is:

- `vocab_scaling_law`: `[non_vocab_parameters, vocab_size, num_characters]`
- `data_constrained_scaling_law`: `[params, unique_tokens, tokens]`

This keeps the candidate task-agnostic while still preserving the original raw
positive scales. The evaluator does not pre-log-transform the inputs.

This first PR intentionally does not include the mixed-arity or multi-output
SLDBench tasks such as `sft_scaling_law`, `parallel_scaling_law`,
`moe_scaling_law`, `domain_mixture_scaling_law`, `lr_bsz_scaling_law`, or
`easy_question_scaling_law`.

Use the SLDBench 3D manifest:

```text
multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/sldbench_3d/<run_name>/
```

### Rust Adaptive Sort

Rust adaptive sort is a good fit for EMO-STA because all tasks share one
evolving Rust sorting routine over integer arrays. The shared phase can learn
generic mechanisms such as insertion-sort thresholds, pivot strategy, duplicate
handling, and nearly-sorted fast paths, and the task-specific phases can then
retune that behavior for one exact input regime.

The evaluator selects tasks through `RUST_ADAPTIVE_SORT_TASK_ID`.

- `RUST_ADAPTIVE_SORT_TASK_ID=all` runs the shared evaluator over all four public tasks and optimizes the average task score.
- `RUST_ADAPTIVE_SORT_TASK_ID=<task_id>` runs a single task-specific evaluator.

The EMO-STA Rust family uses exactly these four public task IDs:

- `ras_random`
- `ras_nearly_sorted`
- `ras_reverse_sorted`
- `ras_duplicates`

The exact public regimes are:

- `ras_random`: random arrays of sizes `1000` and `10000`, seeds `0,1,2`, values sampled uniformly from `[0, 10000)`.
- `ras_nearly_sorted`: ascending arrays of sizes `1000` and `10000`, seeds `0,1,2`, then `floor(size * 0.05)` seeded random swaps.
- `ras_reverse_sorted`: deterministic descending arrays of sizes `1000` and `10000`.
- `ras_duplicates`: duplicate-heavy arrays with `(size=1000, unique_values=10)` and `(size=10000, unique_values=100)`, seeds `0,1,2`.

This EMO-STA version intentionally uses deterministic regime-specific public
tasks instead of the original standalone mixed benchmark. The benchmark compiles
the Rust project once per candidate, then runs the compiled binary once per
selected task, and stores per-task results in `artifacts["task_results"]` so
spawn can project shared checkpoints into task-local checkpoints without forced
reevaluation when stored task artifacts are present.

`partially_sorted` is intentionally excluded from the initial shared family. It
can be used later as a post-hoc holdout/generalization check without changing
the core shared/adaptation/baseline workflow.

Use the manifest:

```text
multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml
```

Outputs go under:

```text
multi_task_shared_then_adapt/results/rust_adaptive_sort/<run_name>/
```

## How The Spawn Step Works

After the shared run finishes:

1. the orchestrator loads the shared checkpoint database
2. it reads each program's stored `artifacts["task_results"]`
3. it extracts the target task's metrics without reevaluating when the artifact is present
4. it rebuilds a fresh task-specific `ProgramDatabase`
5. it writes one normal resumable checkpoint per task under `spawned_checkpoints/<task_id>/`

Each spawned checkpoint:

- preserves code and lineage metadata where practical
- rewrites metrics to the target task
- rewrites stored artifacts to a task-local view for the target task
- recalculates archive / MAP-Elites feature maps / best program under the target task
- sets checkpoint `last_iteration` to `0` so `--iterations N` means `N` adaptation iterations

If the required per-task artifact is missing or malformed, the spawn utility falls back to reevaluating that one program on the target task.

## Output Layout

Runs are written under the manifest output root. For the main-paper manifests:

```text
multi_task_shared_then_adapt/results/k_module_problem_balanced/<run_name>/
multi_task_shared_then_adapt/results/function_minimization/<run_name>/
multi_task_shared_then_adapt/results/heilbronn_triangle/<run_name>/
multi_task_shared_then_adapt/results/circle_packing/<run_name>/
multi_task_shared_then_adapt/results/circle_packing_rectangle/<run_name>/
multi_task_shared_then_adapt/results/sldbench_3d/<run_name>/
multi_task_shared_then_adapt/results/rust_adaptive_sort/<run_name>/
multi_task_shared_then_adapt/results/signal_processing/<run_name>/
  shared_run/
  spawned_checkpoints/<task_id>/
  spawned_checkpoints_best_shared/<task_id>/   # optional one-program shared-best seeds
  spawned_checkpoints_best_local/<task_id>/     # optional one-program local-best seeds
  adaptation/<task_id>/
  adaptation_best_shared/<task_id>/   # optional best-shared branch
  adaptation_best_local/<task_id>/     # optional best-local branch
  baselines/<task_id>/
  holdout_evaluation/   # circle_packing only, unless --skip-holdouts
  comparison_summary.json
  comparison_summary.csv
```

## Adaptation Warmstart Branches

The static EMO-STA runner supports three adaptation-initialization branches. The
standard projected-archive warmstart branch is enabled by default for backward
compatibility, while the best-shared and best-local branches are opt-in.

These branches stay distinct from the direct single-task baseline:

- projected archive adaptation:
  the normal EMO-STA adaptation path that starts from the full projected
  task-local checkpoint under `spawned_checkpoints/<task_id>/`
- best shared adaptation:
  a one-program task-local adaptation branch that starts from the single best
  shared-average program from the shared checkpoint, written under
  `spawned_checkpoints_best_shared/<task_id>/`
- best local adaptation:
  a one-program task-local adaptation branch that starts from the locally best
  retained program for that task from the shared checkpoint, written
  under `spawned_checkpoints_best_local/<task_id>/`
- direct baseline:
  a normal single-task run with the baseline budget and no shared phase

Run all three paper adaptation variants with:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest <family_manifest> \
  --shared-iterations 6 \
  --adaptation-iterations 4 \
  --baseline-iterations 6 \
  --run-warmstart-adaptation \
  --run-best-shared-adaptation \
  --run-best-local-adaptation
```

You can also set explicit iteration budgets for the optional one-program
adaptation branches:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest <family_manifest> \
  --shared-iterations 6 \
  --adaptation-iterations 4 \
  --baseline-iterations 6 \
  --run-warmstart-adaptation \
  --run-best-shared-adaptation \
  --run-best-local-adaptation \
  --best-shared-adaptation-iterations 4 \
  --best-local-adaptation-iterations 4
```

By default, each optional one-program adaptation branch uses the same iteration
budget as the warmstart adaptation stage unless you override it explicitly.
Use `--skip-warmstart-adaptation` to disable only the standard archive-warmstart
branch while still allowing requested best-shared or best-local branches to run.

When enabled, `comparison_summary.json` and `comparison_summary.csv` include, for
each task:

- the best projected shared checkpoint score before adaptation
- the final warmstarted adaptation score
- the final best-shared adaptation score
- the final best-local adaptation score
- the final direct baseline score when baselines were run

## Run EMO-STA

The CLI entrypoints default to the unit-square circle-packing manifest for
convenience, but the workflow scripts are family-generic. Use the manifest/results pair that
matches the family you want to run:

- circle packing:
  `--manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/circle_packing`
- circle packing in rectangles:
  `--manifest multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/circle_packing_rectangle`
- K-module:
  `--manifest multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/k_module_problem_balanced`
- function minimization:
  `--manifest multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/function_minimization`
- Heilbronn triangle:
  `--manifest multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/heilbronn_triangle`
- signal processing:
  `--manifest multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/signal_processing`
- SLDBench 3D scaling laws:
  `--manifest multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/sldbench_3d`
- Rust adaptive sort:
  `--manifest multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/rust_adaptive_sort`

From the repository root:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml \
  --shared-iterations 20 \
  --adaptation-iterations 20 \
  --baseline-iterations 25
```

The K-module family uses the same workflow entrypoint with its own manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml \
  --shared-iterations 20 \
  --adaptation-iterations 20 \
  --baseline-iterations 25
```

Function minimization uses the same workflow entrypoint with its own manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml \
  --shared-iterations 20 \
  --adaptation-iterations 20 \
  --baseline-iterations 25
```

Heilbronn triangle uses the same workflow entrypoint with its own manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml \
  --shared-iterations 15 \
  --adaptation-iterations 12 \
  --baseline-iterations 15
```

Signal processing uses the same workflow entrypoint with its own manifest:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt.py \
  --manifest multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml \
  --shared-iterations 20 \
  --adaptation-iterations 20 \
  --baseline-iterations 25
```

This single command runs:

- Phase A: shared evolution
- Phase B: task checkpoint spawning
- Phase C: task-specific adaptation
- Phase D: direct single-task baselines
- Phase E: post-hoc holdout evaluation for `circle_packing`
- Phase F: comparison summary writing

## Automated Launching

The repeated-trials launcher works for all EMO-STA families:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_trials.py \
  --manifest <family_manifest> \
  --trials 1
```

If the setting directory already contains `run_01`, later launches continue at
`run_02`, `run_03`, and so on automatically. Use `--start-trial-number N` to
override the next displayed trial number explicitly.

The main-paper family manifests are:

- `multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml`
- `multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml`

All main-paper families use the same launcher entrypoint. They rely on the standard
launcher defaults unless you add a `launcher:` section to a manifest or pass
overrides such as `--module`, `--setup-command`, or `--litellm-*` flags.

By default this launcher:

- detaches into a launcher `.nohup.log`
- reuses an existing local LiteLLM server if one is already reachable at the configured `api_base`
- otherwise starts LiteLLM automatically
- writes one per-trial log file under `trial_logs/`
- writes `trial_summary.json`
- refreshes the EMO-STA markdown / JSON report after the trials finish

If you want to stay attached to the terminal, add `--foreground`.

Example with 5 repeated runs:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_trials.py \
  --manifest multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml \
  --trials 5 \
  --foreground
```

Example with 5 repeated runs and all three adaptation branches:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_trials.py \
  --manifest multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml \
  --trials 5 \
  --shared-iterations 60 \
  --adaptation-iterations 15 \
  --baseline-iterations 30 \
  --primary-model claude-sonnet-4-6 \
  --run-warmstart-adaptation \
  --run-best-shared-adaptation \
  --run-best-local-adaptation
```

Example with an explicit launcher log and per-trial logs while detached:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_trials.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml \
  --trials 5 \
  --log-file multi_task_shared_then_adapt/results/circle_packing/emo_sta_trials.nohup.log \
  --trial-log-dir multi_task_shared_then_adapt/results/circle_packing/trial_logs
```

You can still override the automated setup when needed:

- `--litellm skip` if you already started the proxy yourself
- `--module ...` or `--setup-command ...` to replace / extend manifest defaults
- `--parallel-trials N` to run several EMO-STA launches at once
- `--report-latest-per-setting N` to control how many latest repeats are summarized per setting after the launcher finishes

## W&B Logging

EMO-STA runs are configured to log to a separate default W&B project:

- project: `openevolve-emo-sta`

Within one EMO-STA invocation:

- the shared phase, all adaptation tasks, and all baseline tasks resume the same W&B run by default
- metrics are namespaced inside that run as `shared/*`, `adaptation/<task_id>/*`, and `baseline/<task_id>/*`
- the shared phase logs the aggregate average score under `shared/combined_score` and `shared/score`

This keeps EMO-STA logs separate from older `multi_task_evolve` runs, which typically used the `openevolve` project.

If you prefer the older per-phase/per-task W&B layout, set `wandb.single_run: false` in the manifest.

You can still override runtime behavior with normal W&B env vars, for example:

```bash
WANDB_MODE=disabled
WANDB_PROJECT=some_other_project
WANDB_ENTITY=your_entity
```

By default the launcher now uses stable run directories like `run_01_seed_42`, and the
W&B display name includes the manifest stem plus a short setting slug such as
`s20-a20-b25`. It does not append a timestamp.

## Run Baselines Only

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_baselines.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml \
  --iterations 25
```

Or for the K-module family:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_baselines.py \
  --manifest multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml \
  --iterations 25
```

Or for function minimization:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_baselines.py \
  --manifest multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml \
  --iterations 25
```

Or for circle packing:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_baselines.py \
  --manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml \
  --iterations 15
```

Or for signal processing:

```bash
python multi_task_shared_then_adapt/scripts/run_multi_task_shared_then_adapt_baselines.py \
  --manifest multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml \
  --iterations 25
```

For a four-task comparison with `20` shared and `20` adaptation iterations, the
baseline count is set to `25` per task.
That matches the EMO-STA generation budget under the simple iteration-count accounting:

- shared phase: `20`
- adaptation phase: `4 * 20 = 80`
- total EMO-STA iterations across the family: `100`
- direct baselines across 4 tasks: `100 / 4 = 25` each

## Spawn Task Checkpoints Manually

Manual spawning is not required for the main EMO-STA workflow, but the helper exists for inspection/debugging:

```bash
python multi_task_shared_then_adapt/scripts/spawn_multi_task_shared_then_adapt_task_checkpoints.py \
  --manifest <family_manifest> \
  --shared-checkpoint <family_results>/<run_name>/shared_run/checkpoints/checkpoint_20 \
  --output-root <family_results>/<run_name>/spawned_checkpoints
```

For example:

- circle packing uses `multi_task_shared_then_adapt/results/circle_packing`
- the K-module family uses `multi_task_shared_then_adapt/results/k_module_problem_balanced`
- function minimization uses `multi_task_shared_then_adapt/results/function_minimization`
- signal processing uses `multi_task_shared_then_adapt/results/signal_processing`

## Summarize Results

For EMO-STA, the most useful report is phase-aware rather than a flat W&B metric dump.
The local summarizer below reads the run directories directly, so it can report:

- fully completed workflow comparisons
- partial runs that only reached shared or spawn phases
- repeat aggregates later, grouped by setting fingerprint instead of fragile directory names

Generate a combined markdown summary across main-paper families with local results:

```bash
python multi_task_shared_then_adapt/scripts/report_emo_sta_results.py \
  --latest-per-setting 5 \
  --markdown-out multi_task_shared_then_adapt/docs/emo_sta_results_summary.md \
  --json-out multi_task_shared_then_adapt/docs/emo_sta_results_summary.json
```

To summarize just one family, pass its manifest/results pair explicitly:

```bash
python multi_task_shared_then_adapt/scripts/report_emo_sta_results.py \
  --manifest <family_manifest> \
  --results-dir <family_results_dir> \
  --latest-per-setting 5
```

For example:

- circle packing:
  `--manifest multi_task_shared_then_adapt/configs/circle_packing_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/circle_packing`
- circle packing in rectangles:
  `--manifest multi_task_shared_then_adapt/configs/circle_packing_rectangle_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/circle_packing_rectangle`
- K-module:
  `--manifest multi_task_shared_then_adapt/configs/k_module_problem_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/k_module_problem_balanced`
- function minimization:
  `--manifest multi_task_shared_then_adapt/configs/function_minimization_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/function_minimization`
- Heilbronn triangle:
  `--manifest multi_task_shared_then_adapt/configs/heilbronn_triangle_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/heilbronn_triangle`
- signal processing:
  `--manifest multi_task_shared_then_adapt/configs/signal_processing_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/signal_processing`
- SLDBench 3D scaling laws:
  `--manifest multi_task_shared_then_adapt/configs/sldbench_3d_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/sldbench_3d`
- Rust adaptive sort:
  `--manifest multi_task_shared_then_adapt/configs/rust_adaptive_sort_emo_sta.yaml`
  and `--results-dir multi_task_shared_then_adapt/results/rust_adaptive_sort`

When you want all 5 repeated runs for the same setting in the aggregate, use:

```bash
python multi_task_shared_then_adapt/scripts/report_emo_sta_results.py \
  --manifest <family_manifest> \
  --results-dir <family_results_dir> \
  --latest-per-setting 5
```

The report highlights:

- shared archive quality
- spawned warm-start quality
- adaptation lift over spawn
- EMO-STA advantage or loss versus direct baselines
- task-level win / tie / loss counts across repeats
