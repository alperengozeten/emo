# EVOLVE-BLOCK-START
"""
Balanced harder K-module EMO-STA initial program.

This EMO-STA variant evaluates one shared program representation across a hidden
family of four related tasks. The option labels are opaque identifiers. Do not
infer semantic meaning from them.
"""


def configure_pipeline():
    """
    Configure a six-module pipeline with opaque option labels.

    Valid options:
    - loader: ['loader_0', 'loader_1', 'loader_2', 'loader_3', 'loader_4', 'loader_5']
    - preprocess: ['prep_0', 'prep_1', 'prep_2', 'prep_3', 'prep_4', 'prep_5']
    - sampler: ['sample_0', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5']
    - algorithm: ['algo_0', 'algo_1', 'algo_2', 'algo_3', 'algo_4', 'algo_5']
    - scheduler: ['sched_0', 'sched_1', 'sched_2', 'sched_3', 'sched_4', 'sched_5']
    - formatter: ['fmt_0', 'fmt_1', 'fmt_2', 'fmt_3', 'fmt_4', 'fmt_5']

    Returns:
        dict: Pipeline configuration with exactly the required six keys.
    """
    # Deliberately non-optimal starting point for the hidden task family.
    return {
        "loader": "loader_5",
        "preprocess": "prep_5",
        "sampler": "sample_1",
        "algorithm": "algo_1",
        "scheduler": "sched_1",
        "formatter": "fmt_5",
    }


# EVOLVE-BLOCK-END


def run_pipeline():
    """Entry point preferred by the evaluator."""
    return configure_pipeline()


if __name__ == "__main__":
    print(run_pipeline())
