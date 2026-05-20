# EVOLVE-BLOCK-START
"""Generic causal 1D signal-processing baseline for EMO-STA."""

import numpy as np


def process_signal(noisy_signal, window_size=20):
    """
    Filter a noisy 1D signal with a causal weighted moving average.

    Args:
        noisy_signal: 1D array-like sequence of noisy samples.
        window_size: Positive causal window length.

    Returns:
        A 1D NumPy array of length len(noisy_signal) - window_size + 1.
    """
    signal_array = np.asarray(noisy_signal, dtype=float)
    if signal_array.ndim != 1:
        raise ValueError("noisy_signal must be 1D")

    window = int(window_size)
    if window <= 0:
        raise ValueError("window_size must be positive")
    if signal_array.size < window:
        raise ValueError("len(noisy_signal) must be >= window_size")

    weights = np.linspace(1.0, 2.0, window, dtype=float)
    weights /= np.sum(weights)

    output_length = signal_array.size - window + 1
    filtered = np.empty(output_length, dtype=float)
    for index in range(output_length):
        segment = signal_array[index : index + window]
        filtered[index] = float(np.dot(segment, weights))
    return filtered


def run_signal_processing(noisy_signal, window_size=20):
    """Entry point preferred by the evaluator."""
    filtered_signal = process_signal(noisy_signal, window_size=window_size)
    return {"filtered_signal": filtered_signal}


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo_signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 64)) + 0.2 * rng.normal(size=64)
    result = run_signal_processing(demo_signal, window_size=20)
    print(result["filtered_signal"][:5])
