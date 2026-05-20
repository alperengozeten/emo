use rust_adaptive_sort_emo_sta_benchmark::adaptive_sort;
use std::env;
use std::hint::black_box;
use std::process;
use std::time::Instant;

const DATASET_SIZES: [usize; 2] = [1000, 10000];
const SEEDED_TASK_SEEDS: [u64; 3] = [0, 1, 2];
const REVERSE_SORTED_SEEDS: [u64; 0] = [];
const DUPLICATE_UNIQUES: [(usize, i32); 2] = [(1000, 10), (10000, 100)];

#[derive(Clone, Copy)]
struct TaskSpec {
    task_id: &'static str,
    display_name: &'static str,
    pattern: &'static str,
    dataset_sizes: &'static [usize],
    seeds: &'static [u64],
    disorder_rate: Option<f64>,
    unique_values_by_size: &'static [(usize, i32)],
    warmup_repetitions: usize,
    benchmark_repetitions: usize,
}

const TASK_SPECS: [TaskSpec; 4] = [
    TaskSpec {
        task_id: "ras_random",
        display_name: "Random",
        pattern: "random",
        dataset_sizes: &DATASET_SIZES,
        seeds: &SEEDED_TASK_SEEDS,
        disorder_rate: None,
        unique_values_by_size: &[],
        warmup_repetitions: 1,
        benchmark_repetitions: 5,
    },
    TaskSpec {
        task_id: "ras_nearly_sorted",
        display_name: "NearlySorted",
        pattern: "nearly_sorted",
        dataset_sizes: &DATASET_SIZES,
        seeds: &SEEDED_TASK_SEEDS,
        disorder_rate: Some(0.05),
        unique_values_by_size: &[],
        warmup_repetitions: 1,
        benchmark_repetitions: 5,
    },
    TaskSpec {
        task_id: "ras_reverse_sorted",
        display_name: "ReverseSorted",
        pattern: "reverse_sorted",
        dataset_sizes: &DATASET_SIZES,
        seeds: &REVERSE_SORTED_SEEDS,
        disorder_rate: None,
        unique_values_by_size: &[],
        warmup_repetitions: 1,
        benchmark_repetitions: 5,
    },
    TaskSpec {
        task_id: "ras_duplicates",
        display_name: "Duplicates",
        pattern: "duplicates",
        dataset_sizes: &DATASET_SIZES,
        seeds: &SEEDED_TASK_SEEDS,
        disorder_rate: None,
        unique_values_by_size: &DUPLICATE_UNIQUES,
        warmup_repetitions: 1,
        benchmark_repetitions: 5,
    },
];

#[derive(Clone)]
struct DatasetCase {
    label: String,
    size: usize,
    seed: Option<u64>,
    values: Vec<i32>,
}

#[derive(Clone)]
struct DatasetSummary {
    label: String,
    size: usize,
    seed: Option<u64>,
    candidate_median_time: f64,
    reference_median_time: f64,
    speedup_ratio: f64,
    sorted_correctly: bool,
}

struct TaskOutput {
    task_id: String,
    display_name: String,
    pattern: String,
    dataset_count: usize,
    successful_dataset_count: usize,
    correctness_rate: f64,
    mean_speedup: f64,
    median_speedup: f64,
    speed_score: f64,
    consistency_score: f64,
    candidate_avg_time: f64,
    reference_avg_time: f64,
    score: f64,
    combined_score: f64,
    datasets: Vec<DatasetSummary>,
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_usize_below(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            return 0;
        }
        (self.next_u64() % upper_exclusive as u64) as usize
    }

    fn next_i32_below(&mut self, upper_exclusive: i32) -> i32 {
        if upper_exclusive <= 1 {
            return 0;
        }
        (self.next_u64() % upper_exclusive as u64) as i32
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn find_task(task_id: &str) -> Option<TaskSpec> {
    TASK_SPECS.iter().copied().find(|task| task.task_id == task_id)
}

fn parse_task_id() -> Result<String, String> {
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--task" {
            return args
                .next()
                .ok_or_else(|| "--task requires a task id".to_string());
        }
    }
    Err("Usage: benchmark_runner --task <task_id>".to_string())
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control.is_control() => {
                let code = control as u32;
                escaped.push_str(&format!("\\u{code:04x}"));
            }
            _ => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "0.0".to_string()
    }
}

fn dataset_summary_to_json(summary: &DatasetSummary) -> String {
    let seed = match summary.seed {
        Some(seed) => seed.to_string(),
        None => "null".to_string(),
    };
    format!(
        concat!(
            "{{",
            "\"label\":{},",
            "\"size\":{},",
            "\"seed\":{},",
            "\"candidate_median_time\":{},",
            "\"reference_median_time\":{},",
            "\"speedup_ratio\":{},",
            "\"sorted_correctly\":{}",
            "}}"
        ),
        json_escape(&summary.label),
        summary.size,
        seed,
        json_number(summary.candidate_median_time),
        json_number(summary.reference_median_time),
        json_number(summary.speedup_ratio),
        summary.sorted_correctly
    )
}

fn task_output_to_json(output: &TaskOutput) -> String {
    let datasets = output
        .datasets
        .iter()
        .map(dataset_summary_to_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{",
            "\"task_id\":{},",
            "\"display_name\":{},",
            "\"pattern\":{},",
            "\"dataset_count\":{},",
            "\"successful_dataset_count\":{},",
            "\"correctness_rate\":{},",
            "\"mean_speedup\":{},",
            "\"median_speedup\":{},",
            "\"speed_score\":{},",
            "\"consistency_score\":{},",
            "\"candidate_avg_time\":{},",
            "\"reference_avg_time\":{},",
            "\"score\":{},",
            "\"combined_score\":{},",
            "\"datasets\":[{}]",
            "}}"
        ),
        json_escape(&output.task_id),
        json_escape(&output.display_name),
        json_escape(&output.pattern),
        output.dataset_count,
        output.successful_dataset_count,
        json_number(output.correctness_rate),
        json_number(output.mean_speedup),
        json_number(output.median_speedup),
        json_number(output.speed_score),
        json_number(output.consistency_score),
        json_number(output.candidate_avg_time),
        json_number(output.reference_avg_time),
        json_number(output.score),
        json_number(output.combined_score),
        datasets
    )
}

fn generate_random_case(size: usize, seed: u64) -> DatasetCase {
    let mut rng = SimpleRng::new(seed);
    let values = (0..size)
        .map(|_| rng.next_i32_below(10000))
        .collect::<Vec<_>>();
    DatasetCase {
        label: format!("random_size{}_seed{}", size, seed),
        size,
        seed: Some(seed),
        values,
    }
}

fn generate_nearly_sorted_case(size: usize, seed: u64, disorder_rate: f64) -> DatasetCase {
    let mut values = (0..size as i32).collect::<Vec<_>>();
    let mut rng = SimpleRng::new(seed);
    let swaps = (size as f64 * disorder_rate).floor() as usize;
    for _ in 0..swaps {
        let left = rng.next_usize_below(size);
        let right = rng.next_usize_below(size);
        values.swap(left, right);
    }
    DatasetCase {
        label: format!("nearly_sorted_size{}_seed{}", size, seed),
        size,
        seed: Some(seed),
        values,
    }
}

fn generate_reverse_sorted_case(size: usize) -> DatasetCase {
    let values = (0..size as i32).rev().collect::<Vec<_>>();
    DatasetCase {
        label: format!("reverse_sorted_size{}", size),
        size,
        seed: None,
        values,
    }
}

fn duplicates_unique_values(task: &TaskSpec, size: usize) -> Result<i32, String> {
    task.unique_values_by_size
        .iter()
        .find_map(|(candidate_size, unique_values)| {
            if *candidate_size == size {
                Some(*unique_values)
            } else {
                None
            }
        })
        .ok_or_else(|| format!("No duplicate configuration for size {}", size))
}

fn generate_duplicates_case(task: &TaskSpec, size: usize, seed: u64) -> Result<DatasetCase, String> {
    let unique_values = duplicates_unique_values(task, size)?;
    let mut rng = SimpleRng::new(seed);
    let values = (0..size)
        .map(|_| rng.next_i32_below(unique_values))
        .collect::<Vec<_>>();
    Ok(DatasetCase {
        label: format!("duplicates_size{}_seed{}", size, seed),
        size,
        seed: Some(seed),
        values,
    })
}

fn build_datasets(task: &TaskSpec) -> Result<Vec<DatasetCase>, String> {
    let mut datasets = Vec::new();
    match task.pattern {
        "random" => {
            for size in task.dataset_sizes {
                for seed in task.seeds {
                    datasets.push(generate_random_case(*size, *seed));
                }
            }
        }
        "nearly_sorted" => {
            let disorder_rate = task
                .disorder_rate
                .ok_or_else(|| "nearly_sorted task requires disorder_rate".to_string())?;
            for size in task.dataset_sizes {
                for seed in task.seeds {
                    datasets.push(generate_nearly_sorted_case(*size, *seed, disorder_rate));
                }
            }
        }
        "reverse_sorted" => {
            for size in task.dataset_sizes {
                datasets.push(generate_reverse_sorted_case(*size));
            }
        }
        "duplicates" => {
            for size in task.dataset_sizes {
                for seed in task.seeds {
                    datasets.push(generate_duplicates_case(task, *size, *seed)?);
                }
            }
        }
        other => {
            return Err(format!("Unsupported task pattern {}", other));
        }
    }
    Ok(datasets)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        finite_or_zero((sorted[middle - 1] + sorted[middle]) / 2.0)
    } else {
        finite_or_zero(sorted[middle])
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        finite_or_zero(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn population_std_dev(values: &[f64], mean_value: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean_value;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    finite_or_zero(variance.sqrt())
}

fn benchmark_dataset(task: &TaskSpec, dataset: &DatasetCase) -> DatasetSummary {
    let original = dataset.values.clone();
    let mut reference_sorted = original.clone();
    reference_sorted.sort_unstable();

    let mut candidate_ready = true;
    for _ in 0..task.warmup_repetitions {
        let mut candidate_input = black_box(original.clone());
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            adaptive_sort(black_box(candidate_input.as_mut_slice()));
        }));
        black_box(&candidate_input);
        if run.is_err() || candidate_input != reference_sorted {
            candidate_ready = false;
            break;
        }
    }

    for _ in 0..task.warmup_repetitions {
        let mut reference_input = black_box(original.clone());
        black_box(reference_input.as_mut_slice()).sort_unstable();
        black_box(&reference_input);
    }

    let mut candidate_times = Vec::with_capacity(task.benchmark_repetitions);
    let mut sorted_correctly = candidate_ready;
    if candidate_ready {
        for _ in 0..task.benchmark_repetitions {
            let mut candidate_input = black_box(original.clone());
            let start = Instant::now();
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                adaptive_sort(black_box(candidate_input.as_mut_slice()));
            }));
            let elapsed = start.elapsed().as_secs_f64();
            black_box(&candidate_input);
            if run.is_err() || candidate_input != reference_sorted {
                sorted_correctly = false;
                break;
            }
            candidate_times.push(elapsed);
        }
    }

    let mut reference_times = Vec::with_capacity(task.benchmark_repetitions);
    for _ in 0..task.benchmark_repetitions {
        let mut reference_input = black_box(original.clone());
        let start = Instant::now();
        black_box(reference_input.as_mut_slice()).sort_unstable();
        reference_times.push(start.elapsed().as_secs_f64());
        black_box(&reference_input);
    }

    let candidate_median_time = median(&candidate_times);
    let reference_median_time = median(&reference_times);
    let speedup_ratio = if candidate_median_time > 0.0 {
        finite_or_zero(reference_median_time / candidate_median_time)
    } else {
        0.0
    };

    DatasetSummary {
        label: dataset.label.clone(),
        size: dataset.size,
        seed: dataset.seed,
        candidate_median_time,
        reference_median_time,
        speedup_ratio,
        sorted_correctly,
    }
}

fn evaluate_task(task: &TaskSpec) -> Result<TaskOutput, String> {
    let datasets = build_datasets(task)?;
    let dataset_summaries = datasets
        .iter()
        .map(|dataset| benchmark_dataset(task, dataset))
        .collect::<Vec<_>>();

    let dataset_count = dataset_summaries.len();
    let successful_dataset_count = dataset_summaries
        .iter()
        .filter(|summary| summary.sorted_correctly)
        .count();
    let correctness_rate = if dataset_count > 0 {
        successful_dataset_count as f64 / dataset_count as f64
    } else {
        0.0
    };

    let speedup_ratios = dataset_summaries
        .iter()
        .map(|summary| summary.speedup_ratio)
        .collect::<Vec<_>>();
    let candidate_times = dataset_summaries
        .iter()
        .map(|summary| summary.candidate_median_time)
        .collect::<Vec<_>>();
    let reference_times = dataset_summaries
        .iter()
        .map(|summary| summary.reference_median_time)
        .collect::<Vec<_>>();

    let mean_speedup = mean(&speedup_ratios);
    let median_speedup = median(&speedup_ratios);
    let speed_score = if mean_speedup > 0.0 {
        finite_or_zero(mean_speedup / (1.0 + mean_speedup))
    } else {
        0.0
    };
    let speedup_cv = if speedup_ratios.len() > 1 {
        finite_or_zero(population_std_dev(&speedup_ratios, mean_speedup) / mean_speedup.max(1e-9))
    } else {
        0.0
    };
    let consistency_score = finite_or_zero(1.0 / (1.0 + speedup_cv));
    let candidate_avg_time = mean(&candidate_times);
    let reference_avg_time = mean(&reference_times);
    let score = if correctness_rate < 1.0 {
        0.0
    } else {
        finite_or_zero(0.8 * speed_score + 0.2 * consistency_score)
    };

    Ok(TaskOutput {
        task_id: task.task_id.to_string(),
        display_name: task.display_name.to_string(),
        pattern: task.pattern.to_string(),
        dataset_count,
        successful_dataset_count,
        correctness_rate: finite_or_zero(correctness_rate),
        mean_speedup,
        median_speedup,
        speed_score,
        consistency_score,
        candidate_avg_time,
        reference_avg_time,
        score,
        combined_score: score,
        datasets: dataset_summaries,
    })
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));

    let task_id = match parse_task_id() {
        Ok(task_id) => task_id,
        Err(message) => {
            eprintln!("{}", message);
            process::exit(2);
        }
    };

    let task = match find_task(&task_id) {
        Some(task) => task,
        None => {
            eprintln!("Unknown task id {}", task_id);
            process::exit(2);
        }
    };

    let output = match evaluate_task(&task) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{}", message);
            process::exit(1);
        }
    };

    println!("{}", task_output_to_json(&output));
}
