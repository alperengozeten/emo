#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Runtime configuration
TASK="${1:-${TASK:-function_minimization}}"
TASK_ROOT="${TASK_ROOT:-}"
INITIAL_PROGRAM="${INITIAL_PROGRAM:-}"
EVALUATOR="${EVALUATOR:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
CONFIG_VARIANT="${CONFIG_VARIANT:-}"
ITERATIONS="${ITERATIONS:-100}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-300}"

# Local vLLM settings
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"
PRIMARY_MODEL_WEIGHT="${PRIMARY_MODEL_WEIGHT:-1.0}"
SECONDARY_MODEL="${SECONDARY_MODEL:-}"
SECONDARY_MODEL_WEIGHT="${SECONDARY_MODEL_WEIGHT:-0.0}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
VLLM_MODELS_ENDPOINT="${VLLM_BASE_URL}/v1/models"
VLLM_LOG="/tmp/openevolve_vllm.log"

# Task presets
DEFAULT_TASK_ROOT=""
DEFAULT_INITIAL_PROGRAM=""
DEFAULT_EVALUATOR=""
DEFAULT_CONFIG_PATH=""

case "${TASK}" in
  function_minimization)
    DEFAULT_TASK_ROOT="examples/function_minimization"
    DEFAULT_INITIAL_PROGRAM="${DEFAULT_TASK_ROOT}/initial_program.py"
    DEFAULT_EVALUATOR="${DEFAULT_TASK_ROOT}/evaluator.py"
    DEFAULT_CONFIG_PATH="vllm_quickstart/config_vllm.yaml"
    ;;
  circle_packing)
    DEFAULT_TASK_ROOT="examples/circle_packing"
    DEFAULT_INITIAL_PROGRAM="${DEFAULT_TASK_ROOT}/initial_program.py"
    DEFAULT_EVALUATOR="${DEFAULT_TASK_ROOT}/evaluator.py"
    DEFAULT_CONFIG_PATH="${DEFAULT_TASK_ROOT}/config_phase_1.yaml"
    ;;
  circle_packing_with_artifacts)
    DEFAULT_TASK_ROOT="examples/circle_packing_with_artifacts"
    DEFAULT_INITIAL_PROGRAM="${DEFAULT_TASK_ROOT}/initial_program.py"
    DEFAULT_EVALUATOR="${DEFAULT_TASK_ROOT}/evaluator.py"
    DEFAULT_CONFIG_PATH="${DEFAULT_TASK_ROOT}/config_phase_1.yaml"
    ;;
esac

detect_initial_program() {
  local task_root="$1"
  local candidate

  for candidate in \
    "${task_root}/initial_program.py" \
    "${task_root}/initial_program.rs" \
    "${task_root}/initial_program.r" \
    "${task_root}/initial_content_stub.txt" \
    "${task_root}/init_program.py"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

detect_config_path() {
  local task_root="$1"
  local candidate

  if [[ -n "${CONFIG_VARIANT}" ]]; then
    if [[ -f "${CONFIG_VARIANT}" ]]; then
      printf '%s\n' "${CONFIG_VARIANT}"
      return 0
    fi
    if [[ -f "${task_root}/${CONFIG_VARIANT}" ]]; then
      printf '%s\n' "${task_root}/${CONFIG_VARIANT}"
      return 0
    fi
  fi

  for candidate in \
    "${task_root}/config.yaml" \
    "${task_root}/config.yml" \
    "${task_root}/base_config.yaml" \
    "${task_root}/config_phase_1.yaml" \
    "${task_root}/config_phase_1_anthropic.yaml"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

resolve_task_paths() {
  local task_root=""
  local detected_path=""

  if [[ -n "${INITIAL_PROGRAM}" && -n "${EVALUATOR}" && -n "${CONFIG_PATH}" ]]; then
    return 0
  fi

  : "${INITIAL_PROGRAM:=${DEFAULT_INITIAL_PROGRAM}}"
  : "${EVALUATOR:=${DEFAULT_EVALUATOR}}"
  : "${CONFIG_PATH:=${DEFAULT_CONFIG_PATH}}"

  if [[ -n "${TASK_ROOT}" ]]; then
    task_root="${TASK_ROOT}"
  elif [[ -n "${DEFAULT_TASK_ROOT}" ]]; then
    task_root="${DEFAULT_TASK_ROOT}"
  elif [[ -d "${TASK}" ]]; then
    task_root="${TASK}"
  else
    task_root="examples/${TASK}"
  fi

  if [[ ! -d "${task_root}" ]]; then
    echo "Error: task root '${task_root}' does not exist."
    echo "Set TASK to a known preset, pass a task directory as the first argument, or provide TASK_ROOT."
    exit 1
  fi

  if [[ -z "${INITIAL_PROGRAM}" ]]; then
    detected_path="$(detect_initial_program "${task_root}")" || {
      echo "Error: could not detect an initial program under '${task_root}'."
      echo "Set INITIAL_PROGRAM explicitly."
      exit 1
    }
    INITIAL_PROGRAM="${detected_path}"
  fi

  if [[ -z "${EVALUATOR}" ]]; then
    EVALUATOR="${task_root}/evaluator.py"
  fi

  if [[ -z "${CONFIG_PATH}" ]]; then
    detected_path="$(detect_config_path "${task_root}")" || {
      echo "Error: could not detect a config file under '${task_root}'."
      echo "Set CONFIG_PATH explicitly or provide CONFIG_VARIANT."
      exit 1
    }
    CONFIG_PATH="${detected_path}"
  fi
}

build_runtime_config() {
  local source_config="$1"
  local runtime_config="$2"

  python - "${source_config}" "${runtime_config}" "${VLLM_MODEL}" "${VLLM_BASE_URL}/v1" \
    "${OPENAI_API_KEY}" "${PRIMARY_MODEL_WEIGHT}" "${SECONDARY_MODEL}" \
    "${SECONDARY_MODEL_WEIGHT}" "${LLM_MAX_TOKENS}" <<'PY'
import sys

import yaml

(
    source_config,
    runtime_config,
    primary_model,
    api_base,
    api_key,
    primary_weight,
    secondary_model,
    secondary_weight,
    max_tokens,
) = sys.argv[1:]

with open(source_config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

llm = config.setdefault("llm", {})
llm.pop("models", None)
llm.pop("evaluator_models", None)
llm["primary_model"] = primary_model
llm["primary_model_weight"] = float(primary_weight)
llm["secondary_model"] = secondary_model or None
llm["secondary_model_weight"] = float(secondary_weight)
llm["api_base"] = api_base
llm["api_key"] = api_key
llm["max_tokens"] = int(max_tokens)

with open(runtime_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY
}

if ! command -v vllm >/dev/null 2>&1; then
  echo "Error: 'vllm' command not found. Install vLLM first."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: 'curl' command not found."
  exit 1
fi

cd "${REPO_ROOT}"
resolve_task_paths

for path_name in INITIAL_PROGRAM EVALUATOR CONFIG_PATH; do
  path_value="${!path_name}"
  if [[ ! -f "${path_value}" ]]; then
    echo "Error: ${path_name} '${path_value}' not found."
    exit 1
  fi
done

if [[ -z "${RUNTIME_CONFIG_PATH}" ]]; then
  RUNTIME_CONFIG_PATH="$(mktemp /tmp/openevolve_vllm_config.XXXXXX.yaml)"
fi

build_runtime_config "${CONFIG_PATH}" "${RUNTIME_CONFIG_PATH}"

echo "Selected task: ${TASK}"
echo "Initial program: ${INITIAL_PROGRAM}"
echo "Evaluator: ${EVALUATOR}"
echo "Config: ${CONFIG_PATH}"
echo "Runtime config: ${RUNTIME_CONFIG_PATH}"
echo "Iterations: ${ITERATIONS}"
echo "Local model: ${VLLM_MODEL}"
echo "Secondary model: ${SECONDARY_MODEL:-<disabled>}"
echo "LLM max tokens: ${LLM_MAX_TOKENS}"

echo "Starting vLLM on ${VLLM_BASE_URL} with model ${VLLM_MODEL}"
echo "Using GPUs ${CUDA_VISIBLE_DEVICES}, tensor_parallel_size=${TENSOR_PARALLEL_SIZE}, max_model_len=${MAX_MODEL_LEN}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" vllm serve "${VLLM_MODEL}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --api-key "${OPENAI_API_KEY}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "Stopping vLLM process ${VLLM_PID}"
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RUNTIME_CONFIG_PATH}" && -f "${RUNTIME_CONFIG_PATH}" ]]; then
    rm -f "${RUNTIME_CONFIG_PATH}"
  fi
}
trap cleanup EXIT

echo "Waiting for vLLM readiness..."
READY=0
for ((i=1; i<=STARTUP_TIMEOUT_SECONDS; i++)); do
  if curl -sS -H "Authorization: Bearer ${OPENAI_API_KEY}" "${VLLM_MODELS_ENDPOINT}" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "Error: vLLM exited before becoming ready."
    echo "Recent vLLM logs:"
    tail -n 50 "${VLLM_LOG}" || true
    exit 1
  fi
  sleep 1
done

if [[ "${READY}" -ne 1 ]]; then
  echo "Error: vLLM did not become ready within ${STARTUP_TIMEOUT_SECONDS}s."
  echo "Recent vLLM logs:"
  tail -n 50 "${VLLM_LOG}" || true
  exit 1
fi

echo "vLLM is ready. Running OpenEvolve..."
python openevolve-run.py "${INITIAL_PROGRAM}" "${EVALUATOR}" \
  --config "${RUNTIME_CONFIG_PATH}" \
  --iterations "${ITERATIONS}"
