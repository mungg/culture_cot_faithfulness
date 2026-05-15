#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
cd "$ROOT_DIR"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

export HF_HOME="${HF_HOME:-/scratch/login/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(<"${HOME}/.cache/huggingface/token")"
  export HF_TOKEN
fi

echo "host=$(hostname)"
echo "cwd=$(pwd)"
echo "model=${MODEL_SPEC:?MODEL_SPEC is required}"
echo "dataset=${DATASET:?DATASET is required}"
echo "run_name=${RUN_NAME:?RUN_NAME is required}"

"$PYTHON_BIN" scripts/validate_mcqa.py \
  --dataset "${DATASET:-all}" \
  --culture "${CULTURE:-all}"

cmd=(
  "$PYTHON_BIN" scripts/run_pipeline.py
  --model "$MODEL_SPEC"
  --dataset "$DATASET"
  --run-name "$RUN_NAME"
  --temperature "${TEMPERATURE:-1.0}"
  --seed "${SEED:-42}"
)

if [[ -n "${CULTURE:-}" ]]; then
  cmd+=(--culture "$CULTURE")
fi

if [[ -n "${LANGUAGE:-}" ]]; then
  cmd+=(--language "$LANGUAGE")
fi

if [[ -n "${MAX_TOKENS:-}" ]]; then
  cmd+=(--max-tokens "$MAX_TOKENS")
fi

if [[ -n "${BIASED_GEN_MODEL:-}" ]]; then
  cmd+=(--biased-gen-model "$BIASED_GEN_MODEL")
fi

if [[ -n "${BASELINE_LIMIT:-}" ]]; then
  cmd+=(--baseline-limit "$BASELINE_LIMIT")
fi

if [[ -n "${HINT_LIMIT_ITEMS:-}" ]]; then
  cmd+=(--hint-limit-items "$HINT_LIMIT_ITEMS")
fi

if [[ "${WANT_THINKING:-0}" == "1" ]]; then
  cmd+=(--want-thinking)
fi

echo "running: ${cmd[*]}"
"${cmd[@]}"
