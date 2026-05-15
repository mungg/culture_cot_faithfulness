#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
cd "$ROOT_DIR"

module load cuda/12.8
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

PIPELINE_PYTHON_BIN="${PIPELINE_PYTHON_BIN:-python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:?VLLM_PYTHON_BIN is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
VLLM_MODEL_SPEC="${VLLM_MODEL_SPEC:?VLLM_MODEL_SPEC is required}"

VLLM_BIN_DIR="$(dirname "$VLLM_PYTHON_BIN")"
export PATH="${VLLM_BIN_DIR}:$PATH"

export HF_HOME="${HF_HOME:-/scratch/login/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(<"${HOME}/.cache/huggingface/token")"
  export HF_TOKEN
fi

echo "host=$(hostname)"
echo "cwd=$(pwd)"
echo "vllm_model=${VLLM_MODEL_SPEC}"
echo "served_model=${SERVED_MODEL_NAME}"
echo "dataset=${DATASET:?DATASET is required}"
echo "run_name=${RUN_NAME:?RUN_NAME is required}"

pick_free_port() {
  "$PIPELINE_PYTHON_BIN" - <<'PY'
import socket

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

API_PORT="$(pick_free_port)"
DP_MASTER_PORT="$(pick_free_port)"
export VLLM_DP_MASTER_PORT="$DP_MASTER_PORT"

echo "api_port=${API_PORT}"
echo "dp_master_port=${DP_MASTER_PORT}"

server_log="${ROOT_DIR}/slurm_logs/${RUN_NAME}.${SLURM_JOB_ID}.server.log"

"$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL_SPEC" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null; then
  echo "vLLM server did not become ready; see ${server_log}" >&2
  exit 1
fi

export OPENAI_BASE_URL="http://127.0.0.1:${API_PORT}/v1"
export OPENAI_LOCAL_API_KEY="${OPENAI_LOCAL_API_KEY:-EMPTY}"

cmd=(
  "$PIPELINE_PYTHON_BIN" scripts/run_pipeline.py
  --model "openai-local/${SERVED_MODEL_NAME}"
  --dataset "$DATASET"
  --run-name "$RUN_NAME"
  --temperature "${TEMPERATURE:-1.0}"
  --seed "${SEED:-42}"
)

if [[ -n "${LANGUAGE:-}" ]]; then
  cmd+=(--language "$LANGUAGE")
fi

if [[ -n "${CULTURE:-}" ]]; then
  cmd+=(--culture "$CULTURE")
fi

if [[ -n "${MAX_TOKENS:-}" ]]; then
  cmd+=(--max-tokens "$MAX_TOKENS")
fi

echo "running: ${cmd[*]}"
"${cmd[@]}"
