#!/bin/bash
# Multi-turn inference pipeline for the Critic (SQL debugging) model.

set -euo pipefail

GPU="0"
MAX_TURNS=5
MAX_TOKENS=3000
MAX_MODEL_LEN=20000
TEMPERATURE=0.0
BATCH_SIZE=100
GPU_MEMORY_UTILIZATION=0.7
ENFORCE_EAGER_ARGS=()
LIMIT_ARGS=()
SESSION_LIMIT_ARGS=()
EVAL_THREADS=1
EVAL_BATCH_SIZE=1
RUN_EVALUATION=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --db_dir) DB_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --max_turns) MAX_TURNS="$2"; shift 2 ;;
        --max_tokens) MAX_TOKENS="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --enforce_eager) ENFORCE_EAGER_ARGS=(--enforce_eager); shift ;;
        --limit)
            LIMIT_ARGS=(--limit "$2")
            SESSION_LIMIT_ARGS=(--limit "$2")
            shift 2
            ;;
        --eval_threads) EVAL_THREADS="$2"; shift 2 ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --skip_evaluation) RUN_EVALUATION=0; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

for var in MODEL_PATH INPUT DB_DIR OUTPUT_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "Error: --$(echo "$var" | tr '[:upper:]' '[:lower:]') is required"
        exit 1
    fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAJ_DIR="${OUTPUT_DIR}/trajectories"
SESSION_DIR="${OUTPUT_DIR}/session_dbs"
mkdir -p "${TRAJ_DIR}"

cleanup_sessions() {
    python -m bird_rl.inference.critic_session cleanup \
        --session-dir "${SESSION_DIR}" >/dev/null 2>&1 || true
}
trap cleanup_sessions EXIT

echo "============================================================"
echo "Critic Multi-Turn Inference Pipeline"
echo "============================================================"
echo "Model: ${MODEL_PATH}"
echo "Input: ${INPUT}"
echo "DB dir: ${DB_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max turns: ${MAX_TURNS}"
echo "GPU: ${GPU}"
echo "============================================================"

echo "[Setup] Creating and preprocessing isolated trajectory databases..."
python -m bird_rl.inference.critic_session init \
    --input "${INPUT}" \
    --db-dir "${DB_DIR}" \
    --session-dir "${SESSION_DIR}" \
    --force \
    "${SESSION_LIMIT_ARGS[@]}"

for TURN in $(seq 0 $((MAX_TURNS - 1))); do
    echo ""
    echo "==================== Turn ${TURN} ===================="

    PROMPT_FILE="${OUTPUT_DIR}/prompts_turn_${TURN}.jsonl"
    RESPONSE_FILE="${OUTPUT_DIR}/responses_turn_${TURN}.jsonl"
    PARSED_FILE="${OUTPUT_DIR}/parsed_turn_${TURN}.jsonl"
    OBS_FILE="${OUTPUT_DIR}/observations_turn_${TURN}.jsonl"
    TRAJ_FILE="${TRAJ_DIR}/traj_${TURN}.jsonl"

    echo "[Step 1] Generating prompts..."
    python -m bird_rl.inference.critic.generate_prompts \
        --turn "${TURN}" \
        --max-turns "${MAX_TURNS}" \
        --input "${INPUT}" \
        --db-dir "${DB_DIR}" \
        --session-dir "${SESSION_DIR}" \
        --traj-dir "${TRAJ_DIR}" \
        --output "${PROMPT_FILE}" \
        "${LIMIT_ARGS[@]}"

    if [ ! -s "${PROMPT_FILE}" ]; then
        echo "No prompts generated (all instances finished). Stopping."
        break
    fi

    echo "[Step 2] Running vLLM inference..."
    python -m bird_rl.inference.vllm_infer \
        --model_path "${MODEL_PATH}" \
        --prompt_path "${PROMPT_FILE}" \
        --output_path "${RESPONSE_FILE}" \
        --gpu "${GPU}" \
        --batch_size "${BATCH_SIZE}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        "${ENFORCE_EAGER_ARGS[@]}"

    echo "[Step 3] Parsing responses..."
    python -m bird_rl.inference.parse_responses \
        --input "${RESPONSE_FILE}" \
        --output "${PARSED_FILE}"

    echo "[Step 4] Executing SQL against trajectory databases..."
    python -m bird_rl.inference.execute_sql_observations \
        --input "${PARSED_FILE}" \
        --output "${OBS_FILE}" \
        --db-dir "${DB_DIR}" \
        --session-dir "${SESSION_DIR}"

    echo "[Step 5] Building trajectory..."
    python -m bird_rl.inference.build_trajectory \
        --turn "${TURN}" \
        --traj-dir "${TRAJ_DIR}" \
        --observations "${OBS_FILE}" \
        --output "${TRAJ_FILE}" \
        --submit-format sql_list
done

FINAL_TRAJ=$(ls -t "${TRAJ_DIR}"/traj_*.jsonl 2>/dev/null | head -1 || true)
if [ -n "${FINAL_TRAJ}" ]; then
    echo ""
    echo "==================== Preparing Evaluation ===================="
    EVAL_FILE="${OUTPUT_DIR}/eval_ready.jsonl"
    python -m bird_rl.inference.critic.evaluate \
        --trajectory "${FINAL_TRAJ}" \
        --original-data "${INPUT}" \
        --output "${EVAL_FILE}" \
        "${LIMIT_ARGS[@]}"

    if [ "${RUN_EVALUATION}" -eq 1 ]; then
        echo ""
        echo "==================== Official Evaluator ===================="
        bash "${REPO_ROOT}/evaluation/critic/run/run_eval.sh" \
            --jsonl_file "${EVAL_FILE}" \
            --db_dir "${DB_DIR}" \
            --mode pred \
            --num_threads "${EVAL_THREADS}" \
            --batch_size "${EVAL_BATCH_SIZE}"
    fi
fi

echo ""
echo "============================================================"
echo "Pipeline complete. Results in: ${OUTPUT_DIR}"
echo "Trajectory databases will now be cleaned."
echo "============================================================"
