#!/usr/bin/env bash
# ==============================================================================
# Production Service Launcher for AI Proctoring Engine
# ==============================================================================
# Starts the FastAPI HTTP/WebSocket service on the canonical ExamController
# integration port (7001) with GPU acceleration and signal handling.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

# 1. Environment & Port Configuration
export AI_HOST="${AI_HOST:-0.0.0.0}"
export AI_PORT="${AI_PORT:-7001}"
export AI_WORKERS="${AI_WORKERS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 2. Virtual Environment Resolution
if [[ -d "${REPO_DIR}/.venv" ]]; then
    PYTHON="${REPO_DIR}/.venv/bin/python"
    UVICORN="${REPO_DIR}/.venv/bin/uvicorn"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
    UVICORN="$(command -v uvicorn)"
else
    echo "[ERROR] No Python runtime found in .venv or system PATH." >&2
    exit 1
fi

echo "=================================================================="
echo " Starting AI Proctoring Engine (Production Mode)"
echo " Repository:  ${REPO_DIR}"
echo " Host / Port: ${AI_HOST}:${AI_PORT}"
echo " CUDA Device: ${CUDA_VISIBLE_DEVICES}"
echo " Python:      $("${PYTHON}" --version)"
echo "=================================================================="

# 3. Model Integrity Verification
MODELS_DIR="${REPO_DIR}/models"
REQUIRED_MODELS=(
    "face_detection_yunet_2023mar.onnx"
    "face_recognition_sface_2021dec.onnx"
    "face_landmarker.task"
    "hand_landmarker.task"
    "yolo11n.pt"
)

MISSING_MODELS=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [[ ! -f "${MODELS_DIR}/${model}" ]]; then
        echo "[WARNING] Required model missing: ${MODELS_DIR}/${model}" >&2
        MISSING_MODELS=$((MISSING_MODELS + 1))
    fi
done

if [[ ${MISSING_MODELS} -gt 0 ]]; then
    echo "[ERROR] ${MISSING_MODELS} required models missing from ${MODELS_DIR}." >&2
    exit 1
fi

echo "[INFO] All required neural models present in ${MODELS_DIR}."

# 4. Graceful Signal Handling
cleanup() {
    echo "[INFO] Received termination signal. Shutting down AI Proctoring Engine..."
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}"
        wait "${SERVER_PID}" || true
    fi
    echo "[INFO] Service successfully stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM SIGHUP

# 5. Launch Canonical FastAPI Engine via Uvicorn Factory
"${UVICORN}" proctoring.integration.api:create_app \
    --factory \
    --host "${AI_HOST}" \
    --port "${AI_PORT}" \
    --workers "${AI_WORKERS}" \
    --log-level info \
    --access-log &

SERVER_PID=$!
echo "[INFO] Server started with PID: ${SERVER_PID}"

wait "${SERVER_PID}"
