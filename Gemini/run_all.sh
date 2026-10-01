#!/usr/bin/env bash
# ==============================================================================
# Master Automation Script: Gemini Compressed Sensing Cross-Validation Suite
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"

if [ ! -f "${PYTHON}" ]; then
    echo "Error: Python virtual environment not found at ${PYTHON}"
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${SCRIPT_DIR}/src:${PYTHONPATH}"

echo "===================================================================="
echo "Running Unit Test Suite..."
echo "===================================================================="
"${PYTHON}" -m unittest discover "${SCRIPT_DIR}/tests"

echo ""
echo "===================================================================="
echo "Running Fast End-to-End Pipeline Verification..."
echo "===================================================================="
"${PYTHON}" "${SCRIPT_DIR}/src/run_pipeline.py" --mode fast

echo ""
echo "===================================================================="
echo "All tests and pipeline stages verified successfully!"
echo "To run full production experiments with 100 trials per cell:"
echo "    ${PYTHON} ${SCRIPT_DIR}/src/run_pipeline.py --mode full"
echo "===================================================================="
