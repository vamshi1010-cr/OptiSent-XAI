#!/usr/bin/env bash
# ===============================================================================
# OptiSent-XAI: Explainable Multi-Class Product Review Triage & Calibration
# Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad (22ITI02)
# Navodita Infotech, Pune (Intern ID: 10110, Certificate ID: CNI2606698)
# ===============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "==============================================================================="
echo "  OptiSent-XAI Kinetic Visual Observatory Launcher (22ITI02)"
echo "==============================================================================="

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[WARNING] Python not found. Opening standalone HTML in default browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open index.html
    elif command -v open &> /dev/null; then
        open index.html
    fi
    exit 0
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "[1/2] Launching FastAPI Live Inference Kernel on http://127.0.0.1:8000 ..."
$PYTHON_CMD -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload &
SERVER_PID=$!

echo "[2/2] Opening browser..."
sleep 2

URL="http://127.0.0.1:8000"
if command -v xdg-open &> /dev/null; then
    xdg-open "$URL"
elif command -v open &> /dev/null; then
    open "$URL"
else
    echo "Please open $URL in your browser."
fi

echo "Server running with PID $SERVER_PID. Press Ctrl+C to terminate."
wait $SERVER_PID
