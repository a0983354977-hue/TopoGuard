#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

if [ -x ".venv-build/bin/python" ]; then
    PYTHON_BIN=".venv-build/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "[ERROR] Python was not found."
    exit 1
fi

printf '%s\n' \
    '========================================================' \
    '  ___  ___  ____  ___' \
    ' / _ \/ _ \/ __ \/ _ \' \
    '/ .__/\___/ /_/ / __/' \
    '/_/       /____/_/' \
    '  TOPOGUARD SECURITY PROBE - LIVE MONITORING MODE' \
    '  Hard Bottom, Evolving Top | Read-Only SIEM' \
    '========================================================' \
    ''

"$PYTHON_BIN" topoguard_cli.py monitor --source system --ticks 100
exit_code=$?

printf '\n========================================================\n'
if [ "$exit_code" -eq 0 ]; then
    echo "  Monitoring process ended."
else
    echo "  Monitoring ended with exit code $exit_code."
fi
echo "  Press Enter to close."
echo "========================================================"
read -r _ || true
exit "$exit_code"
