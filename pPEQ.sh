#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "No Python virtual environment detected. Creating one ..."
    python3 -m venv "${SCRIPT_DIR}/venv"
    source "${SCRIPT_DIR}/venv/bin/activate"
    pip3 install -U pip wheel setuptools
    pip3 install -r "${SCRIPT_DIR}/requirements.txt"
else
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

python3 "${SCRIPT_DIR}/pPEQ.py"
