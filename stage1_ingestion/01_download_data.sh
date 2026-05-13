#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../config/config.env"

mkdir -p "${RAW_DATA_DIR}"
cd "${RAW_DATA_DIR}"

if [[ -f "${DATA_FILE}" ]]; then
    SIZE=$(du -h "${DATA_FILE}" | cut -f1)
    echo "[INFO] ${DATA_FILE} already present (${SIZE}). Skipping download."
    exit 0
else
    echo "[ERROR] ${DATA_FILE} not found in ${RAW_DATA_DIR}"
    echo "Please upload the file manually to this location."
    exit 1
fi
