#!/bin/bash
# download kaggle dataset
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/../config/config.env"

mkdir -p "${RAW_DATA_DIR}"
cd "${RAW_DATA_DIR}"

if [[ -f "${DATA_FILE}" ]]; then
    SIZE=$(du -h "${DATA_FILE}" | cut -f1)
    echo "[INFO] ${DATA_FILE} already present (${SIZE}). Skipping download."
    exit 0
fi

if ! command -v kaggle &> /dev/null; then
    echo "[ERROR] 'kaggle' CLI not found."
    echo "Install: pip install --user kaggle"
    echo "Auth: place token at ~/.kaggle/kaggle.json (chmod 600)"
    exit 1
fi

kaggle datasets download -d "${KAGGLE_DATASET}" -f "${DATA_FILE}" -p . --force

if [[ -f "${DATA_FILE}.zip" ]]; then
    echo "[INFO] Unzipping..."
    unzip -o "${DATA_FILE}.zip"
    rm -f "${DATA_FILE}.zip"
fi

echo "[INFO] Download complete: $(du -h ${DATA_FILE} | cut -f1)"