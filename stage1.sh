#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config/config.env"

echo "[INFO] Cleaning up HDFS for Stage 1..."
hdfs dfs -rm -r -f -skipTrash /user/team9/ecommerce_project/raw_events_parquet 2>/dev/null || true

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/stage1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "STAGE 1 STARTED: $(date)"

bash "${SCRIPT_DIR}/stage1_ingestion/01_download_data.sh"

echo "[1.2] Loading CSV to PostgreSQL..."
python3 "${SCRIPT_DIR}/stage1_ingestion/02_load_to_postgres.py"

echo "[1.3] Benchmarking storage formats..."
spark-submit --master local[*] --packages org.apache.spark:spark-avro_2.12:3.2.4 "${SCRIPT_DIR}/stage1_ingestion/03_format_benchmark.py"

bash "${SCRIPT_DIR}/stage1_ingestion/04_sqoop_import.sh"

echo "STAGE 1 COMPLETE: $(date)"
