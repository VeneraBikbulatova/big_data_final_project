#!/bin/bash
# ============================================================
# Stage 2 - Data Storage & EDA on YARN.
#
# 1. Runs stage2_eda.py via spark-submit on the YARN cluster.
#
# Idempotent:
#   * The Python script itself handles the case where the raw
#     parquet is already gone (smart-idempotency path).
#   * The HDFS cleanup checks existence before deleting.
#   * The local output/ directory is cleared at the start of
#     every run.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EDA_DIR="${SCRIPT_DIR}/stage2_storage_eda"
PYTHON_SCRIPT="${EDA_DIR}/stage2_eda.py"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/output"
LOG_FILE="${SCRIPT_DIR}/logs/stage2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "  STAGE 2 - DATA STORAGE & EDA"
echo "  Started: $(date)"
echo "================================================================"

export HIVE_DATABASE="${HIVE_DATABASE:-team9_projectdb}"
export HIVE_EVENTS_TABLE="${HIVE_EVENTS_TABLE:-events_partitioned}"
export RAW_PARQUET_PATH="${RAW_PARQUET_PATH:-/user/team9/ecommerce_project/raw_events_parquet}"
export EVENTS_TABLE_LOCATION="${EVENTS_TABLE_LOCATION:-/user/team9/project/warehouse/events_partitioned}"
export HIVE_METASTORE_URI="${HIVE_METASTORE_URI:-thrift://hadoop-02.uni.innopolis.ru:9883}"
export LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-${SCRIPT_DIR}/output}"

# ---- Pre-flight ----
[[ -f "${PYTHON_SCRIPT}" ]] \
    || { echo "[ERROR] Missing ${PYTHON_SCRIPT}"; exit 1; }

# Idempotency: clear stale local CSVs from previous runs
rm -f "${LOCAL_OUTPUT_DIR}"/*.csv 2>/dev/null || true

# ---- Submit to YARN ----
spark-submit \
    --master yarn \
    --deploy-mode client \
    --name "Stage2_StorageAndEDA" \
    --num-executors 3 \
    --executor-cores 3 \
    --executor-memory 5G \
    --driver-memory 3G \
    --conf spark.executor.memoryOverhead=1024 \
    --conf spark.driver.memoryOverhead=512 \
    --conf spark.sql.shuffle.partitions=200 \
    --conf spark.default.parallelism=200 \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    --conf spark.sql.adaptive.skewJoin.enabled=true \
    --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
    --conf spark.kryoserializer.buffer.max=512m \
    --conf spark.network.timeout=600s \
    --conf spark.executor.heartbeatInterval=60s \
    --conf spark.yarn.maxAppAttempts=1 \
    --conf spark.hadoop.hive.metastore.uris="${HIVE_METASTORE_URI}" \
    --conf hive.metastore.uris="${HIVE_METASTORE_URI}" \
    "${PYTHON_SCRIPT}"

echo ""
echo "[INFO] PySpark pipeline succeeded."


echo "[INFO] Removing raw Stage 1 parquet to stay under the 32 GB quota..."
if hdfs dfs -test -d "${RAW_PARQUET_PATH}"; then
    hdfs dfs -rm -r -f -skipTrash "${RAW_PARQUET_PATH}"
    echo "[INFO] Deleted ${RAW_PARQUET_PATH}."
else
    echo "[INFO] ${RAW_PARQUET_PATH} already absent - nothing to clean."
fi

echo ""
echo "[INFO] Local insight CSVs:"
ls -la "${LOCAL_OUTPUT_DIR}"/*.csv 2>/dev/null \
    || echo "[WARN] No CSVs found in ${LOCAL_OUTPUT_DIR}"

echo "================================================================"
echo "  STAGE 2 COMPLETE - Finished: $(date)"
echo "  Log: ${LOG_FILE}"
echo "================================================================"
