#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ML_DIR="${SCRIPT_DIR}/stage3_ml"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/stage3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "  STAGE 3 - PREDICTIVE DATA ANALYTICS"
echo "  Started: $(date)"

export HIVE_TABLE="${HIVE_TABLE:-team9_projectdb.events_partitioned}"
export HDFS_OUT_DIR="${HDFS_OUT_DIR:-/user/team9/project/stage3}"
export SAMPLE_FRACTION="${SAMPLE_FRACTION:-1.0}"

echo "[INFO] Clearing previous HDFS outputs at ${HDFS_OUT_DIR}"
hdfs dfs -rm -r -f -skipTrash "${HDFS_OUT_DIR}" 2>/dev/null || true
hdfs dfs -mkdir -p "${HDFS_OUT_DIR}"

spark-submit \
    --master yarn \
    --deploy-mode client \
    --name "Stage3_PurchasePrediction" \
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
    --conf spark.dynamicAllocation.enabled=false \
    --py-files "${ML_DIR}/cyclical_encoder.py" \
    "${ML_DIR}/stage3_ml.py"

echo ""
echo "[INFO] HDFS outputs:"
hdfs dfs -ls -R "${HDFS_OUT_DIR}" || true

echo "  STAGE 3 COMPLETE - Finished: $(date)"
echo "  Log: ${LOG_FILE}"