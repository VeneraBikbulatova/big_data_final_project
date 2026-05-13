#!/bin/bash
# ============================================================
# Stage 4 - Register Stage 3 CSVs as external Hive tables
# so that Apache Superset can query them.
#
# Reads the Hive password from secrets/.hive.pass (one line,
# trailing newline tolerated). Idempotent through the HQL itself.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HQL_FILE="${SCRIPT_DIR}/sql/create_results_tables.hql"
PASSWORD_FILE="${SCRIPT_DIR}/secrets/.hive.pass"

mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/stage4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "  STAGE 4 - PRESENTATION & DELIVERY"
echo "  Started: $(date)"
echo "================================================================"

[[ -f "${HQL_FILE}"      ]] || { echo "[ERROR] Missing HQL file: ${HQL_FILE}"; exit 1; }
[[ -f "${PASSWORD_FILE}" ]] || { echo "[ERROR] Missing password file: ${PASSWORD_FILE}"; exit 1; }

chmod 600 "${PASSWORD_FILE}" || true

HIVE_HOST="${HIVE_HOST:-hadoop-03.uni.innopolis.ru}"
HIVE_PORT="${HIVE_PORT:-10001}"
HIVE_USER="${HIVE_USER:-team9}"
HIVE_DB="${HIVE_DB:-team9_projectdb}"

HIVE_PASSWORD="$(tr -d '\r\n' < "${PASSWORD_FILE}")"
JDBC_URL="jdbc:hive2://${HIVE_HOST}:${HIVE_PORT}/${HIVE_DB}"

echo "[INFO] Connecting to ${JDBC_URL} as ${HIVE_USER}"
echo "[INFO] Executing ${HQL_FILE}"

beeline \
    -u "${JDBC_URL}" \
    -n "${HIVE_USER}" \
    -p "${HIVE_PASSWORD}" \
    --silent=false \
    --showHeader=true \
    --outputformat=table \
    -f "${HQL_FILE}"

echo ""
echo "[INFO] All result tables registered in ${HIVE_DB}."
echo "[INFO] You can now connect Superset to Hive and build the"
echo "       dashboards described in docs/stage4/."

echo "================================================================"
echo "  STAGE 4 COMPLETE - Finished: $(date)"
echo "  Log: ${LOG_FILE}"
echo "================================================================"
