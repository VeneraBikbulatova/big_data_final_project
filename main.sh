#!/bin/bash
set -euo pipefail

echo "Starting full pipeline..."

bash stage1.sh
bash stage2.sh
bash stage3.sh
bash stage4.sh

echo "All stages done!"
