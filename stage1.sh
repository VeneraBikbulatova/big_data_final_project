#!/bin/bash

bash stage1_ingestion/01_download_data.sh
python3 stage1_ingestion/02_load_to_postgres.py

spark-submit --packages org.apache.spark:spark-avro_2.12:3.3.0 stage1_ingestion/03_format_benchmark.py

bash stage1_ingestion/04_sqoop_import.sh