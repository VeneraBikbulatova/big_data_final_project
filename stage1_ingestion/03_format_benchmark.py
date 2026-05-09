#!/usr/bin/env python3
import os
import time
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAW_DATA_DIR = os.environ['RAW_DATA_DIR']
DATA_FILE = os.environ['DATA_FILE']
CSV_PATH = os.path.join(RAW_DATA_DIR, DATA_FILE)
TMP_DIR = "/tmp/format_benchmark"
SAMPLE_ROWS = 5_000_000 # reduced for testing purposes

def get_dir_size(path):
    s = 0
    for root, _, files in os.walk(path):
        for f in files:
            s += os.path.getsize(os.path.join(root, f))
    return s / (1024 * 1024)

def run_test_query(df):
    t0 = time.time()
    df.select("event_type", "price", "user_id") \
      .filter(F.col("event_type") == "purchase") \
      .groupBy("event_type") \
      .agg(F.count("*"), F.avg("price")) \
      .collect()
    return time.time() - t0

def main():
    spark = SparkSession.builder \
        .appName("FormatBenchmark") \
        .getOrCreate()

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    df = spark.read.csv(CSV_PATH, header=True, inferSchema=True).limit(SAMPLE_ROWS).cache()
    df.count()

    pq_path = os.path.join(TMP_DIR, "data.parquet")
    t0 = time.time()
    df.write.mode("overwrite").parquet(pq_path)
    pq_w_time = time.time() - t0
    pq_size = get_dir_size(pq_path)
    
    pq_df = spark.read.parquet(pq_path)
    pq_r_time = run_test_query(pq_df)

    av_path = os.path.join(TMP_DIR, "data.avro")
    t0 = time.time()
    df.write.mode("overwrite").format("avro").save(av_path)
    av_w_time = time.time() - t0
    av_size = get_dir_size(av_path)
    
    av_df = spark.read.format("avro").load(av_path)
    av_r_time = run_test_query(av_df)

    print("-" * 50)
    print(f"Parquet: Size={pq_size:.2f}MB, Write={pq_w_time:.2f}s, Read={pq_r_time:.2f}s")
    print(f"Avro:    Size={av_size:.2f}MB, Write={av_w_time:.2f}s, Read={av_r_time:.2f}s")
    print("-" * 50)

    spark.stop()
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

if __name__ == "__main__":
    main()