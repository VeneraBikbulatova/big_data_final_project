#!/bin/bash
source config/config.env

hdfs dfs -mkdir -p /user/team9/ecommerce_project

sqoop import \
--connect jdbc:postgresql://hadoop-01.uni.innopolis.ru:5432/team9_projectdb \
--username team9 \
--password $PG_PASSWORD \
--table events \
--target-dir /user/team9/ecommerce_project/raw_events_parquet \
--delete-target-dir \
--as-parquetfile \
--compression-codec snappy \
--split-by id \
--num-mappers 4