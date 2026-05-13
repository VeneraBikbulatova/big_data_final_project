E-commerce Purchase Prediction at Scale

This repository contains an end-to-end Big Data machine learning pipeline
designed to predict e-commerce purchase events. The system processes 42.4
million events to optimize marketing spend by identifying high-intent user
sessions.

Project Structure

  - config/: Project configuration files.
  - docs/: Technical reports and project documentation.
  - stage1_ingestion/: Data collection (Kaggle), bulk loading into PostgreSQL,
    and Sqoop ingestion to HDFS.
  - stage2/: Hive infrastructure setup, partitioning, bucketing, and EDA
    queries.
  - stage3_ml/: PySpark ML pipeline, custom CyclicalEncoder, model training, and
    evaluation.
  - stage4.sh: Deployment of results to Apache Superset via Hive registration.
  - main.sh: Main orchestrator for the end-to-end pipeline.
  - stage1.sh - stage4.sh: Individual scripts to run specific stages.

Execution

The pipeline is fully automated and idempotent. To run the complete process from
data ingestion to dashboard deployment:

bash main.sh

Individual stages can also be executed separately:

bash stage1.sh  # Data Ingestion (PostgreSQL & HDFS)
bash stage2.sh  # Hive Setup & EDA
bash stage3.sh  # ML Training & Evaluation
bash stage4.sh  # Deployment & Dashboarding

Technical Overview

  - Ingestion & Storage: Automated retrieval of 5.6 GB of raw CSV data; bulk
    loading into PostgreSQL; Sqoop ingestion into HDFS using Avro and Snappy
    compression.
  - Data Processing: Hive tables partitioned by event_type and bucketed by
    user_id for efficient analytical queries.
  - Feature Engineering: Custom pyspark.ml.Transformer for cyclical time
    encoding of hour and dayOfWeek to capture temporal patterns.
  - Modeling: Weighted Logistic Regression optimized for PR-AUC and Recall,
    addressing a 56:1 class imbalance.
  - Evaluation: Simulation of economic impact using a price-surge model to
    validate price elasticity and marketing lift.

Authors

Team 9 | Albina Akhmetova, Venera Bikbulatova.
