import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import (StringIndexer, OneHotEncoder,
                                VectorAssembler, StandardScaler)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from cyclical_encoder import CyclicalEncoder


# Configuration
HIVE_TABLE      = os.environ.get("HIVE_TABLE",
                                 "team9_projectdb.events_partitioned")
HDFS_OUT_DIR    = os.environ.get("HDFS_OUT_DIR",
                                 "/user/team9/project/stage3")
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "1.0"))

SEED        = 42
TRAIN_RATIO = 0.6
TEST_RATIO  = 0.4
NUM_FOLDS   = 3     
PARALLELISM = 2

TOP_N_BRAND = 100
TOP_N_CAT   = 50

P_TRAIN          = f"{HDFS_OUT_DIR}/data/train"
P_TEST           = f"{HDFS_OUT_DIR}/data/test"
P_LR_MODEL       = f"{HDFS_OUT_DIR}/models/best_lr"
P_RF_MODEL       = f"{HDFS_OUT_DIR}/models/best_rf"
P_COMPARISON     = f"{HDFS_OUT_DIR}/results/model_comparison_csv"
P_PREDICTIONS    = f"{HDFS_OUT_DIR}/results/best_model_predictions_csv"
P_RF_IMPORTANCES = f"{HDFS_OUT_DIR}/results/rf_feature_importances_csv"

# Spark session
def build_spark():
    return (SparkSession.builder
            .appName("Stage3_PurchasePrediction")
            .enableHiveSupport()
            .getOrCreate())

# Data loading + base feature engineering
def load_and_prepare(spark):
    """
    Read partitioned Hive table, build binary label, decompose
    event_time, fill nulls, optionally stratified-sample.
    """
    print(f"[INFO] Reading {HIVE_TABLE} ...")
    df = spark.table(HIVE_TABLE)

    df = (df
          # Binary target: 1 for purchase, 0 for view/cart
          .withColumn("label",
                      (F.col("event_type") == "purchase").cast("int"))
          # Time decomposition
          .withColumn("hour",      F.hour("event_time"))
          .withColumn("dayOfWeek", F.dayofweek("event_time") - 1)   # 0..6
          # Null handling (brand / category_code are nullable in source)
          .fillna({"brand": "unknown", "category_code": "unknown"})
          .fillna({"price": 0.0})
          .select("label", "hour", "dayOfWeek",
                  "price", "brand", "category_code"))

    # Stratified sample to preserve the imbalance ratio
    if SAMPLE_FRACTION < 1.0:
        print(f"[INFO] Stratified sample at fraction={SAMPLE_FRACTION}")
        df = df.stat.sampleBy(
            "label",
            fractions={0: SAMPLE_FRACTION, 1: SAMPLE_FRACTION},
            seed=SEED,
        )

    return df

# Class-weight computation (from training data only)
def add_class_weight_column(train_df, test_df):

    counts = {row["label"]: row["count"]
              for row in train_df.groupBy("label").count().collect()}
    n_neg, n_pos = counts.get(0, 0), counts.get(1, 0)
    total = n_neg + n_pos
    K = 2
    w_neg = total / (K * n_neg) if n_neg else 1.0
    w_pos = total / (K * n_pos) if n_pos else 1.0

    print(f"[INFO] Train class counts: neg={n_neg:,}  pos={n_pos:,}  "
          f"(positive rate = {100*n_pos/total:.2f}%)")
    print(f"[INFO] Class weights: w_neg={w_neg:.4f}  w_pos={w_pos:.4f}")

    weight_expr = F.when(F.col("label") == 1, w_pos).otherwise(w_neg)
    train_df = train_df.withColumn("classWeight", weight_expr)
    test_df  = test_df.withColumn("classWeight",  weight_expr)
    return train_df, test_df


# Cardinality capping (top-N, computed from TRAIN ONLY)
def top_n_values(df, col_name, top_n):
    #Return the top_n most frequent values of a column as a Python list.
    rows = (df.groupBy(col_name).count()
              .orderBy(F.desc("count"))
              .limit(top_n)
              .collect())
    return [r[col_name] for r in rows]


def cap_to_top_n(df, col_name, keep_values, suffix="_capped"):
    #Replace values outside keep_values with the literal 'other'.
    return df.withColumn(
        f"{col_name}{suffix}",
        F.when(F.col(col_name).isin(keep_values), F.col(col_name))
         .otherwise(F.lit("other")),
    )


# Pipeline construction
def build_pipeline(classifier):
    # Cyclical encoding for time
    hour_enc = CyclicalEncoder(inputCol="hour",      period=24.0)
    dow_enc  = CyclicalEncoder(inputCol="dayOfWeek", period=7.0)

    # Categorical -> index -> one-hot
    brand_idx = StringIndexer(inputCol="brand_capped",
                              outputCol="brand_idx",
                              handleInvalid="keep")
    cat_idx   = StringIndexer(inputCol="category_code_capped",
                              outputCol="cat_idx",
                              handleInvalid="keep")
    brand_ohe = OneHotEncoder(inputCol="brand_idx",
                              outputCol="brand_vec",
                              handleInvalid="keep")
    cat_ohe   = OneHotEncoder(inputCol="cat_idx",
                              outputCol="cat_vec",
                              handleInvalid="keep")

    # Assemble
    assembler = VectorAssembler(
        inputCols=[
            "hour_sin", "hour_cos",
            "dayOfWeek_sin", "dayOfWeek_cos",
            "price",
            "brand_vec", "cat_vec",
        ],
        outputCol="features_raw",
        handleInvalid="keep",
    )

    # Scale (withMean=False keeps the vector sparse - critical for OHE)
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=False,
        withStd=True,
    )

    return Pipeline(stages=[
        hour_enc, dow_enc,
        brand_idx, cat_idx,
        brand_ohe, cat_ohe,
        assembler, scaler,
        classifier,
    ])

# Cross-validated training
def tune(name, pipeline, param_grid, train_df):
    print(f"[INFO] CV tuning {name}: "
          f"{len(param_grid)} combos x {NUM_FOLDS} folds = "
          f"{len(param_grid)*NUM_FOLDS} fits")
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=NUM_FOLDS,
        seed=SEED,
        parallelism=PARALLELISM,
    )
    t0 = time.time()
    model = cv.fit(train_df)
    print(f"[INFO] {name} CV finished in {time.time()-t0:.1f}s. "
          f"Best avg ROC-AUC across folds = {max(model.avgMetrics):.4f}")
    return model


# Evaluation (ROC-AUC + PR-AUC)
def evaluate(model, test_df, name):
    preds = model.transform(test_df).cache()
    auc_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC").evaluate(preds)
    auc_pr  = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderPR").evaluate(preds)
    print(f"[INFO] {name:<25}  ROC-AUC={auc_roc:.4f}  PR-AUC={auc_pr:.4f}")
    return preds, auc_roc, auc_pr

# Feature names + RF importances export
def extract_feature_names(df, col_candidates=("features_raw", "features")):
    for col in col_candidates:
        if col not in df.columns:
            continue
        meta = df.schema[col].metadata
        ml_attr = meta.get("ml_attr") if meta else None
        if not ml_attr or "attrs" not in ml_attr:
            continue
        items = []
        for attr_list in ml_attr["attrs"].values():
            for attr in attr_list:
                items.append((attr["idx"], attr["name"]))
        if items:
            items.sort()
            return [name for _, name in items]
    return None


def export_rf_importances(spark, rf_pipeline_model, transformed_df, path):
    rf_model = rf_pipeline_model.stages[-1]
    importances = rf_model.featureImportances 
    names = extract_feature_names(transformed_df)

    if names and len(names) == importances.size:
        rows = [(names[i], float(importances[i]))
                for i in range(importances.size)]
    else:
        print(f"[WARN] Could not resolve feature names "
              f"(got {len(names) if names else 0}, "
              f"importances size {importances.size}). "
              f"Falling back to f_<idx>.")
        rows = [(f"f_{i}", float(importances[i]))
                for i in range(importances.size)]

    rows.sort(key=lambda r: r[1], reverse=True)
    df = spark.createDataFrame(rows, ["feature", "importance"])
    save_single_csv(df, path)


# CSV helper
def save_single_csv(df, path):
    #Write a small dataframe as ONE CSV file (idempotent).
    (df.coalesce(1)
       .write.mode("overwrite")
       .option("header", "true")
       .csv(path))

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    t_start = time.time()

    # Load + base features
    df = load_and_prepare(spark).cache()
    n_total = df.count()
    n_pos   = df.filter(F.col("label") == 1).count()
    print(f"[INFO] After load+sample: total={n_total:,}  "
          f"pos={n_pos:,} ({100*n_pos/n_total:.2f}%)")

    # Stratified 60/40 split
    train_df, test_df = df.randomSplit([TRAIN_RATIO, TEST_RATIO], seed=SEED)

    # Class weights (computed on TRAIN only) 
    train_df, test_df = add_class_weight_column(train_df, test_df)

    # Cardinality capping (top-N from TRAIN only, applied to both)
    print(f"[INFO] Capping brand to top-{TOP_N_BRAND}, "
          f"category_code to top-{TOP_N_CAT} ...")
    top_brands = top_n_values(train_df, "brand",         TOP_N_BRAND)
    top_cats   = top_n_values(train_df, "category_code", TOP_N_CAT)
    train_df = cap_to_top_n(train_df, "brand",         top_brands)
    train_df = cap_to_top_n(train_df, "category_code", top_cats)
    test_df  = cap_to_top_n(test_df,  "brand",         top_brands)
    test_df  = cap_to_top_n(test_df,  "category_code", top_cats)

    train_df = train_df.cache()
    test_df  = test_df.cache()
    print(f"[INFO] Train rows: {train_df.count():,}   "
          f"Test rows: {test_df.count():,}")

    # Persist splits as JSON (HDFS)
    print("[INFO] Writing train/test JSON to HDFS ...")
    train_df.write.mode("overwrite").json(P_TRAIN)
    test_df.write.mode("overwrite").json(P_TEST)

    # Logistic Regression (regParam + elasticNetParam) 
    lr = LogisticRegression(
        labelCol="label", featuresCol="features",
        weightCol="classWeight",
        maxIter=30,
    )
    lr_pipeline = build_pipeline(lr)
    lr_grid = (ParamGridBuilder()
               .addGrid(lr.regParam,        [0.01, 0.1])
               .addGrid(lr.elasticNetParam, [0.0,  0.5])
               .build())
    lr_cv = tune("LogisticRegression", lr_pipeline, lr_grid, train_df)
    lr_best = lr_cv.bestModel
    lr_preds, lr_roc, lr_pr = evaluate(lr_best, test_df, "LogisticRegression")
    
    # Random Forest
    rf = RandomForestClassifier(
        labelCol="label", featuresCol="features",
        weightCol="classWeight",
        seed=SEED,
    )
    rf_pipeline = build_pipeline(rf)
    rf_grid = (ParamGridBuilder()
               .addGrid(rf.numTrees, [20, 50])
               .addGrid(rf.maxDepth, [5,  10])
               .build())
    rf_cv = tune("RandomForestClassifier", rf_pipeline, rf_grid, train_df)
    rf_best = rf_cv.bestModel
    rf_preds, rf_roc, rf_pr = evaluate(rf_best, test_df, "RandomForestClassifier")

    # Save best models
    print(f"[INFO] Saving best LR -> {P_LR_MODEL}")
    lr_best.write().overwrite().save(P_LR_MODEL)
    print(f"[INFO] Saving best RF -> {P_RF_MODEL}")
    rf_best.write().overwrite().save(P_RF_MODEL)

    # Comparison CSV (Model, ROC-AUC, PR-AUC)
    comparison = spark.createDataFrame(
        [("LogisticRegression",     float(lr_roc), float(lr_pr)),
         ("RandomForestClassifier", float(rf_roc), float(rf_pr))],
        ["Model", "ROC_AUC", "PR_AUC"],
    )
    print("[INFO] Model comparison:")
    comparison.show(truncate=False)
    save_single_csv(comparison, P_COMPARISON)

    # Best-model predictions (label vs prediction)
    # Choose winner by PR-AUC, the right metric under heavy imbalance.
    if rf_pr >= lr_pr:
        best_name, best_preds = "RandomForestClassifier", rf_preds
    else:
        best_name, best_preds = "LogisticRegression", lr_preds
    print(f"[INFO] Best model by PR-AUC: {best_name}")
    save_single_csv(
        best_preds.select("label", "prediction"),
        P_PREDICTIONS,
    )

    print(f"[INFO] Exporting RF feature importances -> {P_RF_IMPORTANCES}")
    export_rf_importances(spark, rf_best, rf_preds, P_RF_IMPORTANCES)

    print(f"[INFO] Stage 3 complete in {time.time()-t_start:.1f}s.")
    spark.stop()


if __name__ == "__main__":
    main()