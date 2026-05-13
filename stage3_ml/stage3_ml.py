"""Stage 3: Predictive Data Analytics on a Hadoop YARN cluster.

This module implements an end-to-end PySpark ML pipeline that predicts
whether a user session will result in a purchase, given the strongly
imbalanced e-commerce events dataset (~1.8% positive rate).

The pipeline:
    1. Reads from a partitioned Hive table backed by Parquet.
    2. Builds a binary target ('purchase' vs anything else).
    3. Decomposes the event timestamp into cyclical (sin, cos) features.
    4. Caps high-cardinality categoricals (brand, category_code) to the
       most frequent values learned from the training fold only.
    5. Runs a comparative balancing experiment that trains each of
       ``LogisticRegression`` and ``RandomForestClassifier`` under two
       strategies and evaluates both on the SAME original imbalanced
       test set:

           * Strategy A: dynamic class weighting.
           * Strategy B: random under-sampling to a 50/50 training set.

    6. Tunes every model with ``CrossValidator`` and
       ``ParamGridBuilder`` (>= 2 hyperparameters per algorithm).
    7. Reports six metrics per model: ROC-AUC, PR-AUC, accuracy, F1,
       precision and recall (the last three computed on the positive
       class, which is the meaningful one under heavy imbalance).
    8. Persists train/test splits as JSON, all four best PipelineModels,
       a model-comparison CSV, the best-model predictions CSV, and
       Random Forest feature importances CSV.
    9. Runs a price-surge risk simulation (+50% on test prices) and
       saves the resulting purchase-probability shift per record.
"""
import os
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_functions

from cyclical_encoder import CyclicalEncoder


# ----------------------------------------------------------------------
# Configuration (overridable via environment variables)
# ----------------------------------------------------------------------
HIVE_TABLE_NAME = os.environ.get(
    "HIVE_TABLE", "team9_projectdb.events_partitioned"
)
HDFS_OUTPUT_DIRECTORY = os.environ.get(
    "HDFS_OUT_DIR", "/user/team9/project/stage3"
)
SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "1.0"))

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
TEST_RATIO = 0.4

NUM_CV_FOLDS = 3
CV_PARALLELISM = 2

TOP_N_BRANDS = 100
TOP_N_CATEGORIES = 50

PRICE_SURGE_FACTOR = 1.5
RISK_SAMPLE_ROWS = 10000

POSITIVE_CLASS_LABEL = 1.0

# HDFS output paths
PATH_TRAIN_JSON = f"{HDFS_OUTPUT_DIRECTORY}/data/train"
PATH_TEST_JSON = f"{HDFS_OUTPUT_DIRECTORY}/data/test"
PATH_LR_WEIGHTED = f"{HDFS_OUTPUT_DIRECTORY}/models/lr_weighted"
PATH_RF_WEIGHTED = f"{HDFS_OUTPUT_DIRECTORY}/models/rf_weighted"
PATH_LR_DOWNSAMPLED = f"{HDFS_OUTPUT_DIRECTORY}/models/lr_downsampled"
PATH_RF_DOWNSAMPLED = f"{HDFS_OUTPUT_DIRECTORY}/models/rf_downsampled"
PATH_COMPARISON_CSV = (
    f"{HDFS_OUTPUT_DIRECTORY}/results/model_comparison_csv"
)
PATH_PREDICTIONS_CSV = (
    f"{HDFS_OUTPUT_DIRECTORY}/results/best_model_predictions_csv"
)
PATH_RF_IMPORTANCES_CSV = (
    f"{HDFS_OUTPUT_DIRECTORY}/results/rf_feature_importances_csv"
)
PATH_RISK_SIMULATION_CSV = (
    f"{HDFS_OUTPUT_DIRECTORY}/results/risk_simulation_csv"
)

MODEL_OUTPUT_PATH_MAP = {
    ("LogisticRegression", "Weighted"): PATH_LR_WEIGHTED,
    ("RandomForestClassifier", "Weighted"): PATH_RF_WEIGHTED,
    ("LogisticRegression", "Downsampled"): PATH_LR_DOWNSAMPLED,
    ("RandomForestClassifier", "Downsampled"): PATH_RF_DOWNSAMPLED,
}


# ======================================================================
# Spark session
# ======================================================================
def build_spark_session():
    """Construct a SparkSession with proper YARN and memory configuration.
    
    Returns:
        Configured SparkSession ready for production YARN deployment.
    """
    import os
    
    master = os.environ.get("SPARK_MASTER", "yarn")
    executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "2g")
    default_parallelism = int(os.environ.get("SPARK_DEFAULT_PARALLELISM", "200"))
    
    print(f"[INFO] Spark Configuration:")
    print(f"  Master: {master}")
    print(f"  Executor Memory: {executor_memory}")
    print(f"  Driver Memory: {driver_memory}")
    print(f"  Default Parallelism: {default_parallelism}")
    
    return (
        SparkSession.builder
        .master(master)
        .appName("Stage3_PurchasePrediction")
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(default_parallelism))
        .config("spark.default.parallelism", str(default_parallelism))
 	.config("hive.metastore.uris", "thrift://hadoop-02.uni.innopolis.ru:9883")
        .config("spark.sql.adaptive.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )


# ======================================================================
# Data loading and base feature engineering
# ======================================================================
def load_source_dataframe(spark_session):
    """Read the events table from the Hive metastore.

    Args:
        spark_session: Active SparkSession with Hive support.

    Returns:
        DataFrame backed by the partitioned Parquet Hive table.
    """
    print(f"[INFO] Reading Hive table: {HIVE_TABLE_NAME}")
    return spark_session.table(HIVE_TABLE_NAME)


def prepare_base_features(source_dataframe):
    """Add the binary label, decompose the timestamp, and fill nulls.

    Args:
        source_dataframe: Raw events DataFrame.

    Returns:
        DataFrame with ``label``, ``hour``, ``dayOfWeek``, ``price``,
        ``brand``, ``category_code``.
    """
    return (
        source_dataframe
        .withColumn(
            "label",
            (spark_functions.col("event_type") == "purchase").cast("int"),
        )
        .withColumn("hour", spark_functions.hour("event_time"))
        .withColumn(
            "dayOfWeek",
            spark_functions.dayofweek("event_time") - 1,
        )
        .fillna({"brand": "unknown", "category_code": "unknown"})
        .fillna({"price": 0.0})
        .select(
            "label", "hour", "dayOfWeek",
            "price", "brand", "category_code",
        )
    )


def apply_stratified_sample(prepared_dataframe, sample_fraction):
    """Take a stratified sample that preserves the class ratio.

    Args:
        prepared_dataframe: DataFrame containing the ``label`` column.
        sample_fraction: Fraction of rows to sample from each class.
            Returns the input unchanged if ``sample_fraction >= 1.0``.

    Returns:
        Sampled (or original) DataFrame.
    """
    if sample_fraction >= 1.0:
        return prepared_dataframe

    print(f"[INFO] Stratified sample at fraction={sample_fraction}")
    return prepared_dataframe.stat.sampleBy(
        "label",
        fractions={0: sample_fraction, 1: sample_fraction},
        seed=RANDOM_SEED,
    )


def split_train_test(prepared_dataframe):
    """Split into 60% training and 40% test DataFrames.

    Args:
        prepared_dataframe: Prepared input data.

    Returns:
        Tuple ``(training_dataframe, test_dataframe)``.
    """
    training_dataframe, test_dataframe = prepared_dataframe.randomSplit(
        [TRAIN_RATIO, TEST_RATIO], seed=RANDOM_SEED
    )
    return training_dataframe, test_dataframe


def collect_top_n_values(dataframe, column_name, top_n):
    """Return the ``top_n`` most frequent values of a column.

    Args:
        dataframe: Input DataFrame.
        column_name: Categorical column whose top values we want.
        top_n: How many top values to keep.

    Returns:
        Python list of the most frequent values.
    """
    grouped_rows = (
        dataframe.groupBy(column_name)
        .count()
        .orderBy(spark_functions.desc("count"))
        .limit(top_n)
        .collect()
    )
    return [row[column_name] for row in grouped_rows]


def apply_cardinality_capping(
    dataframe, column_name, kept_values, suffix="_capped"
):
    """Replace values outside ``kept_values`` with the literal ``other``.

    Args:
        dataframe: Input DataFrame.
        column_name: Source categorical column.
        kept_values: Allowed values; all others become ``other``.
        suffix: Suffix appended to ``column_name`` for the output.

    Returns:
        DataFrame with a new column ``<column_name><suffix>``.
    """
    output_column_name = f"{column_name}{suffix}"
    return dataframe.withColumn(
        output_column_name,
        spark_functions.when(
            spark_functions.col(column_name).isin(kept_values),
            spark_functions.col(column_name),
        ).otherwise(spark_functions.lit("other")),
    )


# ======================================================================
# Class weighting and down-sampling
# ======================================================================
def compute_class_weights(training_dataframe):
    """Compute sklearn-style balanced class weights from training data.

    Uses the formula ``w_c = N / (K * N_c)`` with ``K = 2``.

    Args:
        training_dataframe: Training data with the ``label`` column.

    Returns:
        Tuple ``(weight_negative, weight_positive)``.
    """
    class_counts = {
        row["label"]: row["count"]
        for row in training_dataframe.groupBy("label").count().collect()
    }
    negative_count = class_counts.get(0, 0)
    positive_count = class_counts.get(1, 0)
    total_count = negative_count + positive_count
    number_of_classes = 2

    weight_negative = (
        total_count / (number_of_classes * negative_count)
        if negative_count else 1.0
    )
    weight_positive = (
        total_count / (number_of_classes * positive_count)
        if positive_count else 1.0
    )

    positive_rate_percent = (
        100.0 * positive_count / total_count if total_count else 0.0
    )
    print(
        f"[INFO] Train class counts: negative={negative_count:,}  "
        f"positive={positive_count:,}  "
        f"(positive rate = {positive_rate_percent:.2f}%)"
    )
    print(
        f"[INFO] Class weights: w_negative={weight_negative:.4f}  "
        f"w_positive={weight_positive:.4f}"
    )
    return weight_negative, weight_positive


def add_class_weight_column(dataframe, weight_negative, weight_positive):
    """Append a ``classWeight`` column to a DataFrame.

    Args:
        dataframe: Source DataFrame containing the ``label`` column.
        weight_negative: Weight applied to negative (label=0) rows.
        weight_positive: Weight applied to positive (label=1) rows.

    Returns:
        DataFrame with an additional ``classWeight`` column.
    """
    weight_expression = spark_functions.when(
        spark_functions.col("label") == 1, weight_positive
    ).otherwise(weight_negative)
    return dataframe.withColumn("classWeight", weight_expression)


def create_downsampled_training_set(training_dataframe):
    """Build a 50/50 balanced training set by random under-sampling.

    Args:
        training_dataframe: Original imbalanced training data.

    Returns:
        DataFrame containing all positive rows plus a random subset of
        negative rows of approximately equal size.
    """
    positive_dataframe = training_dataframe.filter(
        spark_functions.col("label") == 1
    )
    negative_dataframe = training_dataframe.filter(
        spark_functions.col("label") == 0
    )
    positive_count = positive_dataframe.count()
    negative_count = negative_dataframe.count()
    sampling_fraction = (
        positive_count / negative_count if negative_count else 0.0
    )
    downsampled_negative_dataframe = negative_dataframe.sample(
        withReplacement=False,
        fraction=sampling_fraction,
        seed=RANDOM_SEED,
    )
    balanced_training_dataframe = positive_dataframe.union(
        downsampled_negative_dataframe
    )
    print(
        f"[INFO] Downsampled training set: "
        f"positives kept={positive_count:,}  "
        f"negatives kept~={sampling_fraction * negative_count:,.0f}"
    )
    return balanced_training_dataframe


# ======================================================================
# Pipeline construction
# ======================================================================
def build_preprocessing_stages():
    """Construct the shared list of preprocessing pipeline stages.

    Returns:
        Ordered list of ``pyspark.ml`` stages applied before the
        classifier.
    """
    hour_encoder = CyclicalEncoder(inputCol="hour", period=24.0)
    day_of_week_encoder = CyclicalEncoder(inputCol="dayOfWeek", period=7.0)

    brand_indexer = StringIndexer(
        inputCol="brand_capped",
        outputCol="brand_idx",
        handleInvalid="keep",
    )
    category_indexer = StringIndexer(
        inputCol="category_code_capped",
        outputCol="cat_idx",
        handleInvalid="keep",
    )
    brand_one_hot = OneHotEncoder(
        inputCol="brand_idx",
        outputCol="brand_vec",
        handleInvalid="keep",
    )
    category_one_hot = OneHotEncoder(
        inputCol="cat_idx",
        outputCol="cat_vec",
        handleInvalid="keep",
    )

    vector_assembler = VectorAssembler(
        inputCols=[
            "hour_sin", "hour_cos",
            "dayOfWeek_sin", "dayOfWeek_cos",
            "price",
            "brand_vec", "cat_vec",
        ],
        outputCol="features_raw",
        handleInvalid="skip",
    )
    standard_scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=False,
        withStd=True,
    )
    return [
        hour_encoder, day_of_week_encoder,
        brand_indexer, category_indexer,
        brand_one_hot, category_one_hot,
        vector_assembler, standard_scaler,
    ]


def build_logistic_regression(use_class_weights):
    """Construct a LogisticRegression classifier.

    Args:
        use_class_weights: If True, configures the classifier to use
            the ``classWeight`` column for sample weighting.

    Returns:
        Configured ``LogisticRegression`` estimator.
    """
    constructor_kwargs = {
        "labelCol": "label",
        "featuresCol": "features",
        "maxIter": 30,
    }
    if use_class_weights:
        constructor_kwargs["weightCol"] = "classWeight"
    return LogisticRegression(**constructor_kwargs)


def build_random_forest(use_class_weights):
    """Construct a RandomForestClassifier classifier.

    Args:
        use_class_weights: If True, configures the classifier to use
            the ``classWeight`` column for sample weighting.

    Returns:
        Configured ``RandomForestClassifier`` estimator.
    """
    constructor_kwargs = {
        "labelCol": "label",
        "featuresCol": "features",
        "seed": RANDOM_SEED,
    }
    if use_class_weights:
        constructor_kwargs["weightCol"] = "classWeight"
    return RandomForestClassifier(**constructor_kwargs)


def build_ml_pipeline(classifier):
    """Compose preprocessing stages with the given classifier.

    Args:
        classifier: Final estimator stage (LR or RF).

    Returns:
        ``pyspark.ml.Pipeline`` with preprocessing followed by the
        classifier.
    """
    pipeline_stages = build_preprocessing_stages() + [classifier]
    return Pipeline(stages=pipeline_stages)


def build_logistic_param_grid(logistic_regression):
    """Build the ParamGrid for LogisticRegression tuning.

    Tunes ``regParam`` and ``elasticNetParam``.

    Args:
        logistic_regression: ``LogisticRegression`` instance whose
            parameters are referenced in the grid.

    Returns:
        List of parameter maps.
    """
    return (
        ParamGridBuilder()
        .addGrid(logistic_regression.regParam, [0.01, 0.1])
        .addGrid(logistic_regression.elasticNetParam, [0.0, 0.5])
        .build()
    )


def build_random_forest_param_grid(random_forest):
    """Build the ParamGrid for RandomForestClassifier tuning.

    Tunes ``numTrees`` and ``maxDepth``.

    Args:
        random_forest: ``RandomForestClassifier`` instance whose
            parameters are referenced in the grid.

    Returns:
        List of parameter maps.
    """
    return (
        ParamGridBuilder()
        .addGrid(random_forest.numTrees, [20, 50])
        .addGrid(random_forest.maxDepth, [5, 10])
        .build()
    )


# ======================================================================
# Tuning and evaluation
# ======================================================================
def tune_with_crossvalidator(ml_pipeline, param_grid, training_dataframe):
    """Run k-fold cross-validated hyperparameter tuning.

    Args:
        ml_pipeline: Pipeline to be tuned.
        param_grid: List of parameter maps to evaluate.
        training_dataframe: Training data.

    Returns:
        Fitted ``CrossValidatorModel`` containing the best
        ``PipelineModel``.
    """
    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    cross_validator = CrossValidator(
        estimator=ml_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=roc_evaluator,
        numFolds=NUM_CV_FOLDS,
        seed=RANDOM_SEED,
        parallelism=CV_PARALLELISM,
    )
    return cross_validator.fit(training_dataframe)


def evaluate_positive_class_metric(predictions_dataframe, metric_name):
    """Evaluate a per-class multiclass metric for the positive class.

    Args:
        predictions_dataframe: DataFrame containing predictions.
        metric_name: One of ``precisionByLabel``, ``recallByLabel``,
            ``fMeasureByLabel``.

    Returns:
        Metric value as float.
    """
    return MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric_name,
        metricLabel=POSITIVE_CLASS_LABEL,
    ).evaluate(predictions_dataframe)

def compute_evaluation_metrics(trained_model, test_dataframe):
    """Compute the full metric suite for a trained model.

    Args:
        trained_model: Fitted ``PipelineModel``.
        test_dataframe: Test set (original imbalanced distribution).

    Returns:
        Tuple ``(predictions_dataframe, metrics_dict)`` where
        ``metrics_dict`` has keys ``roc_auc``, ``pr_auc``, ``accuracy``,
        ``f1_score``, ``precision``, ``recall``.
    """
    predictions_dataframe = trained_model.transform(test_dataframe).cache()
    
    try:
        roc_auc_score = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        ).evaluate(predictions_dataframe)
        
        pr_auc_score = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR",
        ).evaluate(predictions_dataframe)
        
        accuracy_score = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy",
        ).evaluate(predictions_dataframe)
        
        f1_score_value = evaluate_positive_class_metric(
            predictions_dataframe, "fMeasureByLabel"
        )
        
        precision_score = evaluate_positive_class_metric(
            predictions_dataframe, "precisionByLabel"
        )
        
        recall_score = evaluate_positive_class_metric(
            predictions_dataframe, "recallByLabel"
        )
        
        metrics_dict = {
            "roc_auc": roc_auc_score,
            "pr_auc": pr_auc_score,
            "accuracy": accuracy_score,
            "f1_score": f1_score_value,
            "precision": precision_score,
            "recall": recall_score,
        }
        
        return predictions_dataframe, metrics_dict
    
    finally:
        predictions_dataframe.unpersist(blocking=False)





def train_single_experiment(experiment_specification):
    """Train one (algorithm, strategy) combination end-to-end.

    Args:
        experiment_specification: Dict with keys ``strategy_name``,
            ``algorithm_name``, ``classifier``, ``param_grid``,
            ``training_dataframe``, ``test_dataframe``.

    Returns:
        Dict with keys ``algorithm``, ``strategy``, ``best_model``,
        ``metrics``, ``predictions``.
    """
    strategy_name = experiment_specification["strategy_name"]
    algorithm_name = experiment_specification["algorithm_name"]
    classifier = experiment_specification["classifier"]
    param_grid = experiment_specification["param_grid"]
    training_dataframe = experiment_specification["training_dataframe"]
    test_dataframe = experiment_specification["test_dataframe"]

    print(
        f"[INFO] Training {algorithm_name} with strategy "
        f"'{strategy_name}' ..."
    )
    ml_pipeline = build_ml_pipeline(classifier)
    start_time = time.time()
    crossvalidator_model = tune_with_crossvalidator(
        ml_pipeline, param_grid, training_dataframe
    )
    elapsed_seconds = time.time() - start_time
    best_pipeline_model = crossvalidator_model.bestModel
    print(
        f"[INFO] {algorithm_name} ({strategy_name}) CV done in "
        f"{elapsed_seconds:.1f}s; best avg ROC-AUC = "
        f"{max(crossvalidator_model.avgMetrics):.4f}"
    )

    predictions_dataframe, metrics_dict = compute_evaluation_metrics(
        best_pipeline_model, test_dataframe
    )
    print(
        f"[INFO] {algorithm_name} ({strategy_name})  "
        f"ROC-AUC={metrics_dict['roc_auc']:.4f}  "
        f"PR-AUC={metrics_dict['pr_auc']:.4f}  "
        f"F1={metrics_dict['f1_score']:.4f}  "
        f"Precision={metrics_dict['precision']:.4f}  "
        f"Recall={metrics_dict['recall']:.4f}"
    )

    return {
        "algorithm": algorithm_name,
        "strategy": strategy_name,
        "best_model": best_pipeline_model,
        "metrics": metrics_dict,
        "predictions": predictions_dataframe,
    }


# ======================================================================
# Persistence helpers
# ======================================================================
def persist_splits_as_json(training_dataframe, test_dataframe):
    """Write train and test splits to HDFS as JSON (overwrite).

    Args:
        training_dataframe: Training split.
        test_dataframe: Test split.
    """
    print("[INFO] Writing train/test JSON to HDFS ...")
    training_dataframe.write.mode("overwrite").json(PATH_TRAIN_JSON)
    test_dataframe.write.mode("overwrite").json(PATH_TEST_JSON)


def save_models_to_hdfs(experiment_results):
    """Persist every best PipelineModel to HDFS.

    Args:
        experiment_results: List of dicts returned by
            :func:`train_single_experiment`.
    """
    for experiment_result in experiment_results:
        model_key = (
            experiment_result["algorithm"],
            experiment_result["strategy"],
        )
        output_path = MODEL_OUTPUT_PATH_MAP[model_key]
        print(
            f"[INFO] Saving {model_key[0]} ({model_key[1]}) "
            f"-> {output_path}"
        )
        experiment_result["best_model"].write().overwrite().save(
            output_path
        )


def save_single_csv(dataframe, output_path):
    """Write a small DataFrame as a single CSV file (idempotent).

    Args:
        dataframe: DataFrame to write.
        output_path: HDFS path of the output directory.
    """
    (
        dataframe.coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(output_path)
    )


# ======================================================================
# Reporting and interpretation
# ======================================================================
def create_comparison_dataframe(spark_session, experiment_results):
    """Build the model-comparison DataFrame for export.

    Args:
        spark_session: Active SparkSession.
        experiment_results: List of experiment result dicts.

    Returns:
        DataFrame with one row per (algorithm, strategy) and all six
        metric columns.
    """
    comparison_rows = []
    for experiment_result in experiment_results:
        metrics = experiment_result["metrics"]
        comparison_rows.append((
            experiment_result["algorithm"],
            experiment_result["strategy"],
            float(metrics["roc_auc"]),
            float(metrics["pr_auc"]),
            float(metrics["accuracy"]),
            float(metrics["f1_score"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
        ))
    schema_columns = [
        "Algorithm", "Strategy",
        "ROC_AUC", "PR_AUC", "Accuracy",
        "F1_Score", "Precision", "Recall",
    ]
    return spark_session.createDataFrame(comparison_rows, schema_columns)


def _pr_auc_key(experiment_result):
    """Return the PR-AUC value from an experiment result dict.

    Args:
        experiment_result: One experiment result dict.

    Returns:
        Float PR-AUC score.
    """
    return experiment_result["metrics"]["pr_auc"]


def select_best_experiment_result(experiment_results):
    """Pick the best experiment result by PR-AUC.

    PR-AUC is preferred to ROC-AUC under heavy class imbalance because
    it focuses on the positive class.

    Args:
        experiment_results: List of experiment result dicts.

    Returns:
        The single best experiment result dict.
    """
    best_result = max(experiment_results, key=_pr_auc_key)
    print(
        f"[INFO] Best overall model: {best_result['algorithm']} "
        f"({best_result['strategy']})  "
        f"PR-AUC={best_result['metrics']['pr_auc']:.4f}"
    )
    return best_result


def select_best_random_forest_result(experiment_results):
    """Pick the best RandomForestClassifier result by PR-AUC.

    Args:
        experiment_results: List of experiment result dicts.

    Returns:
        Best RF result dict, or ``None`` if no RF was trained.
    """
    rf_results = [
        result for result in experiment_results
        if result["algorithm"] == "RandomForestClassifier"
    ]
    if not rf_results:
        return None
    return max(rf_results, key=_pr_auc_key)


def extract_feature_names_from_metadata(
    dataframe, column_candidates=("features_raw", "features"),
):
    """Read the ordered list of feature names from a vector column.

    Args:
        dataframe: DataFrame that has gone through the pipeline.
        column_candidates: Ordered tuple of vector columns to inspect.

    Returns:
        Ordered list of feature names, or ``None`` if no metadata was
        found.
    """
    for column_name in column_candidates:
        if column_name not in dataframe.columns:
            continue
        column_metadata = dataframe.schema[column_name].metadata
        ml_attributes = (
            column_metadata.get("ml_attr") if column_metadata else None
        )
        if not ml_attributes or "attrs" not in ml_attributes:
            continue
        indexed_attributes = []
        for attribute_list in ml_attributes["attrs"].values():
            for attribute in attribute_list:
                indexed_attributes.append(
                    (attribute["idx"], attribute["name"])
                )
        if indexed_attributes:
            indexed_attributes.sort()
            return [name for _, name in indexed_attributes]
    return None


def export_feature_importances(
    spark_session, rf_pipeline_model, transformed_dataframe, output_path,
):
    """Save Random Forest feature importances to a single CSV.

    Args:
        spark_session: Active SparkSession.
        rf_pipeline_model: Fitted Pipeline whose last stage is a
            ``RandomForestClassificationModel``.
        transformed_dataframe: A DataFrame transformed by the pipeline,
            from which the ML attribute metadata is read.
        output_path: HDFS path for the CSV directory.
    """
    random_forest_model = rf_pipeline_model.stages[-1]
    importance_vector = random_forest_model.featureImportances
    feature_names = extract_feature_names_from_metadata(
        transformed_dataframe
    )

    if feature_names and len(feature_names) == importance_vector.size:
        importance_rows = [
            (feature_names[index], float(importance_vector[index]))
            for index in range(importance_vector.size)
        ]
    else:
        print(
            "[WARN] Feature name metadata unavailable or mismatched; "
            "falling back to indexed names."
        )
        importance_rows = [
            (f"feature_{index}", float(importance_vector[index]))
            for index in range(importance_vector.size)
        ]
    importance_rows.sort(key=lambda row: row[1], reverse=True)
    importances_dataframe = spark_session.createDataFrame(
        importance_rows, ["feature", "importance"]
    )
    save_single_csv(importances_dataframe, output_path)


# ======================================================================
# Risk simulation
# ======================================================================
def simulate_price_surge_risk(
    best_pipeline_model, test_dataframe, surge_factor,
):
    """Re-score the test set with prices multiplied by ``surge_factor``.

    For each test row we compute the original purchase probability and
    the probability after price surge, plus the delta. This answers:
    if prices increased by 50%, how does predicted demand change?

    Args:
        best_pipeline_model: PipelineModel from the best experiment.
        test_dataframe: Prepared test data (post-cardinality capping
            and class-weight column, but pre-transform).
        surge_factor: Multiplicative factor applied to ``price``
            (e.g. ``1.5`` for +50%).

    Returns:
        DataFrame with columns ``original_price``, ``surged_price``,
        ``original_purchase_probability``,
        ``surged_purchase_probability``, ``original_prediction``,
        ``surged_prediction``, ``probability_delta``.
    """
    print(
        f"[INFO] Running price-surge risk simulation "
        f"(factor={surge_factor}) ..."
    )

    test_with_row_id = test_dataframe.withColumn(
        "simulation_row_id",
        spark_functions.monotonically_increasing_id(),
    ).cache()
    
    surged_test_dataframe = test_with_row_id.withColumn(
        "price",
        spark_functions.col("price") * spark_functions.lit(surge_factor),
    )
    
    original_predictions = (
        best_pipeline_model.transform(test_with_row_id)
        .withColumn(
            "positive_class_probability",
            vector_to_array("probability")[1],
        )
        .select(
            "simulation_row_id",
            spark_functions.col("price").alias("original_price"),
            spark_functions.col("positive_class_probability").alias(
                "original_purchase_probability"
            ),
            spark_functions.col("prediction").alias(
                "original_prediction"
            ),
        )
        .cache()
    )

    surged_predictions = (
        best_pipeline_model.transform(surged_test_dataframe)
        .withColumn(
            "positive_class_probability",
            vector_to_array("probability")[1],
        )
        .select(
            "simulation_row_id",
            spark_functions.col("price").alias("surged_price"),
            spark_functions.col("positive_class_probability").alias(
                "surged_purchase_probability"
            ),
            spark_functions.col("prediction").alias(
                "surged_prediction"
            ),
        )
        .cache()
    )

    risk_dataframe = (
        original_predictions.join(
            surged_predictions, on="simulation_row_id", how="inner"
        )
        .withColumn(
            "probability_delta",
            spark_functions.col("surged_purchase_probability")
            - spark_functions.col("original_purchase_probability"),
        )
        .drop("simulation_row_id")
    )
    original_predictions.unpersist(blocking=False)
    surged_predictions.unpersist(blocking=False)
    test_with_row_id.unpersist(blocking=False)
    return risk_dataframe


# ======================================================================
# Orchestration
# ======================================================================
def prepare_data_for_modeling(spark_session):
    """Run all data-prep steps and return the cached splits.

    Args:
        spark_session: Active SparkSession.

    Returns:
        Tuple ``(training_dataframe, test_dataframe)`` ready for
        modelling. Both contain ``brand_capped``,
        ``category_code_capped``, and ``classWeight`` columns.
    """
    raw_dataframe = load_source_dataframe(spark_session)
    prepared_dataframe = prepare_base_features(raw_dataframe)
    
    sampled_dataframe = apply_stratified_sample(
        prepared_dataframe, SAMPLE_FRACTION
    ).repartition(200).cache()
    
    total_row_count = sampled_dataframe.count()
    positive_row_count = sampled_dataframe.filter(
        spark_functions.col("label") == 1
    ).count()
    positive_rate_percent = (
        100.0 * positive_row_count / total_row_count
        if total_row_count else 0.0
    )
    print(
        f"[INFO] After load+sample: total={total_row_count:,}  "
        f"positive={positive_row_count:,} "
        f"({positive_rate_percent:.2f}%)"
    )
    
    training_dataframe, test_dataframe = split_train_test(
        sampled_dataframe
    )
    
    top_brand_values = collect_top_n_values(
        training_dataframe, "brand", TOP_N_BRANDS
    )
    top_category_values = collect_top_n_values(
        training_dataframe, "category_code", TOP_N_CATEGORIES
    )
    print(
        f"[INFO] Capping brand to top-{TOP_N_BRANDS} and "
        f"category_code to top-{TOP_N_CATEGORIES} (from training only)"
    )
    
    training_dataframe = apply_cardinality_capping(
        training_dataframe, "brand", top_brand_values
    )
    training_dataframe = apply_cardinality_capping(
        training_dataframe, "category_code", top_category_values
    )
    test_dataframe = apply_cardinality_capping(
        test_dataframe, "brand", top_brand_values
    )
    test_dataframe = apply_cardinality_capping(
        test_dataframe, "category_code", top_category_values
    )
    
    weight_negative, weight_positive = compute_class_weights(
        training_dataframe
    )
    
    training_dataframe = add_class_weight_column(
        training_dataframe, weight_negative, weight_positive
    ).repartition(200).cache()
    
    print(
        f"[INFO] Train rows: {training_dataframe.count():,}   "
        f"Test rows: {test_dataframe.count():,}"
    )
    
    persist_splits_as_json(training_dataframe, test_dataframe)
    return training_dataframe, test_dataframe


def _build_experiment_specifications(
    training_dataframe, downsampled_training_dataframe, test_dataframe,
):
    """Enumerate the four (algorithm, strategy) experiment dicts.

    Args:
        training_dataframe: Original imbalanced training set with
            ``classWeight`` column.
        downsampled_training_dataframe: 50/50 balanced training set.
        test_dataframe: Original imbalanced test set.

    Returns:
        List of four experiment specification dicts ready to be
        consumed by :func:`train_single_experiment`.
    """
    logistic_weighted = build_logistic_regression(use_class_weights=True)
    forest_weighted = build_random_forest(use_class_weights=True)
    logistic_downsampled = build_logistic_regression(
        use_class_weights=False
    )
    forest_downsampled = build_random_forest(use_class_weights=False)

    return [
        {
            "strategy_name": "Weighted",
            "algorithm_name": "LogisticRegression",
            "classifier": logistic_weighted,
            "param_grid": build_logistic_param_grid(logistic_weighted),
            "training_dataframe": training_dataframe,
            "test_dataframe": test_dataframe,
        },
        {
            "strategy_name": "Weighted",
            "algorithm_name": "RandomForestClassifier",
            "classifier": forest_weighted,
            "param_grid": build_random_forest_param_grid(forest_weighted),
            "training_dataframe": training_dataframe,
            "test_dataframe": test_dataframe,
        },
        {
            "strategy_name": "Downsampled",
            "algorithm_name": "LogisticRegression",
            "classifier": logistic_downsampled,
            "param_grid": build_logistic_param_grid(logistic_downsampled),
            "training_dataframe": downsampled_training_dataframe,
            "test_dataframe": test_dataframe,
        },
        {
            "strategy_name": "Downsampled",
            "algorithm_name": "RandomForestClassifier",
            "classifier": forest_downsampled,
            "param_grid": build_random_forest_param_grid(
                forest_downsampled
            ),
            "training_dataframe": downsampled_training_dataframe,
            "test_dataframe": test_dataframe,
        },
    ]


def run_all_experiments(training_dataframe, test_dataframe):
    """Train both algorithms under both balancing strategies.

    Args:
        training_dataframe: Original (imbalanced) training set with
            ``classWeight`` column.
        test_dataframe: Original (imbalanced) test set.

    Returns:
        List of four experiment result dicts.
    """
    downsampled_training_dataframe = create_downsampled_training_set(
        training_dataframe
    ).cache()

    experiment_specifications = _build_experiment_specifications(
        training_dataframe,
        downsampled_training_dataframe,
        test_dataframe,
    )

    experiment_results = []
    for specification in experiment_specifications:
        experiment_results.append(
            train_single_experiment(specification)
        )
    return experiment_results


def save_all_outputs(spark_session, experiment_results, test_dataframe):
    """Persist comparison CSV, predictions, importances, risk sim.

    Args:
        spark_session: Active SparkSession.
        experiment_results: List of experiment result dicts.
        test_dataframe: Original imbalanced test set used by the risk
            simulation.
    """
    save_models_to_hdfs(experiment_results)

    comparison_dataframe = create_comparison_dataframe(
        spark_session, experiment_results
    )
    print("[INFO] Model comparison:")
    comparison_dataframe.show(truncate=False)
    save_single_csv(comparison_dataframe, PATH_COMPARISON_CSV)

    best_result = select_best_experiment_result(experiment_results)
    save_single_csv(
        best_result["predictions"].select("label", "prediction"),
        PATH_PREDICTIONS_CSV,
    )

    best_random_forest_result = select_best_random_forest_result(
        experiment_results
    )
    if best_random_forest_result is not None:
        print(
            f"[INFO] Exporting feature importances from "
            f"{best_random_forest_result['algorithm']} "
            f"({best_random_forest_result['strategy']}) ..."
        )
        export_feature_importances(
            spark_session,
            best_random_forest_result["best_model"],
            best_random_forest_result["predictions"],
            PATH_RF_IMPORTANCES_CSV,
        )

    risk_dataframe = simulate_price_surge_risk(
        best_result["best_model"],
        test_dataframe,
        PRICE_SURGE_FACTOR,
    )
    save_single_csv(
        risk_dataframe.limit(RISK_SAMPLE_ROWS),
        PATH_RISK_SIMULATION_CSV,
    )


def main():
    """Stage 3 entry point: orchestrate data prep, training, outputs."""
    spark_session = build_spark_session()
    spark_session.sparkContext.setLogLevel("WARN")
    start_time = time.time()

    training_dataframe, test_dataframe = prepare_data_for_modeling(
        spark_session
    )
    experiment_results = run_all_experiments(
        training_dataframe, test_dataframe
    )
    save_all_outputs(
        spark_session, experiment_results, test_dataframe
    )

    elapsed_total = time.time() - start_time
    training_dataframe.unpersist()
    test_dataframe.unpersist()
    
    print(f"[INFO] Stage 3 complete in {elapsed_total:.1f}s.")
    spark_session.stop()


if __name__ == "__main__":
    main()
