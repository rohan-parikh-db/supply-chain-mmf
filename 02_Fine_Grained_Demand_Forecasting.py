# Databricks notebook source
# MAGIC %md This project is based on Databricks' supply chain optimization solution accelerator available at: https://github.com/databricks-industry-solutions/supply-chain-optimization. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/supply-chain-distribution-optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine Grained Demand Forecasting (MMF)

# COMMAND ----------

# MAGIC %md
# MAGIC *Prerequisite: Make sure to run 01_Introduction_And_Setup before running this notebook.*
# MAGIC
# MAGIC In this notebook we execute a one-week-ahead forecast for each (product, wholesaler) series and then aggregate to the distribution center level. The forecasting is delegated to the
# MAGIC [Many Model Forecasting (MMF)](https://github.com/databricks-industry-solutions/many-model-forecasting) solution accelerator, which orchestrates per-series scoring, backtesting,
# MAGIC and MLflow tracking out of the box. For this run we use the **Chronos-2-Small** foundation model (zero-shot, no per-series training) on serverless GPU — much faster than fitting AutoARIMA/AutoETS across 900 series.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute requirement
# MAGIC
# MAGIC This notebook runs the Chronos-2-Small foundation model, which needs a GPU.
# MAGIC **Attach this notebook to serverless compute with `Accelerator = A10` and `Environment version = 5`** —
# MAGIC set these in the notebook's Configuration tab. CPU-only serverless will not work.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install MMF (local + foundation extras)

# COMMAND ----------

# MAGIC %pip install "mmf_sa[local,foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git" hf_transfer --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Create widgets for catalog and database names
dbutils.widgets.text("catalog_name", "serverless_aws_cvs_rev_int_catalog", "Catalog Name")
dbutils.widgets.text("db_name", "supply_chain_mmf_demo", "Database Name")

# COMMAND ----------

# Get values from widgets
catalog_name = dbutils.widgets.get("catalog_name")
db_name = dbutils.widgets.get("db_name")

print(f"Using catalog: {catalog_name}")
print(f"Using database: {db_name}")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalog_name $dbName=$db_name

# COMMAND ----------

print(catalogName)
print(dbName)

# COMMAND ----------

import logging
import os
import uuid

# Serverless GPU runtime sets HF_HUB_ENABLE_HF_TRANSFER=1, but the hf_transfer
# wheel isn't installed → HF model downloads fail. Disable before importing
# mmf_sa (which triggers chronos / transformers imports downstream).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import pyspark.sql.functions as f

from mmf_sa import run_forecast

# Quiet MLflow + py4j noise on serverless
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)
logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read historical demand and reshape to MMF long format
# MAGIC MMF expects a single string `group_id` column. We concatenate `product` and `wholesaler` so we keep the existing per-(product, wholesaler) granularity exactly — one MMF series per pair.

# COMMAND ----------

demand_df = spark.read.table(f"{catalogName}.{dbName}.product_demand_historical")
display(demand_df)

# COMMAND ----------

# NOTE: The source `date` column is Monday-anchored (one row per Mon-Mon week).
# MMF resamples internally using pandas freq="W", which is Sunday-anchored (W-SUN).
# To align, we shift each Monday timestamp forward by 6 days so it lands on the
# Sunday at the end of that week. This is purely a date-alignment trick — the
# `y` values stay the same. We shift forecast dates back in the post-process
# step below (well, we don't need to: the final aggregation drops `date`
# entirely, only `(distribution_center, product, demand)` makes it downstream).
mmf_train = (
    demand_df
    .withColumn("unique_id", f.concat_ws("||", "product", "wholesaler"))
    .selectExpr(
        "unique_id",
        "date_add(date, 6) as ds",
        "cast(demand as double) as y",
    )
)

mmf_train.write.mode("overwrite").saveAsTable(f"{catalogName}.{dbName}.mmf_train")
display(spark.read.table(f"{catalogName}.{dbName}.mmf_train").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run MMF — Chronos-2 family (Small + Base)
# MAGIC We fit two zero-shot foundation models from AWS AutoGluon's Chronos-2 family side-by-side and let MMF pick the winner per backtest:
# MAGIC
# MAGIC | Model | Params | Description |
# MAGIC |---|---|---|
# MAGIC | **Chronos-2-Small** | 28M | 28M-parameter time series forecasting foundation model that generates probabilistic predictions across univariate and universal forecasting tasks. |
# MAGIC | **Chronos-2** (base) | 120M | 120M-parameter encoder-only time series foundation model for zero-shot forecasting supporting univariate, multivariate, and covariate-informed tasks with quantile predictions. |
# MAGIC
# MAGIC Both fit comfortably in an A10's 24 GB VRAM. `accelerator="gpu"` + `serverless=True` switches Chronos to a driver-only predict path (per MMF's foundation-serverless example — Spark Connect Python workers are CPU-only on serverless, so distributed pandas_udf inference doesn't help).
# MAGIC
# MAGIC The single call writes per-model backtests into `mmf_evaluation` and per-model forward forecasts into `mmf_scoring`. The next cell picks the lower-SMAPE winner.

# COMMAND ----------

current_user = spark.sql("select current_user() as u").collect()[0]["u"]
shared_run_id = str(uuid.uuid4())
print(f"shared_run_id = {shared_run_id}")

run_forecast(
    spark=spark,
    train_data=f"{catalogName}.{dbName}.mmf_train",
    scoring_data=f"{catalogName}.{dbName}.mmf_train",
    scoring_output=f"{catalogName}.{dbName}.mmf_scoring",
    evaluation_output=f"{catalogName}.{dbName}.mmf_evaluation",
    model_output=f"{catalogName}.{dbName}",
    group_id="unique_id",
    date_col="ds",
    target="y",
    freq="W",
    prediction_length=1,
    backtest_length=4,
    stride=1,
    metric="smape",
    train_predict_ratio=1,
    data_quality_check=True,
    resample=True,
    active_models=["Chronos2Small", "Chronos2"],
    accelerator="gpu",
    serverless=True,
    experiment_path=f"/Users/{current_user}/mmf_supply_chain",
    use_case_name="supply_chain_demand",
    run_id=shared_run_id,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect backtest metrics

# COMMAND ----------

evaluation_df = (
    spark.read.table(f"{catalogName}.{dbName}.mmf_evaluation")
    .filter(f.col("run_id") == shared_run_id)
)
display(evaluation_df.limit(20))

# Per-model SMAPE — pick the lower one as the winner.
per_model = (
    evaluation_df.groupBy("model")
    .agg(f.avg("metric_value").alias("avg_smape"))
    .orderBy("avg_smape")
)
display(per_model)

rows = per_model.collect()
if not rows:
    best_model = "Chronos2Small"
    print(f"No backtest rows in mmf_evaluation for run_id={shared_run_id} — falling back to {best_model}")
else:
    for r in rows:
        print(f"  {r['model']:<20s} mean SMAPE = {r['avg_smape']:.4f}")
    best_model = rows[0]["model"]
    print(f"=> Winning model: {best_model} (SMAPE={rows[0]['avg_smape']:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reshape MMF scoring back into `(distribution_center, product, demand)`
# MAGIC MMF's `mmf_scoring` table writes one row per (unique_id, model) with `ds` and `y` as arrays (one entry per forecast step). We:
# MAGIC 1. filter to the winning model,
# MAGIC 2. explode the array so each forecast point becomes a row,
# MAGIC 3. split `unique_id` back into `product` and `wholesaler`,
# MAGIC 4. join the wholesaler→DC mapping and sum demand per (DC, product).
# MAGIC
# MAGIC The final table schema matches what downstream notebooks (`03_Derive_Raw_Material_Demand`) expect.

# COMMAND ----------

scoring_df = spark.read.table(f"{catalogName}.{dbName}.mmf_scoring")
display(scoring_df.limit(10))

# COMMAND ----------

per_wholesaler_forecast = (
    scoring_df
    .filter((f.col("model") == best_model) & (f.col("run_id") == shared_run_id))
    .withColumn("pair", f.arrays_zip("ds", "y"))
    .withColumn("pair", f.explode("pair"))
    .withColumn("parts", f.split("unique_id", "\\|\\|"))
    .select(
        f.col("parts")[0].alias("product"),
        f.col("parts")[1].alias("wholesaler"),
        f.col("pair.ds").cast("date").alias("date"),
        f.abs(f.col("pair.y")).cast("float").alias("demand"),
    )
)

# Integrity check: one forecast row per (product, wholesaler, step)
assert (
    demand_df.select("product", "wholesaler").distinct().count()
    == per_wholesaler_forecast.select("product", "wholesaler").distinct().count()
)

display(per_wholesaler_forecast)

# COMMAND ----------

distribution_center_to_wholesaler_mapping_table = spark.read.table(
    f"{catalogName}.{dbName}.distribution_center_to_wholesaler_mapping"
)
display(distribution_center_to_wholesaler_mapping_table)

# COMMAND ----------

distribution_center_demand = (
    per_wholesaler_forecast
    .join(distribution_center_to_wholesaler_mapping_table, on="wholesaler", how="left")
    .groupBy("distribution_center", "product")
    .agg(f.sum("demand").alias("demand"))
)

display(distribution_center_demand)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to delta

# COMMAND ----------

distribution_center_demand.write.mode("overwrite").saveAsTable(
    f"{catalogName}.{dbName}.product_demand_forecasted"
)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source]. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library | description | license | source |
# MAGIC |---|---|---|---|
# MAGIC | many-model-forecasting | MMF solution accelerator | Databricks License | https://github.com/databricks-industry-solutions/many-model-forecasting |
# MAGIC | statsforecast | Lightning fast time series forecasting | Apache 2.0 | https://github.com/Nixtla/statsforecast |
# MAGIC | pulp | A python Linear Programming API | https://github.com/coin-or/pulp/blob/master/LICENSE | https://github.com/coin-or/pulp |