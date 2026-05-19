# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Fine-Grained Demand Forecasting with Chronos-2
# MAGIC
# MAGIC > **Prerequisite:** run `01_Introduction_And_Setup` first to seed the source tables.
# MAGIC
# MAGIC This notebook produces a one-week-ahead demand forecast for **every (product, wholesaler) pair** — 900 time series in our synthetic dataset — and then aggregates the results back to the **distribution-center level** that downstream notebooks consume.
# MAGIC
# MAGIC The work is delegated to two open-source pieces:
# MAGIC
# MAGIC 1. **[Many Model Forecasting (MMF)](https://github.com/databricks-industry-solutions/many-model-forecasting)** — a Databricks solution accelerator that wraps per-series scoring, rolling backtests, MLflow logging, and UC model registration into a single `run_forecast(...)` call.
# MAGIC 2. **[Chronos-2 foundation models](https://huggingface.co/autogluon)** from AWS AutoGluon — pretrained time-series transformers that produce probabilistic forecasts **zero-shot** (no per-series training). We fit two variants side-by-side and pick the winner by mean SMAPE.
# MAGIC
# MAGIC ### Why foundation models for demand forecasting?
# MAGIC
# MAGIC The legacy pattern in this dataset would fit 900 separate AutoARIMA / Holt-Winters / ExponentialSmoothing models — one per series — typically distributed via a `pandas_udf`. That works, but on serverless it has two friction points: Spark Connect's Python workers are CPU-only (so you don't benefit from the GPUs that *do* exist on serverless GPU compute), and per-series fitting time scales linearly with series count.
# MAGIC
# MAGIC A zero-shot foundation model flips that: **one model, batched inference, one GPU**. On 900 weekly series, this notebook runs end-to-end in ~30 seconds and hits ~10–11% mean SMAPE — competitive with tuned per-series classical models that take 30–60× more compute.
# MAGIC
# MAGIC ### Compute requirement
# MAGIC
# MAGIC Attach this notebook to **serverless compute** with **`Accelerator = A10`** and **`Environment version = 5`** via the notebook's *Configuration* tab. CPU-only serverless will not work — Chronos needs a GPU.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC
# MAGIC `mmf_sa[foundation]` pulls in Chronos and the HuggingFace stack. We also pin `hf_transfer` so model downloads work whether or not the runtime pre-sets `HF_HUB_ENABLE_HF_TRANSFER=1`.

# COMMAND ----------

# MAGIC %pip install "mmf_sa[local,foundation] @ git+https://github.com/databricks-industry-solutions/many-model-forecasting.git" hf_transfer --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog_name", "main", "Catalog Name")
dbutils.widgets.text("db_name", "supply_chain_mmf", "Database Name")

catalog_name = dbutils.widgets.get("catalog_name")
db_name = dbutils.widgets.get("db_name")

print(f"Using catalog: {catalog_name}")
print(f"Using database: {db_name}")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalog_name $dbName=$db_name

# COMMAND ----------

import logging
import os
import uuid

# Belt-and-suspenders: also disable HF_TRANSFER at runtime in case the
# environment ships it enabled but without the wheel.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import pyspark.sql.functions as f

from mmf_sa import run_forecast

# Quiet MLflow + py4j noise on serverless
logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)
logging.getLogger("py4j.clientserver").setLevel(logging.WARNING)
logging.getLogger("py4j.java_gateway").setLevel(logging.WARNING)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reshape historical demand into MMF's long format
# MAGIC
# MAGIC MMF expects a long-format frame with three columns: a string `group_id`, a date column, and a numeric target. We concatenate `product` and `wholesaler` to form one MMF series per pair (preserving the original granularity), then write it to `mmf_train`.
# MAGIC
# MAGIC > **Date-alignment note.** The source `date` column is Monday-anchored. MMF resamples internally with pandas `freq="W"`, which is Sunday-anchored (`W-SUN`). We shift each timestamp forward 6 days so it lands on Sunday — purely a label fix, the `y` values are unchanged.

# COMMAND ----------

demand_df = spark.read.table(f"{catalogName}.{dbName}.product_demand_historical")

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
# MAGIC
# MAGIC We fit two Chronos-2 variants and let MMF pick the lower-SMAPE winner:
# MAGIC
# MAGIC | Model | Params | Description |
# MAGIC |---|---|---|
# MAGIC | **Chronos-2-Small** | 28M | Time-series foundation model that produces probabilistic predictions across univariate and universal forecasting tasks. |
# MAGIC | **Chronos-2** (base) | 120M | Encoder-only foundation model for zero-shot forecasting with quantile predictions; supports univariate, multivariate, and covariate-informed tasks. |
# MAGIC
# MAGIC Both fit comfortably in an A10's 24 GB VRAM. `accelerator="gpu"` + `serverless=True` tells MMF to use a driver-only predict path (Spark Connect Python workers are CPU-only on serverless, so distributed pandas-UDF inference doesn't help).
# MAGIC
# MAGIC A single `run_forecast` call:
# MAGIC
# MAGIC - writes per-model rolling-backtest metrics into `mmf_evaluation`
# MAGIC - writes the per-model forward forecasts into `mmf_scoring`
# MAGIC - registers each trained model into Unity Catalog under `{catalog}.{db}.<model>_supply_chain_demand`
# MAGIC - logs the MLflow run to the experiment at `/Users/<your-user>/mmf_supply_chain`

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
# MAGIC ## Pick the winning model by backtest SMAPE
# MAGIC
# MAGIC SMAPE = Symmetric Mean Absolute Percentage Error. Lower is better; under 10% is excellent for one-week-ahead distribution-style demand.

# COMMAND ----------

evaluation_df = (
    spark.read.table(f"{catalogName}.{dbName}.mmf_evaluation")
    .filter(f.col("run_id") == shared_run_id)
)

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
# MAGIC ## Reshape the winner's forecast into `(distribution_center, product, demand)`
# MAGIC
# MAGIC `mmf_scoring` stores one row per (unique_id, model) with `ds` and `y` as arrays (one element per forecast step). We:
# MAGIC
# MAGIC 1. filter to the winning model and this notebook's `run_id`
# MAGIC 2. explode the arrays so each step becomes a row
# MAGIC 3. split `unique_id` back into `product` and `wholesaler`
# MAGIC 4. join the DC→wholesaler mapping and sum demand per (DC, product)
# MAGIC
# MAGIC The final schema matches what `03_Derive_Raw_Material_Demand` expects.

# COMMAND ----------

scoring_df = spark.read.table(f"{catalogName}.{dbName}.mmf_scoring")

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

# Integrity check: every (product, wholesaler) in the source has a forecast row
assert (
    demand_df.select("product", "wholesaler").distinct().count()
    == per_wholesaler_forecast.select("product", "wholesaler").distinct().count()
)

display(per_wholesaler_forecast)

# COMMAND ----------

dc_mapping = spark.read.table(
    f"{catalogName}.{dbName}.distribution_center_to_wholesaler_mapping"
)

distribution_center_demand = (
    per_wholesaler_forecast
    .join(dc_mapping, on="wholesaler", how="left")
    .groupBy("distribution_center", "product")
    .agg(f.sum("demand").alias("demand"))
)

display(distribution_center_demand)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the forecast for downstream notebooks

# COMMAND ----------

distribution_center_demand.write.mode("overwrite").saveAsTable(
    f"{catalogName}.{dbName}.product_demand_forecasted"
)

print(f"Wrote {catalogName}.{dbName}.product_demand_forecasted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC
# MAGIC Open `03_Derive_Raw_Material_Demand` on standard serverless — it consumes `product_demand_forecasted` and writes raw-material requirements via a `networkx` BOM traversal.
# MAGIC
# MAGIC ## Third-party libraries
# MAGIC
# MAGIC | Library | License | Source |
# MAGIC |---|---|---|
# MAGIC | [many-model-forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting) | Databricks License | Databricks Industry Solutions |
# MAGIC | [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | Apache 2.0 | Amazon Science |
# MAGIC | [autogluon/chronos-2](https://huggingface.co/autogluon/chronos-2) | Apache 2.0 | AWS AutoGluon |
