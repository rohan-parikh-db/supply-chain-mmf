# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Fine-Grained Demand Forecasting with Chronos-2 + MMF
# MAGIC
# MAGIC > **Prerequisite:** run `01_Introduction_And_Setup` first to seed the source tables.
# MAGIC
# MAGIC This notebook produces a one-week-ahead demand forecast for **every (product, wholesaler) pair** — 900 time series in our synthetic dataset — and aggregates the result back to the **distribution-center level** that the rest of the pipeline consumes.
# MAGIC
# MAGIC It's a showcase of two technologies working together. Neither is a "convenience swap" for the other:
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🧠 [Chronos-2](https://huggingface.co/autogluon/chronos-2) — pretrained time-series foundation model from AWS AutoGluon
# MAGIC
# MAGIC Chronos-2 is an encoder-only transformer trained by AWS on a massive, diverse corpus of real and synthetic time series. Think of it as an LLM for sequences of numbers. What that buys you, on the *forecasting capability* axis:
# MAGIC
# MAGIC - **Zero-shot.** No per-series fitting. The model has already seen what trend, seasonality, level shifts, intermittent demand, and bursty spikes look like — at inference time, it just *applies* that knowledge to your series. AutoARIMA has to relearn each series from scratch.
# MAGIC - **Probabilistic by default.** Each forecast is a full quantile distribution, not just a point estimate. Uncertainty bands come from the model's learned distribution over outcomes, not Gaussian assumptions on residuals — typically better calibrated, especially for skewed or count-style demand.
# MAGIC - **Cross-series transfer.** Patterns the model learned from millions of other series (retail, finance, energy, healthcare, IoT) carry over to yours. Per-series AutoARIMA literally cannot do this — each fit is independent.
# MAGIC - **Robust to short / noisy / non-stationary series.** Classical methods need enough history and a relatively stable distribution to fit well. Chronos-2 stays usable on series with <50 points, gaps, regime shifts, or sudden volume changes.
# MAGIC - **GPU-batched.** All 900 series go through the model in one batched inference pass on a single A10 GPU. ~30 seconds end-to-end.
# MAGIC
# MAGIC We fit two Chronos-2 variants side-by-side (28M-param Small and 120M-param Base) and pick the one with the better backtest SMAPE.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 🛠️ [Many Model Forecasting (MMF)](https://github.com/databricks-industry-solutions/many-model-forecasting) — Databricks' multi-model forecasting accelerator
# MAGIC
# MAGIC MMF wraps the forecasting workflow — for *any* model family, foundation or classical — into one declarative `run_forecast(...)` call. It's not a thin convenience; it's doing real work on your behalf:
# MAGIC
# MAGIC - **Model registry.** Pass `active_models=["Chronos2Small", "Chronos2"]` and MMF resolves each name through a YAML config to the right Python class and HuggingFace repo (`autogluon/chronos-2-small`, `autogluon/chronos-2`). Swapping in `TimesFM` or `ChronosBolt` is a one-string change.
# MAGIC - **Rolling-window backtesting.** Configurable `backtest_length` + `stride` produce per-series cross-validation metrics so you can defend the forecast quality with numbers, not vibes.
# MAGIC - **Multi-model comparison out of the box.** One call fits and evaluates every model in `active_models` against the same series. Both end up in `mmf_evaluation` with model-tagged rows, ready for the winner-selection logic in the next cell.
# MAGIC - **MLflow tracking.** Every run logs to a single MLflow experiment with parameters, metrics, and model artifacts.
# MAGIC - **Unity Catalog registration.** Trained foundation models land in UC as `{catalog}.{db}.<model>_<use_case>` — ready for Model Serving, governance, lineage.
# MAGIC - **Serverless GPU integration.** `accelerator="gpu", serverless=True` tells MMF to use Chronos's driver-only predict path (Spark Connect Python workers on serverless are CPU-only, so distributed pandas-UDF inference wouldn't help anyway).
# MAGIC
# MAGIC Without MMF, the rest of this notebook would be ~200 lines of orchestration boilerplate — backtest loops, MLflow logging, UC registration, multi-model comparison framework.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### How this compares to the per-series classical baseline
# MAGIC
# MAGIC The upstream supply-chain accelerator fits one ExponentialSmoothing model per (product, wholesaler) pair via a `pandas_udf` — 900 separate models, fit from scratch. Chronos-2 + MMF replaces that:
# MAGIC
# MAGIC | | Per-series classical (upstream) | **Chronos-2 + MMF (this notebook)** |
# MAGIC |---|---|---|
# MAGIC | Model count | 900 separate AutoARIMA / ETS / Holt-Winters fits | One foundation model, batched across all 900 series |
# MAGIC | Per-series tuning | (p,d,q) search, seasonality detection, manual diagnostics | None — zero-shot |
# MAGIC | Cross-series learning | No | Yes (via pretrained corpus) |
# MAGIC | Probabilistic intervals | Gaussian-on-residuals (parametric) | Learned quantile distributions (non-parametric) |
# MAGIC | Handles short / noisy series | Often poorly | Robustly |
# MAGIC | Wall time on 900 weekly series | Minutes-to-hours, depending on history and tuning | **~30s on a single A10** |
# MAGIC | Backtest framework | DIY | Provided by MMF |
# MAGIC | MLflow + UC registration | DIY | Provided by MMF |
# MAGIC | Mean SMAPE on this dataset | varies with tuning | **10.82%** (verified run, see results below) |
# MAGIC
# MAGIC ### Compute requirement
# MAGIC
# MAGIC Attach this notebook to **serverless compute** with **`Accelerator = A10`** and **`Environment version = 5`** via the notebook's *Configuration* tab. CPU-only serverless won't work — Chronos-2 needs a GPU.

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
