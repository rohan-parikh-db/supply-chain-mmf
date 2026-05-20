# supply-chain-mmf

An end-to-end supply-chain optimization pipeline on Databricks — demand forecasting, raw-material planning, shipment optimization, and Unity Catalog SQL functions for AI agents. The worked example is pharmaceutical (BOM-driven manufacturer with a raw → primary → intermediate → finished-product hierarchy), but the pipeline generalizes to any BOM-driven supply chain: manufacturing, retail, consumer goods, automotive.

The pipeline runs entirely on serverless compute, with notebook 02 using serverless GPU.

The forecasting step is built on two technologies:

- [**Chronos-2**](https://huggingface.co/autogluon/chronos-2) — pretrained time-series foundation models from the AutoGluon project at AWS. Encoder-only transformers that produce zero-shot probabilistic forecasts.
- [**Many Model Forecasting (MMF)**](https://github.com/databricks-industry-solutions/many-model-forecasting) — Databricks' forecasting accelerator. Provides the model registry, rolling backtests, multi-model comparison, MLflow tracking, and Unity Catalog registration that wrap a single `run_forecast(...)` call.

| | |
|---|---|
| Pipeline | One-week-ahead SKU demand → raw-material requirements → least-cost shipment plan → UC SQL functions |
| Forecast accuracy | 10.82% mean SMAPE on 900 weekly series, zero-shot (4-window rolling backtest, verified run) |
| Forecast wall time | ~30 seconds on a single A10 GPU |
| Models compared | Chronos-2-Small (28M params) and Chronos-2 base (120M params), evaluated side-by-side |
| Compute | Standard serverless throughout; A10 GPU only for notebook 02 |

## Why these tools

### Chronos-2 instead of per-series classical methods

The [upstream Databricks accelerator](https://github.com/databricks-industry-solutions/supply-chain-optimization) fits one ExponentialSmoothing model per (product, wholesaler) — 900 independent fits via a pandas-UDF. Chronos-2 produces forecasts for the same 900 series in one batched GPU pass.

Beyond throughput, Chronos-2 brings forecasting capabilities that per-series classical models cannot:

- Zero-shot forecasting from a pretrained corpus of real and synthetic time series. No per-series fitting.
- Probabilistic outputs as quantile distributions learned from data, rather than Gaussian intervals from residuals.
- Cross-series pattern transfer — knowledge from retail, finance, energy, and IoT series in the training corpus generalizes to new domains.
- Robust behavior on short, noisy, or non-stationary series where AutoARIMA / ETS typically fail or fit poorly.

### MMF as the forecasting framework

`run_forecast(...)` reads like one line of code, but the orchestration it provides would be ~200 lines if hand-rolled:

- Model registry. Pass `active_models=["Chronos2Small", "Chronos2"]` and MMF resolves each name through a YAML config to the right Python class and HuggingFace repo. Swapping in `TimesFM` or `ChronosBoltBase` is a one-string change.
- Rolling-window backtests with configurable `backtest_length` and `stride`, producing per-series cross-validation metrics in a queryable Delta table.
- Multi-model comparison within a single call — all models' results land in the same evaluation table, tagged by model name, ready for winner selection or ensembling.
- MLflow tracking for every run, with parameters, metrics, and artifacts.
- Unity Catalog registration of each trained model, signed and versioned as an MLflow PyFunc.
- A serverless GPU code path (`accelerator="gpu", serverless=True`) that selects Chronos's driver-only predict mode, because Spark Connect Python workers on serverless are CPU-only.

This repo uses MMF idiomatically — through its public `run_forecast` API, with the canonical long-format input schema (`unique_id`, `ds`, `y`) and the standard kwargs that MMF's own `examples/serverless/foundation_serverless.ipynb` demonstrates. No monkey-patching, no manual model instantiation, no hand-rolled backtest loop.

## Pipeline

```mermaid
flowchart LR
    A[01: Setup<br/>data generation<br/>standard serverless] --> B[02: Forecasting<br/>MMF + Chronos-2<br/>serverless GPU A10]
    B --> C[03: Raw material demand<br/>networkx BOM traversal<br/>standard serverless]
    B --> D[04: Transport LP<br/>pulp + applyInPandas<br/>standard serverless]
    C --> E[05: UC SQL functions<br/>product_from_raw, raw_from_product,<br/>revenue_risk<br/>standard serverless]
    D --> E
```

| # | Notebook | Output |
|---|---|---|
| 1 | `01_Introduction_And_Setup` | 6 source Delta tables: `product_demand_historical`, `distribution_center_to_wholesaler_mapping`, `bom`, `plant_supply`, `transport_cost`, `list_prices` |
| 2 | `02_Fine_Grained_Demand_Forecasting` | MMF tables (`mmf_train`, `mmf_evaluation`, `mmf_scoring`) and the consumer-ready `product_demand_forecasted`; registers each Chronos-2 variant in UC |
| 3 | `03_Derive_Raw_Material_Demand` | `raw_material_demand` and `raw_material_supply` (the latter includes a synthetic shortage scenario) |
| 4 | `04_Optimize_Transportation` | `shipment_recommendations` — one row per (product, plant, distribution_center) with optimal `qty_shipped` |
| 5 | `05_Data_Analysis_&_Functions` | Three UC SQL functions: `product_from_raw`, `raw_from_product`, `revenue_risk` |

## Quick start

The repo requires a Databricks workspace with serverless GPU compute enabled. Only notebook 02 uses GPU; the rest run on standard serverless.

1. **Add the repo to your workspace** via `Repos → Add Repo → https://github.com/rohan-parikh-db/supply-chain-mmf.git`.

2. **Run notebook 01** on standard serverless. Set the widgets:
   - `catalog_name` — an existing catalog where you can create schemas (default `main`)
   - `db_name` — schema name to create (default `supply_chain_mmf`)

   Synthetic data generation takes ~3–5 minutes.

3. **Configure notebook 02 for serverless GPU** via the *Configuration* tab:
   - Accelerator: `A10`
   - Environment version: `5`

   Run with the same widget values as step 2. The model download and inference together take ~3–4 minutes.

4. **Run notebooks 03, 04, 05** on standard serverless. Each completes in under a minute.

5. **Query the results from SQL:**

   ```sql
   -- Most stressed raw material
   SELECT RAW, sum(Demand_Raw) - coalesce(sum(supply), 0) AS shortage
   FROM main.supply_chain_mmf.raw_material_demand d
   LEFT JOIN main.supply_chain_mmf.raw_material_supply s USING (RAW)
   GROUP BY RAW
   ORDER BY shortage DESC
   LIMIT 1;

   -- Products affected by that shortage
   SELECT * FROM main.supply_chain_mmf.product_from_raw('<RAW_id>');

   -- Revenue at risk
   SELECT product, sum(revenue_risk) AS revenue_at_risk
   FROM main.supply_chain_mmf.revenue_risk('<RAW_id>')
   GROUP BY product;
   ```

## Verified forecast results

End-to-end run on the synthetic dataset (900 weekly series, 4-window rolling backtest, A10 GPU, serverless env v5):

| Model | Params | Mean SMAPE | Notes |
|---|---|---|---|
| Chronos-2-Small | 28M | 0.108 | Selected as winner in the verified run. |
| Chronos-2 (base) | 120M | comparable | Marginal gains on this dataset; slower inference. |

Mean SMAPE of 0.108 corresponds to roughly 10.8% error. For one-week-ahead distribution-style demand forecasting, a common rule of thumb is: under 10% excellent, 10–20% good, 20–30% acceptable. A zero-shot foundation model lands in the "good" band with no per-series tuning.

## Using with Genie

The 13 Delta tables and 3 UC SQL functions produced by notebooks 01-05 are designed to be a complete grounded surface for a Databricks Genie Space — no code changes required, just Genie Space configuration.

**Setup (10-15 min):** run `06_Configure_Genie_Space.py`. The notebook verifies the schema is ready, adds table-level comments to sharpen Genie's grounding, and either provisions the Genie Space via the Workspace API or prints the manual UI steps as a fallback.

**Booth narrative — a rehearsed 5-question linear flow that lands in ~90 seconds:**

1. *"Which raw materials are short next week?"* → SQL aggregation on `raw_material_demand` vs `raw_material_supply`.
2. *"Of those, which one puts the most revenue at risk?"* → Genie calls `revenue_risk()` and ranks.
3. *"If we can't source enough of that material, which finished products are affected?"* → Calls `product_from_raw()`.
4. *"How much weekly revenue does that represent?"* → Sums `revenue_risk()` output.
5. *"Where should we ship the available supply to minimize loss?"* → Joins `shipment_recommendations` with the affected products.

The full question catalog (booth narrative + extended depth questions for one-on-one customer conversations) is in [`genie_seed_questions.md`](./genie_seed_questions.md), with a mapping from each question to the UC SQL function it should invoke.

## Extending with agents (Databricks-native and AWS-native)

The same Unity Catalog SQL functions that ground Genie also serve as the universal tool surface for any agent runtime. Two recommended paths:

| Path | Best fit | Where to start |
|---|---|---|
| **Mosaic AI Agent Framework / Agent Bricks** | Databricks-standardized customers; agent should run inside the Databricks identity + governance boundary | Wrap `product_from_raw`, `raw_from_product`, `revenue_risk` as agent tools in Mosaic AI Agent Framework; deploy to Model Serving |
| **Amazon Bedrock AgentCore** | AWS-standardized customers; agent composes Databricks tools with other AWS services (Lambda, Step Functions, EventBridge) in a single AWS-resident runtime | Walk through [`07_Bedrock_AgentCore_Integration.md`](./07_Bedrock_AgentCore_Integration.md) |

Both paths share the same UC functions and produce the same answers. Joint AWS-Databricks customers commonly run both, with Unity Catalog as the shared data + tool layer. See `07_Bedrock_AgentCore_Integration.md` for the detailed architecture, OAuth M2M auth setup, and Lambda action-group code stub.

## Project structure

```
supply-chain-mmf/
├── 01_Introduction_And_Setup.py
├── 02_Fine_Grained_Demand_Forecasting.py    (serverless GPU)
├── 03_Derive_Raw_Material_Demand.py
├── 04_Optimize_Transportation.py
├── 05_Data_Analysis_&_Functions.py
├── 06_Configure_Genie_Space.py
├── 07_Bedrock_AgentCore_Integration.md
├── genie_seed_questions.md
├── _resources/
│   ├── 00-setup.py
│   ├── 01-data-generator.py
│   └── 02-generate-supply.py
├── LICENSE
└── README.md
```

## Serverless compatibility notes

The upstream accelerator targets classic Databricks Runtime. The following modifications were needed to run on serverless (Spark Connect):

| File | Issue | Fix |
|---|---|---|
| `02_Fine_Grained_Demand_Forecasting` | Upstream uses statsmodels ExponentialSmoothing inside a pandas-UDF — runs on CPU executors and doesn't benefit from serverless GPU | Replaced with `mmf_sa.run_forecast(active_models=["Chronos2Small", "Chronos2"], accelerator="gpu", serverless=True)`. Added `hf_transfer` to the pip install and set `HF_HUB_ENABLE_HF_TRANSFER=0` at runtime so HuggingFace downloads succeed regardless of how the runtime configures the env var. |
| `03_Derive_Raw_Material_Demand` | `graphframes.GraphFrame(df)` accesses `DataFrame.sql_ctx`, which Spark Connect removed | Replaced with a single-node `networkx` traversal on `bom.toPandas()`. The BOM has under 100 nodes, so single-node is the right tool regardless of compatibility. |
| `04_Optimize_Transportation` | `spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")` is not settable on serverless | Wrapped in try/except. |
| `_resources/00-setup.py` | `dbutils.notebook.entry_point.getDbutils()...getContext().tags().apply('user')` returns None on serverless | Replaced with `spark.sql("select current_user()")`. |
| `_resources/01-data-generator.py` | Same AQE-config issue; `statsmodels` and `matplotlib` not preinstalled on env v5 | Added explicit `%pip install statsmodels matplotlib`; wrapped AQE config set in try/except. |
| `_resources/02-generate-supply.py` | `df.rdd.flatMap(...)` is not supported on Spark Connect | Replaced with `[r[col] for r in df.collect()]`. |

## Acknowledgments

- Upstream supply-chain accelerator: [databricks-industry-solutions/supply-chain-optimization](https://github.com/databricks-industry-solutions/supply-chain-optimization)
- Intermediate fork with agentic functions: [lara-openai/databricks-supply-chain](https://github.com/lara-openai/databricks-supply-chain)
- Many Model Forecasting: [databricks-industry-solutions/many-model-forecasting](https://github.com/databricks-industry-solutions/many-model-forecasting)
- Chronos-2 foundation models: [autogluon on HuggingFace](https://huggingface.co/autogluon)

## License

Apache 2.0 — see [LICENSE](LICENSE).
