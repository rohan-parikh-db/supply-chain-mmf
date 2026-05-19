# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Introduction & Data Setup
# MAGIC
# MAGIC Welcome! This notebook seeds the synthetic pharmaceutical supply-chain dataset that the rest of the pipeline (notebooks 02 → 05) operates on. **Run this first.**
# MAGIC
# MAGIC ## The scenario
# MAGIC
# MAGIC A pharma manufacturer operates:
# MAGIC
# MAGIC - **3 plants** that produce **30 product SKUs**
# MAGIC - **5 distribution centers** that warehouse those SKUs
# MAGIC - **30–60 wholesalers per DC** who order from them weekly
# MAGIC
# MAGIC The downstream notebooks answer the questions a planner cares about: *next week's demand per SKU per wholesaler, how much raw material we need to make that, and how to ship it for the lowest cost*. Notebook 05 adds three SQL functions so the whole pipeline becomes queryable from an AI agent.
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC 1. Reads two widgets (`catalog_name`, `db_name`)
# MAGIC 2. Uses the chosen catalog and creates the schema if it doesn't exist
# MAGIC 3. Calls `_resources/00-setup` with `reset_all_data=true`, which in turn triggers `_resources/01-data-generator` to write the six source tables:
# MAGIC    - `product_demand_historical` — 104 weeks of demand per (product, wholesaler)
# MAGIC    - `distribution_center_to_wholesaler_mapping`
# MAGIC    - `bom` — bill of materials (raw → intermediate → product)
# MAGIC    - `plant_supply` — max units each plant can produce per product
# MAGIC    - `transport_cost` — shipping cost from each plant to each DC
# MAGIC    - `list_prices` — unit price per SKU (used by notebook 05's revenue_at_risk function)
# MAGIC
# MAGIC ## Compute
# MAGIC
# MAGIC Standard **serverless** compute is fine. Data generation takes ~3–5 minutes (104 weeks × 900 series of synthetic ARMA demand, plus BOM + transport tables).

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

# Use the existing catalog (the user must have CREATE SCHEMA on it); create the schema if missing.
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{db_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate source data
# MAGIC
# MAGIC `reset_all_data=true` drops any existing tables in the schema and regenerates them.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=true

# COMMAND ----------

# MAGIC %md
# MAGIC # Next steps
# MAGIC
# MAGIC Source data is now in `{catalog_name}.{db_name}`. Run the remaining notebooks one at a time:
# MAGIC
# MAGIC | # | Notebook | Compute |
# MAGIC |---|---|---|
# MAGIC | 2 | `02_Fine_Grained_Demand_Forecasting` | **Serverless GPU** — Accelerator: A10, Environment version: 5 (set via *Configuration* tab) |
# MAGIC | 3 | `03_Derive_Raw_Material_Demand` | Standard serverless |
# MAGIC | 4 | `04_Optimize_Transportation` | Standard serverless |
# MAGIC | 5 | `05_Data_Analysis_&_Functions` | Standard serverless |
# MAGIC
# MAGIC > **Why no `%run` cascade?** Notebook 02 needs a GPU; 03/04/05 don't. Chaining everything via `%run` from this notebook would force GPU compute on the cheap steps. Running each notebook on its own lets you pick the right compute per step.
