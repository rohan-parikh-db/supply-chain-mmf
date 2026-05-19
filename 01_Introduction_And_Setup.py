# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Introduction and Data Setup
# MAGIC
# MAGIC Run this notebook first. It seeds the synthetic pharmaceutical supply-chain dataset that the rest of the pipeline (notebooks 02 through 05) operates on.
# MAGIC
# MAGIC ### Scenario
# MAGIC
# MAGIC A pharmaceutical manufacturer operates:
# MAGIC
# MAGIC - 3 plants producing 30 product SKUs
# MAGIC - 5 distribution centers warehousing those SKUs
# MAGIC - 30 to 60 wholesalers per distribution center ordering weekly
# MAGIC
# MAGIC The downstream notebooks forecast next week's demand per SKU per wholesaler, derive the raw-material requirements to meet that demand, solve a least-cost transportation plan, and expose three Unity Catalog SQL functions for AI-agent integration.
# MAGIC
# MAGIC ### What this notebook does
# MAGIC
# MAGIC 1. Reads two widgets (`catalog_name`, `db_name`).
# MAGIC 2. Uses the chosen catalog and creates the schema if it does not exist.
# MAGIC 3. Calls `_resources/00-setup` with `reset_all_data=true`, which triggers `_resources/01-data-generator` to write six source tables:
# MAGIC    - `product_demand_historical` — 104 weeks of demand per (product, wholesaler)
# MAGIC    - `distribution_center_to_wholesaler_mapping`
# MAGIC    - `bom` — bill of materials (raw → intermediate → product)
# MAGIC    - `plant_supply` — maximum units each plant can produce per product
# MAGIC    - `transport_cost` — shipping cost from each plant to each distribution center
# MAGIC    - `list_prices` — unit price per SKU (used by notebook 05's `revenue_risk` function)
# MAGIC
# MAGIC ### Compute
# MAGIC
# MAGIC Standard serverless compute is sufficient. Data generation takes approximately 3 to 5 minutes (104 weeks across 900 synthetic ARMA series, plus BOM and transport tables).

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
# MAGIC | 2 | `02_Fine_Grained_Demand_Forecasting` | Serverless GPU — Accelerator A10, Environment version 5 (set in the Configuration tab) |
# MAGIC | 3 | `03_Derive_Raw_Material_Demand` | Standard serverless |
# MAGIC | 4 | `04_Optimize_Transportation` | Standard serverless |
# MAGIC | 5 | `05_Data_Analysis_&_Functions` | Standard serverless |
# MAGIC
# MAGIC Notebook 02 requires GPU compute while 03 through 05 do not. Running each notebook independently lets you select appropriate compute per step rather than forcing GPU compute on the entire pipeline.
