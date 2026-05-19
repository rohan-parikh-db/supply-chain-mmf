# Databricks notebook source
# MAGIC %md Credits: This is adapted from a Databricks solution accelerator for supply chain optimization, available at https://github.com/databricks-industry-solutions/supply-chain-optimization. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/supply-chain-distribution-optimization. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction
# MAGIC
# MAGIC
# MAGIC **Context**:
# MAGIC We have a pharmaceutical supply chain, with 3 plants that deliver a set of 30 product SKUs to 5 distribution centers. Each distribution center is assigned to a set of between 30 and 60 wholesalers. All these parameters are treated as variables such that the pattern of the code may be scaled. Each wholesaler has a demand series for each of the products. 
# MAGIC
# MAGIC
# MAGIC **The following are given**:
# MAGIC - the demand series for each product in each wholesaler
# MAGIC - a mapping table that uniquely assigns each distribution center to a wholesaler. This is a simplification as it is possible that one wholesaler obtains products from different distribution centers.
# MAGIC - a table that assigns the costs of shipping a product from each manufacturing plant to each distribution center
# MAGIC - a table of the maximum quantities of product that can be produced and shipped from each plant to each of the distribution centers

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup
# MAGIC Run this notebook to generate the data and run all subsequent notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up catalog and database name
# MAGIC
# MAGIC The catalog and database will be created automatically if they do not exist. The data generator script will create the necessary tables and populate them with sample data.
# MAGIC
# MAGIC The default catalog is 'main' and schema/database is 'supply_chain_db'. Feel free to change to another catalog and schema/database.

# COMMAND ----------

# Create widgets for catalog and database names
dbutils.widgets.text("catalog_name", "main", "Catalog Name")
dbutils.widgets.text("db_name", "supply_chain_db", "Database Name")

# Get values from widgets
catalog_name = dbutils.widgets.get("catalog_name")
db_name = dbutils.widgets.get("db_name")

# Display the values being used
print(f"Using catalog: {catalog_name}")
print(f"Using database: {db_name}")

# COMMAND ----------

# Use existing catalog; create schema if missing.
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{db_name}")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=true

# COMMAND ----------

# MAGIC %md
# MAGIC # Next steps
# MAGIC
# MAGIC Data is now seeded. Run the remaining notebooks in order:
# MAGIC
# MAGIC 1. **`02_Fine_Grained_Demand_Forecasting`** — MMF + Chronos-2 foundation models. **Requires serverless GPU (A10, Environment version 5)** — set this in the notebook's Configuration tab.
# MAGIC 2. **`03_Derive_Raw_Material_Demand`** — BOM traversal via `networkx`. Standard serverless.
# MAGIC 3. **`04_Optimize_Transportation`** — LP via `pulp` + `applyInPandas`. Standard serverless.
# MAGIC 4. **`05_Data_Analysis_&_Functions`** — Pure SQL + UC functions. Standard serverless.
# MAGIC
# MAGIC No `%run` cascade — each subsequent notebook is meant to be triggered on its own (notebook 02 needs GPU compute, the others don't, so chaining via `%run` from this notebook isn't possible on serverless).