# Databricks notebook source
# MAGIC %md
# MAGIC This project is based on Databricks' supply chain optimization solution accelerator available at: https://github.com/databricks-industry-solutions/supply-chain-optimization. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/supply-chain-distribution-optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC # Derive Raw Material Demand
# MAGIC
# MAGIC In this notebook, we process product demand forecasts to determine raw material requirements using a graph-based approach.
# MAGIC

# COMMAND ----------

# MAGIC %pip install networkx --quiet
# MAGIC %restart_python

# COMMAND ----------

# Create widgets for catalog and database names
dbutils.widgets.text("catalog_name", "main", "Catalog Name")
dbutils.widgets.text("db_name", "supply_chain_db", "Database Name")

# COMMAND ----------

# Get values from widgets
catalog_name = dbutils.widgets.get("catalog_name")
db_name = dbutils.widgets.get("db_name")

# Display the values being used
print(f"Using catalog: {catalog_name}")
print(f"Using database: {db_name}")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalog_name $dbName=$db_name

# COMMAND ----------

print(catalogName)
print(dbName)


# COMMAND ----------

import os
import string
import random
import numpy as np
import pandas as pd
import networkx as nx
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# COMMAND ----------

# MAGIC %md
# MAGIC Read the product_demand_forecasted Delta table that we just created. We retrieve the forecasted demand as an input for subsequent calculations. 
# MAGIC
# MAGIC Read the BOM (bill of materials) which contains the relationship between products and raw materials. This allows us to derive raw material demand. 

# COMMAND ----------

demand_df = spark.read.table(f"{catalogName}.{dbName}.product_demand_forecasted").select("product", "demand")
bom = spark.read.table(f"{catalogName}.{dbName}.bom")

# COMMAND ----------

# MAGIC %md
# MAGIC We build a single-node `networkx` DiGraph from the BOM (BOM is small, < 100 nodes).
# MAGIC Edges point from `material_in` (raw side) toward `material_out` (finished side), with `qty` on each edge.
# MAGIC
# MAGIC * **Raw materials** = nodes with `in_degree == 0` (nothing feeds them).
# MAGIC * **Finished products** = nodes with `out_degree == 0` (they feed nothing further).
# MAGIC
# MAGIC For each (raw, product) pair where the product is reachable from the raw, we sum the qty along every simple path — this handles diamond BOMs where one raw feeds the same product via multiple intermediates.
# MAGIC
# MAGIC We swapped out the original `GraphFrame` / `aggregateMessages` traversal because Spark Connect (serverless Spark) doesn't expose `DataFrame.sql_ctx`, which `graphframes` requires.

# COMMAND ----------

bom_pdf = bom.toPandas()

G = nx.DiGraph()
for _, row in bom_pdf.iterrows():
    G.add_edge(row["material_in"], row["material_out"], qty=int(row["qty"]))

raws = [n for n in G.nodes if G.in_degree(n) == 0]
finals = {n for n in G.nodes if G.out_degree(n) == 0}

rows = []
for raw in raws:
    reachable = nx.descendants(G, raw) & finals
    for product in reachable:
        total_qty = 0
        for path in nx.all_simple_paths(G, raw, product):
            path_qty = 1
            for u, v in zip(path[:-1], path[1:]):
                path_qty *= G[u][v]["qty"]
            total_qty += path_qty
        rows.append({"RAW": raw, "product": product, "QTY_RAW": int(total_qty)})

aggregated_bom_pdf = (
    pd.DataFrame(rows, columns=["RAW", "product", "QTY_RAW"])
    .sort_values(["product", "RAW"])
    .reset_index(drop=True)
)

agg_bom_schema = StructType([
    StructField("RAW", StringType()),
    StructField("product", StringType()),
    StructField("QTY_RAW", IntegerType()),
])
aggregated_bom = spark.createDataFrame(aggregated_bom_pdf, schema=agg_bom_schema)
display(aggregated_bom)

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate raw material demand: computes teh raw material demand by multiplying raw material quantities with forecasted product demand

# COMMAND ----------

demand_raw_df = (demand_df.
      join(aggregated_bom, ["product"], how="inner").
      withColumn("Demand_Raw", f.col("QTY_RAW")*f.col("Demand")).
      withColumnRenamed("Demand","Demand_product").
      orderBy(f.col("product"),f.col("RAW"))
)
display(demand_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the raw material demand as a table 

# COMMAND ----------

# Write the data 
demand_raw_df.write.mode("overwrite").saveAsTable(f"{catalogName}.{dbName}.raw_material_demand")

# COMMAND ----------

# MAGIC %md
# MAGIC Execute next notebook to generate supply data 

# COMMAND ----------

import os
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
notebook_path = os.path.join(os.path.dirname(notebook_path),"_resources/02-generate-supply")
dbutils.notebook.run(notebook_path, 600, {"dbName": dbName, "catalogName": catalogName})

# COMMAND ----------

raw_material_supply_df = spark.read.table(f"{catalogName}.{dbName}.raw_material_supply")
raw_material_supply_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pulp                                 | A python Linear Programming API      | https://github.com/coin-or/pulp/blob/master/LICENSE        | https://github.com/coin-or/pulp                      |