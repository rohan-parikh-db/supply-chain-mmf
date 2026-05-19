# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Derive Raw Material Demand from BOM
# MAGIC
# MAGIC > **Prerequisite:** notebook 02 has populated `product_demand_forecasted`.
# MAGIC
# MAGIC Given the forecasted demand per finished SKU (from notebook 02) and the **Bill of Materials (BOM)** (from notebook 01), this notebook computes how much of each **raw material** the supply chain needs to fulfill that demand.
# MAGIC
# MAGIC ## The graph
# MAGIC
# MAGIC The BOM describes a directed graph: raw materials feed primary materials, which feed intermediate materials, which feed finished products. Each edge carries a `qty` — how many units of the source are needed per unit of the destination.
# MAGIC
# MAGIC ```
# MAGIC raw_material  --qty-->  primary  --qty-->  intermediate  --qty-->  product
# MAGIC ```
# MAGIC
# MAGIC For each (raw, product) pair we sum the qty along every simple path — that handles "diamond" BOMs where one raw feeds the same product via multiple intermediates.
# MAGIC
# MAGIC ## Why networkx
# MAGIC
# MAGIC The right tool for a graph depends on its size. The upstream accelerator uses `graphframes` — a distributed graph engine built for billion-edge graphs (think: PageRank on the web crawl). Powerful, but the Spark coordination overhead dominates wall-time on small graphs.
# MAGIC
# MAGIC This BOM has fewer than 100 nodes. At that scale, `networkx` — the canonical single-node Python graph library — is genuinely the better fit: well-tested implementations of every standard algorithm, no distributed-execution overhead, and the traversal completes in milliseconds on the driver. As a bonus, `graphframes` calls `DataFrame.sql_ctx` (removed in Spark Connect), so it wouldn't run on serverless Spark anyway — but the primary reason for `networkx` here is simply that it's the right size of tool for the problem.

# COMMAND ----------

# MAGIC %pip install networkx --quiet
# MAGIC %restart_python

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

import pandas as pd
import networkx as nx
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read inputs
# MAGIC
# MAGIC - `product_demand_forecasted` — written by notebook 02 (one row per DC × product).
# MAGIC - `bom` — written by notebook 01 (one row per `material_in → material_out` edge with `qty`).

# COMMAND ----------

demand_df = (
    spark.read.table(f"{catalogName}.{dbName}.product_demand_forecasted")
    .select("product", "demand")
)
bom = spark.read.table(f"{catalogName}.{dbName}.bom")

display(bom.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the BOM graph and compute (raw, product, total_qty)
# MAGIC
# MAGIC - **Raw materials** = nodes with `in_degree == 0` (nothing feeds them).
# MAGIC - **Finished products** = nodes with `out_degree == 0` (they feed nothing further).

# COMMAND ----------

bom_pdf = bom.toPandas()

G = nx.DiGraph()
for _, row in bom_pdf.iterrows():
    G.add_edge(row["material_in"], row["material_out"], qty=int(row["qty"]))

raws = [n for n in G.nodes if G.in_degree(n) == 0]
finals = {n for n in G.nodes if G.out_degree(n) == 0}
print(f"BOM has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; "
      f"{len(raws)} raw materials, {len(finals)} finished products.")

rows = []
for raw in raws:
    for product in nx.descendants(G, raw) & finals:
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
# MAGIC ## Multiply by forecasted product demand → raw-material demand
# MAGIC
# MAGIC `Demand_Raw = QTY_RAW × forecasted_product_demand` summed over each (raw, product) pair.

# COMMAND ----------

demand_raw_df = (
    demand_df
    .join(aggregated_bom, on="product", how="inner")
    .withColumn("Demand_Raw", f.col("QTY_RAW") * f.col("demand"))
    .withColumnRenamed("demand", "Demand_product")
    .orderBy("product", "RAW")
)
display(demand_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta

# COMMAND ----------

demand_raw_df.write.mode("overwrite").saveAsTable(f"{catalogName}.{dbName}.raw_material_demand")
print(f"Wrote {catalogName}.{dbName}.raw_material_demand")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate raw-material supply
# MAGIC
# MAGIC `_resources/02-generate-supply` writes a `raw_material_supply` table. For two random SKUs and three random raw materials it caps supply below demand (so notebook 05 can compute revenue-at-risk); for everything else it sets supply slightly above demand.

# COMMAND ----------

import os

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
supply_notebook = os.path.join(os.path.dirname(notebook_path), "_resources/02-generate-supply")
dbutils.notebook.run(supply_notebook, 600, {"dbName": dbName, "catalogName": catalogName})

# COMMAND ----------

raw_material_supply_df = spark.read.table(f"{catalogName}.{dbName}.raw_material_supply")
display(raw_material_supply_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC
# MAGIC Open `04_Optimize_Transportation` — it uses `product_demand_forecasted` and the plant/transport tables to solve a per-product LP via `pulp` + `applyInPandas`.
