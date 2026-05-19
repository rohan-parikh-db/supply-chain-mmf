# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Transport Optimization
# MAGIC
# MAGIC Prerequisite: notebook 02 has populated `product_demand_forecasted`.
# MAGIC
# MAGIC Given the forecasted demand per (DC, product), the per-plant supply caps, and the per-(plant, DC) shipping costs, this notebook solves one integer linear program (ILP) per product to find the cheapest shipment plan. The result is written to `shipment_recommendations`, one row per (product, plant, DC) with the recommended `qty_shipped`.
# MAGIC
# MAGIC ### The optimization problem
# MAGIC
# MAGIC For each product, minimize:
# MAGIC
# MAGIC ```
# MAGIC Σ over (plant, dc):  cost[plant→dc] * qty_shipped[plant→dc]
# MAGIC ```
# MAGIC
# MAGIC Subject to:
# MAGIC
# MAGIC - `qty_shipped` is a non-negative integer
# MAGIC - Sum of units leaving each plant ≤ that plant's supply for this product
# MAGIC - Sum of units arriving at each DC ≥ that DC's forecasted demand for this product
# MAGIC
# MAGIC ### Scaling
# MAGIC
# MAGIC One ILP per product is independent of the others, so all 30 are solved in parallel via Spark's `groupBy(...).applyInPandas(...)`. The same code scales to hundreds of thousands of products by adding executors.

# COMMAND ----------

# MAGIC %pip install pulp --quiet

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

import re
import pandas as pd
import pulp
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read inputs and reshape for the LP
# MAGIC
# MAGIC - **Demand** — pivot `product_demand_forecasted` so each row is one product and columns are `Demand_Distribution_Center_*`.
# MAGIC - **Supply** — per-(product, plant) caps, renamed to `Supply_Plant_*`.
# MAGIC - **Cost** — per-(product, plant) costs to each DC, renamed to `Cost_Distribution_Center_*`.
# MAGIC
# MAGIC Joining all three on `product` gives one row per (product, plant) carrying everything the LP solver needs.

# COMMAND ----------

# Demand for each distribution center, one line per product
forecasted_demand = spark.read.table(f"{catalogName}.{dbName}.product_demand_forecasted")
forecasted_demand = forecasted_demand.groupBy("Product").pivot("distribution_center").agg(f.first("demand").alias("demand"))
for name in forecasted_demand.schema.names:
  forecasted_demand = forecasted_demand.withColumnRenamed(name, name.replace("Distribution_Center", "Demand_Distribution_Center"))
forecasted_demand = forecasted_demand.withColumnRenamed("Product", "Product".replace("Product", "product")).sort("product")
display(forecasted_demand)

# COMMAND ----------

# MAGIC %md
# MAGIC We rename columns for clarity, e.g., Distribution_Center → Demand_Distribution_Center.

# COMMAND ----------

# Plant supply, one line per product
plant_supply = spark.read.table(f"{catalogName}.{dbName}.plant_supply")
for name in plant_supply.schema.names:
  plant_supply = plant_supply.withColumnRenamed(name, name.replace("plant", "Supply_Plant"))
plant_supply = plant_supply.sort("product")
display(plant_supply)

# COMMAND ----------

# Transportation cost table, one, line per product and plant
transport_cost_table = spark.read.table(f"{catalogName}.{dbName}.transport_cost")
for name in transport_cost_table.schema.names:
  transport_cost_table = transport_cost_table.withColumnRenamed(name, name.replace("Distribution_Center", "Cost_Distribution_Center"))
display(transport_cost_table)

# COMMAND ----------

# MAGIC %md
# MAGIC We join transportation costs, supply, and demand tables into a single DataFrame for optimization.

# COMMAND ----------

# Create a table with all information to iterate over. The table has one row per product and plant, with column-wise
# - The costs to ship from plant (rowwise) to distribution center (columnwise)
# - The supply restrictions for each product (rowwise) from each plant (columnwise)
# - The demand resctrictions for each product (rowwise) to mee the demand of each distribution center (columnwise)
lp_table_all_info = (transport_cost_table.
                     join(plant_supply, ["product"], how="inner").
                     join(forecasted_demand, ["product"], how="inner")
                    )
display(lp_table_all_info)

# COMMAND ----------

# MAGIC %md
# MAGIC We specify the structured of the output table

# COMMAND ----------

# Define output schema of final result table
res_schema = StructType(
  [
    StructField('product', StringType()),
    StructField('plant', StringType()),
    StructField('distribution_center', StringType()),
    StructField('qty_shipped', IntegerType())
  ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC define a function that solves the optimisation problem

# COMMAND ----------

# Define a function that solves the LP for one product
def transport_optimization(pdf: pd.DataFrame) -> pd.DataFrame:

  #Plants list, this defines the order of other data structures related to plants
  plants_lst = sorted(pdf["plant"].unique().tolist())

  # Distribution center list, this defines the order of other data structures related to distribution centers
  p = re.compile('^Cost_(Distribution_Center_\d)$')
  distribution_centers_lst = sorted([ s[5:] for s in list(pdf.columns.values) if p.search(s) ])

  # Define all possible routes
  routes = [(p, d) for p in plants_lst for d in distribution_centers_lst]

  # Create a dictionary which contains the LP variables. The reference keys to the dictionary are the plant's name, then the distribution center's name and the
  # data is Route_Tuple. (e.g. ["plant_1"]["distribution_center_1"]: Route_plant_1_distribution_center_1). Set lower limit to zero, upper limit to None and
  # define as integers
  vars = pulp.LpVariable.dicts("Route", (plants_lst, distribution_centers_lst), 0, None, pulp.LpInteger)

  # Subset other lookup tables
  ss_prod = pdf[ "product" ][0]

  # Costs, order of distribution centers and plants matter
  transport_cost_table_pdf = pdf.filter(regex="^Cost_Distribution_Center_\d+$|^plant$")
  transport_cost_table_pdf = (transport_cost_table_pdf.
                              rename(columns=lambda x: re.sub("^Cost_Distribution_Center","Distribution_Center",x)).
                              set_index("plant").
                              reindex(plants_lst, axis=0).
                              reindex(distribution_centers_lst, axis=1)
                             )
  costs = pulp.makeDict([plants_lst, distribution_centers_lst], transport_cost_table_pdf.values.tolist(), 0)

  # Supply, order of plants matters
  plant_supply_pdf = (pdf.
                      filter(regex="^Supply_Plant_\d+$").
                      drop_duplicates().
                      rename(columns=lambda x: re.sub("^Supply_Plant","plant",x)).
                      reindex(plants_lst, axis=1)
                     )

  supply = plant_supply_pdf.to_dict("records")[0]

  # Demand, order of distribution centers matters
  distribution_center_demand_pdf =  (pdf.
                      filter(regex="^Demand_Distribution_Center_\d+$").
                      drop_duplicates().
                      rename(columns=lambda x: re.sub("^Demand_Distribution_Center","Distribution_Center",x)).
                      reindex(distribution_centers_lst, axis=1)
                     )

  demand = distribution_center_demand_pdf.to_dict("records")[0]

  # Create the 'prob' variable to contain the problem data
  prob = pulp.LpProblem("Product_Distribution_Problem", pulp.LpMinimize)

  # Add objective function to 'prob' first
  prob += (
      pulp.lpSum([vars[p][d] * costs[p][d] for (p, d) in routes]),
      "Sum_of_Transporting_Costs",
  )

  # Add supply restrictions
  for p in plants_lst:
      prob += (
          pulp.lpSum([vars[p][d] for d in distribution_centers_lst]) <= supply[p],
          f"Sum_of_Products_out_of_Plant_{p}",
      )

  # Add demand restrictions
  for d in distribution_centers_lst:
      prob += (
          pulp.lpSum([vars[p][d] for p in plants_lst]) >= demand[d],
          f"Sum_of_Products_into_Distibution_Center{d}",
      )

  # The problem is solved using PuLP's choice of Solver
  prob.solve()

  # Write output fot the product
  if (pulp.LpStatus[prob.status] == "Optimal"):
    name_lst = [ ]
    value_lst = [ ]
    for v in prob.variables():
      name_lst.append(v.name) 
      value_lst.append(v.varValue)
      res = pd.DataFrame(data={'name': name_lst, 'qty_shipped': value_lst})
      res[ "qty_shipped" ] = res[ "qty_shipped" ].astype("int")
      res[ "plant" ] =  res[ "name" ].str.extract(r'(plant_\d+)')
      res[ "distribution_center" ] =  res[ "name" ].str.extract(r'(Distribution_Center_\d+)')
      res[ "product" ] = ss_prod
      res = res.drop("name", axis = 1)
      res = res[[ "product", "plant", "distribution_center", "qty_shipped"]]
  else:
      res = pd.DataFrame(data= {  "product" : [ ss_prod ] , "plant" : [ None ], "distribution_center" : [ None ], "qty_shipped" : [ None ]})
  return res

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solve all products in parallel via `applyInPandas`
# MAGIC
# MAGIC Each Spark partition runs `transport_optimization` for one product on its own row. With 30 products this fits trivially on serverless; with hundreds of thousands the same code works — just throw more executors at it.

# COMMAND ----------

try:
    spark.conf.set("spark.databricks.optimizer.adaptive.enabled", "false")
except Exception:
    pass  # AQE config is not settable on serverless — safe to ignore.

n_tasks = lp_table_all_info.select("product").distinct().count()

optimal_transport_df = (
    lp_table_all_info
    .repartition(n_tasks, "product")
    .groupBy("product")
    .applyInPandas(transport_optimization, schema=res_schema)
)
display(optimal_transport_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the shipment plan

# COMMAND ----------

optimal_transport_df.write.mode("overwrite").saveAsTable(
    f"{catalogName}.{dbName}.shipment_recommendations"
)
print(f"Wrote {catalogName}.{dbName}.shipment_recommendations")

display(spark.sql(f"SELECT * FROM {catalogName}.{dbName}.shipment_recommendations"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next
# MAGIC
# MAGIC Open `05_Data_Analysis_&_Functions` — it builds three Unity Catalog SQL functions (`product_from_raw`, `raw_from_product`, `revenue_risk`) so this whole pipeline becomes queryable from an AI agent.
# MAGIC
# MAGIC ## Third-party libraries
# MAGIC
# MAGIC | Library | License | Source |
# MAGIC |---|---|---|
# MAGIC | [pulp](https://github.com/coin-or/pulp) | MIT | COIN-OR Foundation |