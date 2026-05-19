# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Data Analysis & UC SQL Functions
# MAGIC
# MAGIC > **Prerequisite:** notebooks 02–04 have populated `product_demand_forecasted`, `raw_material_demand`, `raw_material_supply`, and `shipment_recommendations`.
# MAGIC
# MAGIC This notebook does two things:
# MAGIC
# MAGIC 1. **Surfaces the most critical raw material** — the one whose total supply most undershoots total demand. This SKU is the natural anchor for the rest of the analysis.
# MAGIC 2. **Creates three reusable Unity Catalog SQL functions** so the pipeline becomes queryable by any SQL client *and* directly callable as tools from a Genie/Agent integration:
# MAGIC
# MAGIC | Function | What it returns |
# MAGIC |---|---|
# MAGIC | `product_from_raw(raw_material)` | All downstream products that depend on a given raw material, with per-step quantities. |
# MAGIC | `raw_from_product(product)` | All upstream raw materials needed to make a given product. |
# MAGIC | `revenue_risk(raw_material)` | For a given raw material, the dollar revenue at risk if the current shortage propagates to finished-product output. |
# MAGIC
# MAGIC Once these exist, an AI agent can answer questions like *"How much revenue is at risk if we can't get enough material H7AZR?"* with a single SQL call.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog_name", "main", "Catalog Name")
dbutils.widgets.text("db_name", "supply_chain_mmf", "Database Name")

catalog_name = dbutils.widgets.get("catalog_name")
db_name = dbutils.widgets.get("db_name")

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=false $catalogName=$catalog_name $dbName=$db_name

# COMMAND ----------

catalogName = catalog_name
dbName = db_name

print(f"Using catalog: {catalogName}")
print(f"Using database: {dbName}")

# COMMAND ----------

# MAGIC %md
# MAGIC The goal is to identify the most critical raw material with potential shortages.
# MAGIC
# MAGIC We join raw_material_demand and raw_material_supply to compute shortages for each raw material and look at the difference between total supply and total demand

# COMMAND ----------

mat_number = spark.sql(f"""
SELECT demand.RAW, SUM(demand.Demand_Raw) as total_demand, SUM(supply.supply) as total_supply, total_supply-total_demand as demand_difference
FROM {catalogName}.{dbName}.raw_material_demand demand
LEFT JOIN {catalogName}.{dbName}.raw_material_supply supply ON demand.RAW=supply.raw
WHERE supply.supply is not null
GROUP BY demand.RAW
ORDER BY demand_difference ASC
LIMIT 1
""").collect()[0][0]
print(f"Most critical raw material: {mat_number}")

# COMMAND ----------

# MAGIC %md
# MAGIC We query the bill of materials relationships. We map raw materials, intermediate materials and products. It gives us the hierarchical relationships: `raw_material → primary_material → intermediate_material → product`.
# MAGIC

# COMMAND ----------

spark.sql(f"""SELECT
    bom_primary.material_in as raw_material, bom_intermediate.qty as raw_qty, 
    bom_intermediate.material_in as primary_material, bom_intermediate.qty as primary_qty, 
    bom_intermediate.material_out as intermediate_material, bom_product.qty as intermediate_qty,
    bom_product.material_out as product 
FROM {catalogName}.{dbName}.bom bom_primary
JOIN {catalogName}.{dbName}.bom bom_intermediate ON bom_primary.material_out = bom_intermediate.material_in
JOIN {catalogName}.{dbName}.bom bom_product ON bom_intermediate.material_out = bom_product.material_in""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC We create a function `product_from_raw`. It takes a `raw_material` as input and returns all downstream relationships (raw --> product)
# MAGIC
# MAGIC We filter results by the given material. 

# COMMAND ----------

spark.sql(f'''
CREATE OR REPLACE FUNCTION {catalogName}.{dbName}.product_from_raw (
    input_material STRING COMMENT "MAT number of the raw material" DEFAULT "{mat_number}"
) RETURNS TABLE (
    raw_material STRING,
    raw_qty INT,
    primary_material STRING,
    primary_qty INT,
    intermediate_material STRING,
    intermediate_qty INT,
    product STRING
) COMMENT "Returns the raw and intermediate materials required to produce a SKU / Product"  RETURN 
SELECT
    bom_primary.material_in as raw_material, bom_intermediate.qty as raw_qty, 
    bom_intermediate.material_in as primary_material, bom_intermediate.qty as primary_qty, 
    bom_intermediate.material_out as intermediate_material, bom_product.qty as intermediate_qty,
    bom_product.material_out as product 
FROM {catalogName}.{dbName}.bom bom_primary
JOIN {catalogName}.{dbName}.bom bom_intermediate ON bom_primary.material_out = bom_intermediate.material_in
JOIN {catalogName}.{dbName}.bom bom_product ON bom_intermediate.material_out = bom_product.material_in
WHERE 
    product_from_raw.input_material = bom_primary.material_in 
    OR product_from_raw.input_material = bom_intermediate.material_in
    OR product_from_raw.input_material = bom_product.material_in
''')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We define a function that takes a product as input and returns all upstream relationships (product → raw).

# COMMAND ----------

product = "syringe_1"

spark.sql(f'''
CREATE OR REPLACE FUNCTION {catalogName}.{dbName}.raw_from_product (
    input_product STRING COMMENT "get the raw materials for a particular product" DEFAULT "{product}"
) RETURNS TABLE (
    raw_material STRING,
    raw_qty INT,
    primary_material STRING,
    primary_qty INT,
    intermediate_material STRING,
    intermediate_qty INT,
    product STRING
) COMMENT "Returns the raw and intermediate materials required to produce a SKU / Product"  RETURN 
SELECT
    bom_primary.material_in as raw_material, bom_intermediate.qty as raw_qty, 
    bom_intermediate.material_in as primary_material, bom_intermediate.qty as primary_qty, 
    bom_intermediate.material_out as intermediate_material, bom_product.qty as intermediate_qty,
    bom_product.material_out as product 
FROM {catalogName}.{dbName}.bom bom_primary
JOIN {catalogName}.{dbName}.bom bom_intermediate ON bom_primary.material_out = bom_intermediate.material_in
JOIN {catalogName}.{dbName}.bom bom_product ON bom_intermediate.material_out = bom_product.material_in
WHERE 
    input_product = bom_product.material_out
''')

# COMMAND ----------

spark.sql(f"SELECT * FROM {catalogName}.{dbName}.raw_from_product('syringe_1')").display()

# COMMAND ----------

spark.sql(f""" 
    SELECT * FROM {catalogName}.{dbName}.product_from_raw('{mat_number}')          
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC We create a function that calculates the risk of revenue loss due to raw material shortages.
# MAGIC
# MAGIC * material_shortage: Raw material shortage.
# MAGIC * product_shortage: Impact on product availability.
# MAGIC * revenue_risk: Potential revenue loss.
# MAGIC
# MAGIC We compute the shortages for each raw material. We map raw materials to products. We then join with prices to calculate revenues loss based on product prices. 

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalogName}.{dbName}.revenue_risk (
  raw_material STRING COMMENT "MAT number of the raw material, for example revenue_risk.raw_material" DEFAULT "{mat_number}"
) RETURNS TABLE (
  material STRING,
  product STRING,
  material_supply INT,
  material_demand INT,
  material_shortage INT,
  product_shortage FLOAT,
  revenue_risk DECIMAL(10,2)
) COMMENT "Returns the forecasted revenue risk due to shortages given a raw material" RETURN 

WITH material_shortage AS (
    SELECT demand.RAW as material, SUM(demand.Demand_Raw) as total_demand, SUM(supply.supply) as total_supply, 
        CASE WHEN SUM(demand.Demand_Raw) > SUM(supply.supply) THEN SUM(demand.Demand_Raw)-SUM(supply.supply) ELSE 0 END as shortage
    FROM {catalogName}.{dbName}.raw_material_demand demand
    LEFT JOIN {catalogName}.{dbName}.raw_material_supply supply ON demand.RAW=supply.raw
    GROUP BY demand.RAW
)

SELECT 
    material_shortage.material, 
    pfr.product,
    material_shortage.total_supply as material_supply,
    material_shortage.total_demand as material_demand,
    material_shortage.shortage as material_shortage,
    material_shortage.shortage / SUM(pfr.intermediate_qty) / SUM(pfr.primary_qty / pfr.raw_qty) as product_shortage,
    (material_shortage.shortage / SUM(pfr.intermediate_qty) / SUM(pfr.primary_qty / pfr.raw_qty)) * SUM(list_prices.price) as revenue_at_risk
FROM material_shortage
JOIN (
    SELECT *
    FROM {catalogName}.{dbName}.product_from_raw(revenue_risk.raw_material)
) pfr ON material_shortage.material=pfr.raw_material
JOIN (
    SELECT price, product
    FROM {catalogName}.{dbName}.list_prices
) list_prices ON lower(list_prices.product) = lower(pfr.product)
WHERE material_shortage.material = revenue_risk.raw_material
GROUP BY material_shortage.material, pfr.product, material_shortage.total_supply, material_shortage.total_demand, material_shortage.shortage
;
""")

# COMMAND ----------

spark.sql(f"""
SELECT product, revenue_risk
FROM {catalogName}.{dbName}.revenue_risk('{mat_number}')
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## You're done!
# MAGIC
# MAGIC The full pipeline (notebooks 01 → 05) has produced:
# MAGIC
# MAGIC - **13 Delta tables** in `{catalog}.{db}` (source data + MMF intermediates + forecasted/derived/optimized outputs).
# MAGIC - **3 UC SQL functions** (`product_from_raw`, `raw_from_product`, `revenue_risk`) that any downstream tool can call.
# MAGIC
# MAGIC Try these against the schema:
# MAGIC
# MAGIC ```sql
# MAGIC -- Which products will be hit by a shortage of the most-stressed raw material?
# MAGIC SELECT * FROM {catalogName}.{dbName}.product_from_raw('<some_raw_id>');
# MAGIC
# MAGIC -- Which raw materials go into a specific finished SKU?
# MAGIC SELECT * FROM {catalogName}.{dbName}.raw_from_product('syringe_1');
# MAGIC
# MAGIC -- How much weekly revenue is at risk?
# MAGIC SELECT product, SUM(revenue_risk) AS revenue_at_risk
# MAGIC FROM {catalogName}.{dbName}.revenue_risk('<some_raw_id>')
# MAGIC GROUP BY product;
# MAGIC ```
# MAGIC
# MAGIC Connect this schema to a Databricks Genie space (or a custom agent) and the same questions become natural-language tools.