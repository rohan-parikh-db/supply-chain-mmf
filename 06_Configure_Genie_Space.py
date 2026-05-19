# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Configure a Genie Space over the supply-chain schema
# MAGIC
# MAGIC Prerequisite: notebooks 01-05 have populated the `<catalog>.<db>` schema with all 13 Delta tables and registered the 3 UC SQL functions (`product_from_raw`, `raw_from_product`, `revenue_risk`).
# MAGIC
# MAGIC This notebook does two things:
# MAGIC
# MAGIC 1. **Verifies** that the schema is in the correct state for Genie consumption (all tables exist, all functions registered, table comments populated where helpful).
# MAGIC 2. **Provisions** the Genie Space via the Databricks Workspace API, seeded with the curated example questions from `genie_seed_questions.md`.
# MAGIC
# MAGIC Once configured, the Genie Space supports natural-language conversations like *"Which raw materials are short next week?"* and *"How much weekly revenue is at risk if we can't source material H7AZR?"* — see `genie_seed_questions.md` for the full booth narrative.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog_name", "main", "Catalog Name")
dbutils.widgets.text("db_name", "supply_chain_mmf", "Database Name")
dbutils.widgets.text("genie_space_title", "Supply Chain — Demand & Shortage Risk", "Genie Space Title")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse ID (Pro or Serverless)")

catalog_name = dbutils.widgets.get("catalog_name")
db_name = dbutils.widgets.get("db_name")
space_title = dbutils.widgets.get("genie_space_title")
warehouse_id = dbutils.widgets.get("warehouse_id")

print(f"Catalog: {catalog_name}")
print(f"Schema: {db_name}")
print(f"Genie Space title: {space_title}")
print(f"Warehouse: {warehouse_id or '(not set — required for Genie Space creation)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Verify schema readiness for Genie

# COMMAND ----------

# Tables Genie should see
expected_tables = [
    "product_demand_historical",
    "product_demand_forecasted",
    "bom",
    "raw_material_demand",
    "raw_material_supply",
    "shipment_recommendations",
    "list_prices",
    "distribution_center_to_wholesaler_mapping",
    "plant_supply",
    "transport_cost",
    "mmf_train",
    "mmf_evaluation",
    "mmf_scoring",
]

actual_tables = {row.tableName for row in spark.sql(f"SHOW TABLES IN {catalog_name}.{db_name}").collect()}
missing = sorted(set(expected_tables) - actual_tables)

if missing:
    print(f"⚠️  Missing tables: {missing}")
    print("Run notebooks 01-04 first.")
else:
    print(f"✓ All {len(expected_tables)} expected tables are present.")

# COMMAND ----------

# UC SQL functions Genie should call as tools
expected_functions = ["product_from_raw", "raw_from_product", "revenue_risk"]

actual_functions = {row.function_name for row in spark.sql(
    f"SHOW USER FUNCTIONS IN {catalog_name}.{db_name}"
).collect()}
# Function names sometimes come back qualified
actual_function_short = {f.split(".")[-1] for f in actual_functions}
missing_fns = sorted(set(expected_functions) - actual_function_short)

if missing_fns:
    print(f"⚠️  Missing UC SQL functions: {missing_fns}")
    print("Run notebook 05 first.")
else:
    print(f"✓ All {len(expected_functions)} UC SQL functions are registered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Add table-level comments to sharpen Genie grounding
# MAGIC
# MAGIC Genie's accuracy improves substantially when tables and columns have semantic comments. The function COMMENT clauses are already set in notebook 05; this step adds comments to the source and derived tables.

# COMMAND ----------

table_comments = {
    "product_demand_historical": "Historical weekly demand by product and wholesaler. Columns: product, wholesaler, ds (week date), y (units).",
    "product_demand_forecasted": "Forecasted one-week-ahead demand per (product, wholesaler) produced by MMF + Chronos-2. Columns: product, wholesaler, ds, y_forecast (mean), and quantile columns.",
    "bom": "Bill of materials. Hierarchical relationship: raw_material → primary_material → intermediate_material → product. Columns: material_in, material_out, qty.",
    "raw_material_demand": "Aggregated demand for each raw material derived by BOM traversal from product forecasts. Columns: RAW, Demand_Raw.",
    "raw_material_supply": "Available supply per raw material. May include synthetic shortage scenarios. Columns: raw, supply.",
    "shipment_recommendations": "Least-cost shipment plan from a linear program. One row per (product, plant, distribution_center) with qty_shipped.",
    "list_prices": "List price per finished product. Columns: product, price.",
    "distribution_center_to_wholesaler_mapping": "Which distribution centers serve which wholesalers. Columns: distribution_center, wholesaler.",
    "plant_supply": "Per-plant supply capacity for each product.",
    "transport_cost": "Per-unit transport cost between plants and distribution centers.",
}

for table, comment in table_comments.items():
    try:
        # Escape single quotes in the comment
        safe = comment.replace("'", "''")
        spark.sql(f"COMMENT ON TABLE {catalog_name}.{db_name}.{table} IS '{safe}'")
        print(f"✓ {table}")
    except Exception as e:
        print(f"⚠️  {table}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Curated seed questions
# MAGIC
# MAGIC The booth narrative and depth questions from `genie_seed_questions.md`, ready to load into the Genie Space.

# COMMAND ----------

booth_narrative = [
    "Which raw materials are short next week?",
    "Of those raw materials, which one puts the most revenue at risk?",
    "If we can't source enough of that material, which finished products are affected?",
    "How much weekly revenue does that represent?",
    "Where should we ship the available supply to minimize loss?",
]

extended_depth = [
    "What raw materials does syringe_1 need to be produced?",
    "Show me the demand forecast for syringe_1 over the next 4 weeks.",
    "How does next week's forecasted demand compare to the same week last year?",
    "Which distribution centers serve the wholesalers most affected by these shortages?",
    "Give me a one-paragraph executive summary of next week's supply-chain risk.",
]

all_seed_questions = booth_narrative + extended_depth
print(f"Booth narrative: {len(booth_narrative)} questions")
print(f"Extended depth: {len(extended_depth)} questions")
print(f"Total seed: {len(all_seed_questions)} questions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Provision the Genie Space
# MAGIC
# MAGIC The Databricks Genie management API is evolving. The cleanest reliable path today is a small REST call against the Workspace API; if the API call fails or the endpoint isn't yet available in your workspace, the cell below prints the manual UI steps to complete the setup.

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

w = WorkspaceClient()
workspace_url = w.config.host

if not warehouse_id:
    print("⚠️  warehouse_id widget is empty. Falling back to manual UI steps below.")

# COMMAND ----------

# Try the Genie Space creation API. The endpoint and payload below match the documented
# 2.0 surface as of 2026; if your workspace is on an older runtime that hasn't exposed
# this endpoint yet, the call will raise and we fall through to manual steps.

space_payload = {
    "title": space_title,
    "description": (
        "Supply-chain demand and shortage-risk analysis. Backed by Databricks MMF + "
        "Chronos-2 forecasts and three UC SQL functions: product_from_raw, "
        "raw_from_product, revenue_risk. Designed for the DAIS 2026 AWS booth demo."
    ),
    "warehouse_id": warehouse_id,
    "tables": [
        f"{catalog_name}.{db_name}.{t}" for t in expected_tables
    ],
    "functions": [
        f"{catalog_name}.{db_name}.{fn}" for fn in expected_functions
    ],
    "sample_queries": [
        {"question": q} for q in all_seed_questions
    ],
}

space_url = None
try:
    response = w.api_client.do(
        method="POST",
        path="/api/2.0/genie/spaces",
        body=space_payload,
    )
    space_id = response.get("space_id") or response.get("id")
    if space_id:
        space_url = f"{workspace_url}/genie/rooms/{space_id}"
        print(f"✓ Genie Space created: {space_url}")
    else:
        print(f"Response did not include space_id. Raw response: {json.dumps(response)[:500]}")
except DatabricksError as e:
    print(f"⚠️  Programmatic Genie Space creation not available in this workspace: {e}")
    print("Falling through to manual UI steps below.")
except Exception as e:
    print(f"⚠️  Unexpected error: {e}")
    print("Falling through to manual UI steps below.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Manual UI steps (fallback if the API call above did not succeed)
# MAGIC
# MAGIC The Genie Space configuration is also doable via the Databricks UI in ~5-10 minutes:
# MAGIC
# MAGIC 1. **Navigate to Genie** in the left nav of your Databricks workspace.
# MAGIC 2. **Create a new Space** with the title configured by the widget.
# MAGIC 3. **Attach a SQL Warehouse** — Pro or Serverless. Get the warehouse_id from `SQL → Warehouses` if you don't have it.
# MAGIC 4. **Add tables** — all 13 listed in `expected_tables` above, fully qualified with your catalog + schema.
# MAGIC 5. **Add functions** — the 3 UC SQL functions registered by notebook 05.
# MAGIC 6. **Add sample questions** — paste the 10 questions from the `booth_narrative` + `extended_depth` lists above. Sample questions are the single highest-leverage configuration step for Genie reliability.
# MAGIC 7. **Test** — run the booth narrative end-to-end and confirm Genie picks the right tables / functions for each question.
# MAGIC
# MAGIC See `genie_seed_questions.md` in the repo for the full question catalog and the mapping from each question to the UC SQL function it should invoke.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - Share the Genie Space URL with the booth team.
# MAGIC - Rehearse the 5-question booth narrative end-to-end. If any question lands inconsistently, edit it in the Space's sample-questions list — Genie's grounding improves with more examples covering the same intent in different phrasings.
# MAGIC - For AWS-side agent extensions (Bedrock AgentCore, Amazon Q), see `07_Bedrock_AgentCore_Integration.md`.
