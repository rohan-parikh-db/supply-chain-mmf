# Genie seed questions — supply-chain-mmf

Curated example questions for the Genie Space scoped to the `<catalog>.supply_chain_mmf` schema. Use these when configuring the Genie Space — examples are the single highest-leverage way to make Genie reliable for repeat demos.

The first five form a **linear 90-second booth narrative**: situation → rank → propagate → quantify → mitigate. The remaining questions support deeper one-on-one customer conversations.

## Booth narrative (rehearsed, 5 questions, ~90s)

1. **Situation awareness** — *"Which raw materials are short next week?"*
2. **Impact ranking** — *"Of those raw materials, which one puts the most revenue at risk?"*
3. **Downstream propagation** — *"If we can't source enough of that material, which finished products are affected?"*
4. **Quantification** — *"How much weekly revenue does that represent?"*
5. **Mitigation** — *"Where should we ship the available supply to minimize loss?"*

## Extended depth (one-on-one conversations)

6. **Diagnostic drill-down** — *"What raw materials does syringe_1 need to be produced?"*
7. **Forecast inspection** — *"Show me the demand forecast for syringe_1 over the next 4 weeks."*
8. **Comparative analysis** — *"How does next week's forecasted demand compare to the same week last year?"*
9. **Vendor / sourcing reach** — *"Which distribution centers serve the wholesalers most affected by these shortages?"*
10. **Executive summary** — *"Give me a one-paragraph executive summary of next week's supply-chain risk."*

## Mapping to UC SQL functions

| Question theme | UC SQL function invoked | Notes |
|---|---|---|
| 1 | (none — direct SQL on `raw_material_demand` + `raw_material_supply`) | Genie writes the SQL |
| 2 | `revenue_risk(raw_material)` | One call per candidate; Genie ranks |
| 3 | `product_from_raw(raw_material)` | Returns affected products + per-step quantities |
| 4 | `revenue_risk(raw_material)` | Sum over returned rows |
| 5 | (direct SQL on `shipment_recommendations`) | Filtered by affected products |
| 6 | `raw_from_product(product)` | BOM traversal upstream |
| 7 | (direct SQL on `product_demand_forecasted`) | Time-series chart |
| 8 | (direct SQL joining `product_demand_forecasted` + `product_demand_historical`) | Year-over-year compare |
| 9 | (direct SQL joining `distribution_center_to_wholesaler_mapping`) | Multi-table join |
| 10 | (Genie summary mode over the underlying data) | Free-form synthesis |

## Notes for booth operators

- **Genie may ask for clarification** on which raw material to use in questions 2-5. The fastest path is to pin the answer from question 1 in the conversation and reference it.
- **For booth time pressure**, pre-populate the catalog/schema names in the Genie Space defaults so Genie doesn't ask for them.
- **If a question times out or returns unexpected results**, fall back to question 6 (`raw_from_product`) which has the highest grounding (BOM is deterministic, no forecast variance).
