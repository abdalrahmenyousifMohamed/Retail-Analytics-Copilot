# Pre-Submission Checklist

## ✅ Required Files

- [ ] `agent/graph_hybrid.py` - LangGraph with ≥6 nodes + repair loop
- [ ] `agent/dspy_signatures.py` - DSPy modules (Router, SQL Gen, Synthesizer)
- [ ] `agent/rag/retrieval.py` - TF-IDF/BM25 retriever
- [ ] `agent/tools/sqlite_tool.py` - SQLite database access
- [ ] `run_agent_hybrid.py` - CLI entrypoint (exact flags)
- [ ] `requirements.txt` - All dependencies
- [ ] `README.md` - Design + DSPy optimization + assumptions
- [ ] `outputs_hybrid.jsonl` - Generated outputs (6 lines)
- [ ] `.gitignore` - Includes `.env` and `venv/`

## ✅ Data Files

- [ ] `data/northwind.sqlite` - Downloaded Northwind database
- [ ] `docs/marketing_calendar.md` - Marketing dates (Summer/Winter 1997)
- [ ] `docs/kpi_definitions.md` - AOV and Gross Margin formulas
- [ ] `docs/catalog.md` - Product categories
- [ ] `docs/product_policy.md` - Return policies
- [ ] `sample_questions_hybrid_eval.jsonl` - 6 evaluation questions

## ✅ Code Requirements

### LangGraph (20 points)
- [ ] Minimum 6 nodes (you have 7: router, retriever, planner, sql_gen, executor, synthesizer, validator)
- [ ] Repair loop implemented (max 2 iterations)
- [ ] Stateful execution with proper state typing
- [ ] Conditional edges for repair routing

### DSPy (20 points)
- [ ] At least one optimized module documented
- [ ] ChainOfThought or other DSPy module used
- [ ] Before/after metrics shown in README
- [ ] Clear improvement demonstrated

### Output Contract (40 points)
- [ ] All 6 questions processed
- [ ] `final_answer` matches `format_hint` type exactly
- [ ] `sql` field contains query or empty string
- [ ] `confidence` is float between 0.0-1.0
- [ ] `explanation` is ≤2 sentences
- [ ] `citations` include all used DB tables
- [ ] `citations` include all relevant doc chunks (format: `filename::chunkN`)

### Code Quality (20 points)
- [ ] Readable, well-structured code
- [ ] Proper error handling
- [ ] README with design decisions
- [ ] No hardcoded answers

## ✅ Output Validation

Run the validator:
```bash
python validate_output.py outputs_hybrid.jsonl
```

Expected: All 6 questions pass validation

## ✅ Type Checking

### Question 1: RAG only (int)
```json
{"id":"rag_policy_beverages_return_days","final_answer":14,...}
```
- [ ] Answer is integer (not string "14")
- [ ] SQL field is empty string
- [ ] Citations include `product_policy::chunk0` or similar

### Question 2: Hybrid (dict)
```json
{"id":"hybrid_top_category_qty_summer_1997","final_answer":{"category":"Beverages","quantity":455},...}
```
- [ ] Answer is dict with "category" (string) and "quantity" (int)
- [ ] SQL field contains query
- [ ] Citations include: Orders, Order Details, Products, Categories
- [ ] Citations include: marketing_calendar::chunk1 or similar

### Question 3: Hybrid (float)
```json
{"id":"hybrid_aov_winter_1997","final_answer":856.19,...}
```
- [ ] Answer is float rounded to 2 decimals
- [ ] SQL uses AOV formula from docs
- [ ] Citations include: Orders, Order Details, kpi_definitions

### Question 4: SQL only (list)
```json
{"id":"sql_top3_products_by_revenue_alltime","final_answer":[{"product":"...","revenue":123.45}],...}
```
- [ ] Answer is list of dicts
- [ ] Each dict has "product" (string) and "revenue" (float, 2 decimals)
- [ ] Exactly 3 items in list
- [ ] SQL contains ORDER BY ... DESC LIMIT 3

### Question 5: Hybrid (float)
```json
{"id":"hybrid_revenue_beverages_summer_1997","final_answer":3459.50,...}
```
- [ ] Answer is float rounded to 2 decimals
- [ ] SQL filters by category AND date range
- [ ] Citations include: marketing_calendar, Categories

### Question 6: Hybrid (dict)
```json
{"id":"hybrid_best_customer_margin_1997","final_answer":{"customer":"...","margin":12345.67},...}
```
- [ ] Answer is dict with "customer" (string) and "margin" (float)
- [ ] SQL uses 0.3 * UnitPrice (because CostOfGoods = 0.7 * UnitPrice)
- [ ] Citations include: Customers, kpi_definitions

## ✅ README Content

- [ ] 2-4 bullet points explaining graph design
- [ ] DSPy module optimized (which one?)
- [ ] Metric improvement shown (before → after)
- [ ] CostOfGoods approximation documented (0.7 * UnitPrice)
- [ ] Any other assumptions clearly stated

## ✅ CLI Contract

Test the exact command:
```bash
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

- [ ] Command runs without errors
- [ ] Outputs 6 lines to `outputs_hybrid.jsonl`
- [ ] Each line is valid JSON
- [ ] No extra flags required

## ✅ Common Issues to Check

### SQL Issues
- [ ] Table names with spaces quoted: `"Order Details"`
- [ ] Date filtering uses BETWEEN or strftime
- [ ] Revenue formula: `SUM(UnitPrice * Quantity * (1 - Discount))`
- [ ] Joins are correct (Orders → Order Details → Products)

### Citation Issues
- [ ] Table names match database exactly (case-sensitive)
- [ ] "Orders" not "orders"
- [ ] "Order Details" not "OrderDetails"
- [ ] Doc chunks format: `filename::chunk0` (no .md extension)

### Type Issues
- [ ] Integers are not strings: `14` not `"14"`
- [ ] Floats rounded to 2 decimals: `123.45` not `123.456789`
- [ ] Dicts use correct keys from format_hint
- [ ] Lists contain dicts, not tuples or strings

### Confidence Issues
- [ ] Value is between 0.0 and 1.0
- [ ] Higher confidence for successful SQL + good retrieval
- [ ] Lower confidence for repairs or RAG-only

## ✅ Final Checks

- [ ] Run full pipeline end-to-end successfully
- [ ] All 6 questions produce output
- [ ] No Python exceptions or crashes
- [ ] Output file has exactly 6 lines
- [ ] Each line is valid JSON
- [ ] Validator script passes
- [ ] README is clear and concise
- [ ] Code is commented where needed
- [ ] `.env` file NOT committed to git
- [ ] GitHub repo link ready to share

## 🚀 Submit When All Checked!

Expected time: ~60 seconds runtime, $0.08 API cost

Good luck! 🎉