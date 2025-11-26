#!/bin/bash
# Complete setup script for Retail Analytics Copilot (OpenAI Version)

set -e

echo "🚀 Setting up Retail Analytics Copilot (OpenAI Version)..."
echo ""

# Check if OpenAI API key is provided
if [ -z "$1" ]; then
    echo "❌ Error: OpenAI API key required!"
    echo ""
    echo "Usage: ./setup.sh YOUR_OPENAI_API_KEY"
    echo ""
    echo "Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

OPENAI_API_KEY=$1

# Create directory structure
echo "📁 Creating project structure..."
mkdir -p retail_copilot/{agent/rag,agent/tools,data,docs}
cd retail_copilot

# Create .env file
echo "🔑 Creating .env file..."
cat > .env << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
EOF

# Create .gitignore
echo "🚫 Creating .gitignore..."
cat > .gitignore << 'EOF'
.env
venv/
__pycache__/
*.pyc
.DS_Store
outputs_*.jsonl
EOF

# Create docs files
echo "📄 Creating documentation files..."

cat > docs/marketing_calendar.md << 'EOF'
# Northwind Marketing Calendar (1997)

## Summer Beverages 1997
- Dates: 1997-06-01 to 1997-06-30
- Notes: Focus on Beverages and Condiments.

## Winter Classics 1997
- Dates: 1997-12-01 to 1997-12-31
- Notes: Push Dairy Products and Confections for holiday gifting.
EOF

cat > docs/kpi_definitions.md << 'EOF'
# KPI Definitions

## Average Order Value (AOV)
- AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)

## Gross Margin
- GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))
- If cost is missing, approximate with category-level average (document your approach).
EOF

cat > docs/catalog.md << 'EOF'
# Catalog Snapshot

- Categories include Beverages, Condiments, Confections, Dairy Products, Grains/Cereals, Meat/Poultry, Produce, Seafood.
- Products map to categories as in the Northwind DB.
EOF

cat > docs/product_policy.md << 'EOF'
# Returns & Policy

- Perishables (Produce, Seafood, Dairy): 3–7 days.
- Beverages unopened: 14 days; opened: no returns.
- Non-perishables: 30 days.
EOF

# Create eval file
echo "📝 Creating evaluation questions..."
cat > sample_questions_hybrid_eval.jsonl << 'EOF'
{"id":"rag_policy_beverages_return_days","question":"According to the product policy, what is the return window (days) for unopened Beverages? Return an integer.","format_hint":"int"}
{"id":"hybrid_top_category_qty_summer_1997","question":"During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}.","format_hint":"{category:str, quantity:int}"}
{"id":"hybrid_aov_winter_1997","question":"Using the AOV definition from the KPI docs, what was the Average Order Value during 'Winter Classics 1997'? Return a float rounded to 2 decimals.","format_hint":"float"}
{"id":"sql_top3_products_by_revenue_alltime","question":"Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}].","format_hint":"list[{product:str, revenue:float}]"}
{"id":"hybrid_revenue_beverages_summer_1997","question":"Total revenue from the 'Beverages' category during 'Summer Beverages 1997' dates. Return a float rounded to 2 decimals.","format_hint":"float"}
{"id":"hybrid_best_customer_margin_1997","question":"Per the KPI definition of gross margin, who was the top customer by gross margin in 1997? Assume CostOfGoods is approximated by 70% of UnitPrice if not available. Return {customer:str, margin:float}.","format_hint":"{customer:str, margin:float}"}
EOF

# Download database
echo "💾 Downloading Northwind database..."
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Create lowercase views (optional)
echo "🔧 Creating database views..."
sqlite3 data/northwind.sqlite <<'SQL'
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
SQL

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
dspy-ai>=2.4.0
langgraph>=0.1.0
langchain-core>=0.2.0
pydantic>=2.0.0
click>=8.1.7
rich>=13.7.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.3.0
rank-bm25>=0.2.2
openai>=1.0.0
python-dotenv>=1.0.0
EOF

# Create __init__.py files
touch agent/__init__.py
touch agent/rag/__init__.py
touch agent/tools/__init__.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create Python files from the provided code artifact"
echo "2. Create a virtual environment:"
echo "   python -m venv venv"
echo ""
echo "3. Activate it:"
echo "   source venv/bin/activate   # Linux/Mac"
echo "   venv\\Scripts\\activate     # Windows"
echo ""
echo "4. Install dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "5. Copy the Python code into:"
echo "   - agent/tools/sqlite_tool.py"
echo "   - agent/rag/retrieval.py"
echo "   - agent/dspy_signatures.py"
echo "   - agent/graph_hybrid.py"
echo "   - run_agent_hybrid.py"
echo ""
echo "6. Run the agent:"
echo "   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl"
echo ""
echo "💡 Your .env file has been created with your API key"
echo "⚠️  IMPORTANT: Never commit .env to git!"
echo ""