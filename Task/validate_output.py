"""
Validation script for outputs_hybrid.jsonl
Checks that output matches the contract specification
"""
import json
import sys

def validate_output(output_file: str):
    """Validate the output JSONL file."""
    
    print(f"🔍 Validating {output_file}...")
    print("-" * 60)
    
    required_fields = ["id", "final_answer", "sql", "confidence", "explanation", "citations"]
    
    # Expected format hints from sample questions
    format_hints = {
        "rag_policy_beverages_return_days": "int",
        "hybrid_top_category_qty_summer_1997": "{category:str, quantity:int}",
        "hybrid_aov_winter_1997": "float",
        "sql_top3_products_by_revenue_alltime": "list[{product:str, revenue:float}]",
        "hybrid_revenue_beverages_summer_1997": "float",
        "hybrid_best_customer_margin_1997": "{customer:str, margin:float}"
    }
    
    issues = []
    successes = 0
    
    try:
        with open(output_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    
                    # Check required fields
                    missing = [f for f in required_fields if f not in record]
                    if missing:
                        issues.append(f"Line {line_num}: Missing fields: {missing}")
                        continue
                    
                    # Validate types
                    question_id = record["id"]
                    expected_format = format_hints.get(question_id, "unknown")
                    
                    # Check final_answer type
                    answer = record["final_answer"]
                    if expected_format == "int":
                        if not isinstance(answer, int):
                            issues.append(f"{question_id}: Expected int, got {type(answer).__name__}")
                    
                    elif expected_format == "float":
                        if not isinstance(answer, (int, float)):
                            issues.append(f"{question_id}: Expected float, got {type(answer).__name__}")
                        elif isinstance(answer, float):
                            # Check 2 decimal places
                            if len(str(answer).split('.')[-1]) > 2:
                                issues.append(f"{question_id}: Float should have max 2 decimals, got {answer}")
                    
                    elif expected_format.startswith("{"):
                        if not isinstance(answer, dict):
                            issues.append(f"{question_id}: Expected dict, got {type(answer).__name__}")
                        else:
                            # Extract expected keys
                            import re
                            keys = re.findall(r'(\w+):', expected_format)
                            for key in keys:
                                if key not in answer:
                                    issues.append(f"{question_id}: Missing key '{key}' in dict")
                    
                    elif expected_format.startswith("list"):
                        if not isinstance(answer, list):
                            issues.append(f"{question_id}: Expected list, got {type(answer).__name__}")
                        elif len(answer) > 0 and not isinstance(answer[0], dict):
                            issues.append(f"{question_id}: Expected list of dicts")
                    
                    # Check sql field
                    if not isinstance(record["sql"], str):
                        issues.append(f"{question_id}: SQL must be string, got {type(record['sql']).__name__}")
                    
                    # Check confidence
                    conf = record["confidence"]
                    if not isinstance(conf, (int, float)):
                        issues.append(f"{question_id}: Confidence must be numeric")
                    elif not (0.0 <= conf <= 1.0):
                        issues.append(f"{question_id}: Confidence must be 0.0-1.0, got {conf}")
                    
                    # Check explanation length (<=2 sentences approximation)
                    explanation = record["explanation"]
                    if not isinstance(explanation, str):
                        issues.append(f"{question_id}: Explanation must be string")
                    elif len(explanation.split('.')) > 3:  # Rough check for 2 sentences
                        issues.append(f"{question_id}: Explanation too long (should be <=2 sentences)")
                    
                    # Check citations
                    citations = record["citations"]
                    if not isinstance(citations, list):
                        issues.append(f"{question_id}: Citations must be list")
                    else:
                        # Valid DB tables
                        valid_tables = ["Orders", "Order Details", "Products", "Customers", 
                                       "Categories", "Suppliers", "Employees", "Shippers"]
                        # Valid doc prefixes
                        valid_docs = ["marketing_calendar", "kpi_definitions", "catalog", "product_policy"]
                        
                        for citation in citations:
                            if not isinstance(citation, str):
                                issues.append(f"{question_id}: Citation must be string, got {citation}")
                            elif citation in valid_tables:
                                pass  # Valid table
                            elif any(citation.startswith(doc + "::") for doc in valid_docs):
                                pass  # Valid doc chunk
                            else:
                                issues.append(f"{question_id}: Invalid citation format: {citation}")
                    
                    if not issues or line_num > successes:
                        successes += 1
                        print(f"✅ {question_id}: Valid")
                
                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: Invalid JSON - {e}")
        
        print("-" * 60)
        print(f"\n📊 Results:")
        print(f"   Successes: {successes}/6")
        print(f"   Issues: {len(issues)}")
        
        if issues:
            print("\n❌ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("\n✅ All validations passed!")
            return True
    
    except FileNotFoundError:
        print(f"❌ Error: File '{output_file}' not found")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_output.py outputs_hybrid.jsonl")
        sys.exit(1)
    
    success = validate_output(sys.argv[1])
    sys.exit(0 if success else 1)
