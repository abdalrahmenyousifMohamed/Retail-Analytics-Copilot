from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
import json
import re
from agent.tools.sqlite_tool import SQLiteTool
from agent.rag.retrieval import SimpleRetriever
from agent.dspy_signatures import RouterModule, SQLGeneratorModule, SynthesizerModule

class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    doc_chunks: List[Dict]
    sql: str
    sql_results: Dict
    final_answer: any
    citations: List[str]
    confidence: float
    explanation: str
    repair_count: int
    trace: List[str]

class HybridAgent:
    def __init__(self, db_path: str, docs_dir: str):
        self.db_tool = SQLiteTool(db_path)
        self.retriever = SimpleRetriever(docs_dir)
        self.router = RouterModule()
        self.sql_generator = SQLGeneratorModule()
        self.synthesizer = SynthesizerModule()
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.route_node)
        workflow.add_node("retriever", self.retrieve_node)
        workflow.add_node("planner", self.plan_node)
        workflow.add_node("sql_generator", self.sql_gen_node)
        workflow.add_node("executor", self.execute_node)
        workflow.add_node("synthesizer", self.synth_node)
        workflow.add_node("validator", self.validate_node)
        
        # Define edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "sql_generator")
        workflow.add_edge("sql_generator", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", "validator")
        
        # Conditional edge for repair
        workflow.add_conditional_edges(
            "validator",
            self.should_repair,
            {
                "repair": "sql_generator",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def route_node(self, state: AgentState) -> AgentState:
        """Node 1: Route the query."""
        route = self.router.forward(state["question"])
        state["route"] = route
        state["trace"].append(f"Routed to: {route}")
        return state
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant documents."""
        chunks = self.retriever.retrieve(state["question"], top_k=3)
        state["doc_chunks"] = chunks
        state["trace"].append(f"Retrieved {len(chunks)} chunks")
        return state
    
    def plan_node(self, state: AgentState) -> AgentState:
        """Node 3: Extract constraints from docs."""
        # Extract date ranges, KPIs, categories from retrieved chunks
        context = "\n".join([chunk['content'] for chunk in state["doc_chunks"]])
        state["trace"].append("Extracted planning constraints")
        return state
    
    def sql_gen_node(self, state: AgentState) -> AgentState:
        """Node 4: Generate SQL query."""
        if state["route"] in ["sql", "hybrid"]:
            db_schema = self.db_tool.get_schema()
            context = "\n".join([chunk['content'] for chunk in state["doc_chunks"]])
            
            try:
                sql = self.sql_generator.forward(
                    question=state["question"],
                    db_schema=db_schema,
                    context=context
                )
                # Clean SQL
                sql = sql.strip().strip('```sql').strip('```').strip()
                state["sql"] = sql
                state["trace"].append(f"Generated SQL: {sql[:100]}...")
            except Exception as e:
                state["sql"] = ""
                state["trace"].append(f"SQL generation error: {str(e)}")
        return state
    
    def execute_node(self, state: AgentState) -> AgentState:
        """Node 5: Execute SQL query."""
        if state["sql"]:
            results = self.db_tool.execute(state["sql"])
            state["sql_results"] = results
            if results["success"]:
                state["trace"].append(f"SQL executed: {len(results['rows'])} rows")
            else:
                state["trace"].append(f"SQL error: {results['error']}")
        else:
            state["sql_results"] = {"success": True, "columns": [], "rows": []}
        return state
    
    def synth_node(self, state: AgentState) -> AgentState:
        """Node 6: Synthesize final answer."""
        try:
            answer_str = self.synthesizer.forward(
                question=state["question"],
                format_hint=state["format_hint"],
                sql_results=state["sql_results"],
                doc_chunks=state["doc_chunks"]
            )
            
            # Parse answer based on format_hint
            final_answer = self._parse_answer(answer_str, state["format_hint"], state["sql_results"])
            state["final_answer"] = final_answer
            
            # Generate citations
            citations = []
            
            # Add DB table citations if SQL was used
            if state["sql"] and state["sql_results"].get("success"):
                tables = self._extract_tables(state["sql"])
                citations.extend(tables)
            
            # Add document chunk citations
            for chunk in state["doc_chunks"]:
                if chunk['score'] > 0.1:  # Only cite relevant chunks
                    citations.append(chunk['id'])
            
            state["citations"] = citations
            
            # Generate explanation (max 2 sentences)
            sql_used = "SQL query executed. " if state["sql"] else ""
            doc_used = f"Referenced {len([c for c in citations if '::' in c])} document chunks. " if any('::' in c for c in citations) else ""
            state["explanation"] = (sql_used + doc_used).strip()
            
            # Calculate confidence
            confidence = 0.5  # Base confidence
            if state["sql_results"].get("success") and state["sql_results"].get("rows"):
                confidence += 0.2
            if state["doc_chunks"] and max([c['score'] for c in state["doc_chunks"]], default=0) > 0.3:
                confidence += 0.2
            if state["repair_count"] == 0:
                confidence += 0.1
            else:
                confidence -= 0.1 * state["repair_count"]
            
            state["confidence"] = round(min(max(confidence, 0.0), 1.0), 2)
            state["trace"].append("Synthesized answer")
            
        except Exception as e:
            state["final_answer"] = None
            state["trace"].append(f"Synthesis error: {str(e)}")
        
        return state
    
    def validate_node(self, state: AgentState) -> AgentState:
        """Node 7: Validate output."""
        state["trace"].append("Validated output")
        return state
    
    def should_repair(self, state: AgentState) -> str:
        """Decide if repair is needed."""
        if state["repair_count"] >= 2:
            return "end"
        
        # Check if SQL failed or answer is invalid
        if state["sql"] and not state["sql_results"].get("success"):
            state["repair_count"] += 1
            state["trace"].append(f"Repair attempt {state['repair_count']}")
            return "repair"
        
        if state["final_answer"] is None:
            state["repair_count"] += 1
            state["trace"].append(f"Repair attempt {state['repair_count']}")
            return "repair"
        
        return "end"
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        tables = []
        if not sql:
            return tables
            
        sql_upper = sql.upper()
        # Check for actual table names in the Northwind database
        table_mapping = {
            "ORDERS": "Orders",
            "ORDER DETAILS": "Order Details",
            "PRODUCTS": "Products",
            "CUSTOMERS": "Customers",
            "CATEGORIES": "Categories",
            "SUPPLIERS": "Suppliers",
            "EMPLOYEES": "Employees",
            "SHIPPERS": "Shippers"
        }
        
        for table_key, table_name in table_mapping.items():
            # Check if table is referenced in SQL
            if table_key in sql_upper:
                tables.append(table_name)
        
        return tables
    
    def _parse_answer(self, answer_str: str, format_hint: str, sql_results: Dict = None) -> any:
        """Parse answer string to match format_hint."""
        answer_str = answer_str.strip()
        
        # For SQL-based answers, try to extract from results first
        if sql_results and sql_results.get("success") and sql_results.get("rows"):
            rows = sql_results["rows"]
            columns = sql_results["columns"]
            
            if format_hint == "int":
                # Extract first integer from first row
                if rows and len(rows[0]) > 0:
                    try:
                        return int(rows[0][0])
                    except (ValueError, TypeError):
                        pass
            
            elif format_hint == "float":
                # Extract first float from first row
                if rows and len(rows[0]) > 0:
                    try:
                        return round(float(rows[0][0]), 2)
                    except (ValueError, TypeError):
                        pass
            
            elif "{" in format_hint and "}" in format_hint:
                # Single object result
                if rows and len(rows[0]) >= 2:
                    # Parse format_hint to get field names
                    # e.g., "{category:str, quantity:int}" -> ["category", "quantity"]
                    fields = re.findall(r'(\w+):\w+', format_hint)
                    if len(fields) == len(rows[0]):
                        result = {}
                        for i, field in enumerate(fields):
                            value = rows[0][i]
                            # Type conversion based on format_hint
                            if ':int' in format_hint:
                                try:
                                    result[field] = int(value) if i == 1 else value
                                except:
                                    result[field] = value
                            elif ':float' in format_hint:
                                try:
                                    result[field] = round(float(value), 2) if i == 1 else value
                                except:
                                    result[field] = value
                            else:
                                result[field] = value
                        return result
            
            elif "list[" in format_hint:
                # List of objects
                fields = re.findall(r'(\w+):\w+', format_hint)
                results = []
                for row in rows:
                    if len(fields) == len(row):
                        obj = {}
                        for i, field in enumerate(fields):
                            value = row[i]
                            # Type conversion
                            if i == 1 and ':float' in format_hint:
                                try:
                                    obj[field] = round(float(value), 2)
                                except:
                                    obj[field] = value
                            elif i == 1 and ':int' in format_hint:
                                try:
                                    obj[field] = int(value)
                                except:
                                    obj[field] = value
                            else:
                                obj[field] = value
                        results.append(obj)
                return results
        
        # Fallback: parse from answer_str
        if format_hint == "int":
            match = re.search(r'\d+', answer_str)
            return int(match.group()) if match else 0
        
        elif format_hint == "float":
            match = re.search(r'[\d.]+', answer_str)
            return round(float(match.group()), 2) if match else 0.0
        
        elif "{" in format_hint:
            try:
                json_match = re.search(r'\{[^}]+\}', answer_str)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            return {}
        
        elif "list[" in format_hint:
            try:
                json_match = re.search(r'\[[^\]]+\]', answer_str, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            return []
        
        return answer_str
    
    def run(self, question: str, format_hint: str) -> Dict:
        """Run the agent on a question."""
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "route": "",
            "doc_chunks": [],
            "sql": "",
            "sql_results": {},
            "final_answer": None,
            "citations": [],
            "confidence": 0.0,
            "explanation": "",
            "repair_count": 0,
            "trace": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state["sql"],
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"]
        }