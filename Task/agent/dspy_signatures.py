import dspy
from typing import Literal

class RouteQuery(dspy.Signature):
    """Classify if query needs RAG, SQL, or both."""
    question = dspy.InputField(desc="User question")
    route = dspy.OutputField(desc="One of: rag, sql, hybrid")

class GenerateSQL(dspy.Signature):
    """Generate SQL query from natural language."""
    question = dspy.InputField(desc="User question")
    db_schema = dspy.InputField(desc="Database schema")
    context = dspy.InputField(desc="Relevant document context")
    sql = dspy.OutputField(desc="Valid SQLite query")

class SynthesizeAnswer(dspy.Signature):
    """Synthesize final answer with citations."""
    question = dspy.InputField(desc="User question")
    format_hint = dspy.InputField(desc="Expected output format")
    sql_results = dspy.InputField(desc="SQL execution results")
    doc_chunks = dspy.InputField(desc="Retrieved document chunks")
    answer = dspy.OutputField(desc="Final answer matching format_hint")


class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouteQuery)
    
    def forward(self, question):
        result = self.classify(question=question)
        route = result.route.lower().strip()
        if route not in ['rag', 'sql', 'hybrid']:
            route = 'hybrid'
        return route

class SQLGeneratorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateSQL)
    
    def forward(self, question, db_schema, context=""):
        result = self.generate(question=question, db_schema=db_schema, context=context)
        return result.sql

class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question, format_hint, sql_results, doc_chunks):
        result = self.synthesize(
            question=question,
            format_hint=format_hint,
            sql_results=str(sql_results),
            doc_chunks=str(doc_chunks)
        )
        return result.answer
