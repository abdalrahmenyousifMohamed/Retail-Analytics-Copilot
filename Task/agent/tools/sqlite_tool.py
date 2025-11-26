import sqlite3
from typing import Dict, List, Optional, Tuple

class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_schema(self) -> str:
        """Get database schema for SQL generation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_parts = []
        for (table_name,) in tables:
            # Quote table names that have spaces or special characters
            quoted_name = f'"{table_name}"' if ' ' in table_name else table_name
            cursor.execute(f"PRAGMA table_info({quoted_name});")
            columns = cursor.fetchall()
            cols_str = ", ".join([f"{col[1]} {col[2]}" for col in columns])
            # Use the original table name in schema (with quotes if needed)
            display_name = f'"{table_name}"' if ' ' in table_name else table_name
            schema_parts.append(f"{display_name}({cols_str})")
        
        conn.close()
        return "; ".join(schema_parts)
    
    def execute(self, query: str) -> Dict:
        """Execute SQL and return results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            conn.close()
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e)
            }
