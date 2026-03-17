from mcp.server.fastmcp import FastMCP
import pandas as pd
from typing import Optional
import json
from rag_agent.rag_service import answer_question, index_uploaded_pdfs
from sql_agent.agent import generate_and_execute
from sql_agent.db import run_query

mcp = FastMCP(
    name="Multi-Agent MCP Server",
    host="0.0.0.0", 
    port=8000,  
)

@mcp.tool()
def sql_query(question: Optional[str] = None, query: Optional[str] = None) -> str:
    """Execute SQL queries from natural language or raw SQL"""
    
    try:
        if not question and not query:
            return json.dumps({"error": "Provide either 'question' or 'query'"})
        
        if question:
            sql, result = generate_and_execute(question)
        else:
            sql = query
            result = run_query(sql)
        
        if isinstance(result, pd.DataFrame):
            return json.dumps({
                "sql": sql,
                "columns": list(result.columns),
                "rows": result.fillna("").values.tolist(),
                "meta": {"row_count": len(result)}
            })
        else:
            return json.dumps({"sql": sql, "error": str(result)})
            
    except Exception as e:
        return json.dumps({"error": str(e)})
    
@mcp.tool()
def document_search(query: str, top_k: int = 5) -> str:
    """Search documents for relevant information based on a query"""
    try:
        answer, sources = answer_question(query)

        return json.dumps({
            "question": query,
            "answer": answer,
            "sources": sources,
            "top_k": top_k
        })
    
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def index_pdfs(paths: list) -> dict:
    try:
        count = index_uploaded_pdfs(paths)
        return {"indexed": count}
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    print("Running server with SSE transport")
    mcp.run(transport="sse")
    # mcp.run()