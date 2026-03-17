import json
from typing import Any

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from agent_graph.state import AgentState
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv(override=True)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

from langchain_core.tools import StructuredTool
from pydantic import create_model
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

def mcp_tool_to_langchain(mcp_tool_info, session):
    """
    Wraps an MCP tool into a LangChain StructuredTool.
    """
    name = mcp_tool_info.name
    description = mcp_tool_info.description
    
    # We dynamically create the tool function
    async def tool_func(**kwargs):
        # The agent calls this, which forwards the call to the MCP server
        result = await session.call_tool(name, arguments=kwargs)
        
        # Handle the MCP result format
        if hasattr(result, "content"):
            text_content = []
            for item in result.content:
                if hasattr(item, "text"):
                    text_content.append(item.text)
            return "\n".join(text_content)
        return str(result)

    return StructuredTool.from_function(
        func=None,
        coroutine=tool_func,
        name=name,
        description=description
    )


# ============================================================================
# SUPERVISOR NODE
# ============================================================================

CLASSIFIER_PROMPT = """
You are a router.

If the question is about:
- sales
- customers
- orders
- products
- revenue
- database records

Return ONLY: SQL

If the question is about:
- medical research
- clinical trials
- treatments
- outcomes
- studies

Return ONLY: RAG

Question: {question}
"""

SQL_KEYWORDS = [
    "product",
    "products",
    "order",
    "orders",
    "customer",
    "customers",
    "employee",
    "employees",
    "supplier",
    "suppliers",
    "shipper",
    "sales",
    "revenue",
    "category",
    "categories",
]

RAG_KEYWORDS = [
    "trial",
    "study",
    "clinical",
    "treatment",
    "therapy",
    "disease",
    "patient",
    "brain",
    "tumor",
    "lassa",
    "fibromyalgia",
    "stroke",
    "pneumonia",
    "lupus",
    "hiv",
    "autism",
]

def supervisor_node(state: AgentState) -> AgentState:
    """
    Route incoming questions to SQL or RAG agent based on keywords and LLM.

    Args:
        state: Current agent state

    Returns:
        Updated state with routing decision
    """
    question = state["question"].lower()

    # Rule-based routing first
    if any(keyword in question for keyword in SQL_KEYWORDS):
        route = "sql"
    elif any(keyword in question for keyword in RAG_KEYWORDS):
        route = "rag"
    else:
        # LLM fallback for ambiguous cases
        prompt = CLASSIFIER_PROMPT.format(question=question)
        decision = llm.invoke(prompt).content.lower()
        route = "sql" if "sql" in decision else "rag"

    return {**state, "route": route}


# ============================================================================
# GUARDRAIL NODE
# ============================================================================

GUARDRAIL_PROMPT = """
You are a safety classifier.

If the question is related to:
- databases
- SQL
- sales
- products
- customers
- biomedical research
- clinical trials
- medical treatments

Return: ALLOW

Otherwise return: BLOCK

Question: {question}
"""

def guardrail_node(state: AgentState) -> AgentState:
    """
    Check if query is within allowed domains.

    Args:
        state: Current agent state

    Returns:
        Updated state with guardrail decision, ends if query is blocked
    """
    question = state["question"]

    decision = llm.invoke(GUARDRAIL_PROMPT.format(question=question)).content.strip().lower()

    if "block" in decision:
        return {
            **state,
            "response": {
                "answer": "I can only help with database analytics and biomedical research questions."
            },
            "route": "end",
        }

    return state

# ============================================================================
# SQL AGENT NODE
# ============================================================================

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

async def sql_node(state: AgentState) -> AgentState:
    question = state["question"]
    messages = state.get("messages", [])
    if not messages:
        messages = [
            SystemMessage(content=(
                "You are an expert PostgreSQL Data Analyst. "
                "Your goal is to generate valid SQL queries to answer user questions. "
                "\n\n"
                "RULES:\n"
                "1. Always generate the SQL query yourself.\n"
                "2. Use the `sql_query` tool. Pass your SQL code to the `query` argument.\n"
                "3. Do NOT pass the natural language question to the tool.\n"
                "4. If you get an error (e.g., column not found), analyze the error and try a fixed SQL query.\n"
                "5. Once you have data, formulate a final natural language answer."
            )),
            HumanMessage(content=question)
        ]

    # Connect to MCP Server
    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # --- Tool Discovery & Binding ---
            tools_list = await session.list_tools()
            lc_tools = []
            tools_map = {}

            for tool_info in tools_list.tools:
                if tool_info.name == "sql_query":
                    tool_wrapper = mcp_tool_to_langchain(tool_info, session)
                    lc_tools.append(tool_wrapper)
                    tools_map[tool_info.name] = tool_wrapper
            
            # Bind tools so the LLM knows the schema (query vs question)
            llm_with_tools = llm.bind_tools(lc_tools)

            # --- The Reasoning Loop ---
            while True:
                # 1. Ask LLM what to do
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                # 2. Check if LLM is done (no tool calls)
                if not response.tool_calls:
                    return {
                        **state,
                        "messages": messages, # Preserve history
                        "response": {
                            "answer": response.content,
                            "sources": ["PostgreSQL Database"]
                        }
                    }

                # 3. Execute Tool Calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    if tool_name in tools_map:
                        # Call the MCP server
                        try:
                            # The result comes back as a JSON string from server
                            raw_result = await tools_map[tool_name].coroutine(**args)
                            
                            parsed_result = json.loads(raw_result)
                            
                            if "error" in parsed_result:
                                # Feed the error string back to the LLM
                                content = f"SQL Execution Error: {parsed_result['error']}"
                            else:
                                # Success! Feed the rows back
                                content = raw_result
                                
                        except Exception as e:
                            content = f"Tool Communication Error: {str(e)}"
                    else:
                        content = "Error: Tool not found."

                    # 4. Create Observation
                    tool_msg = ToolMessage(
                        tool_call_id=tool_call_id,
                        content=str(content),
                        name=tool_name
                    )
                    messages.append(tool_msg)
                

# ============================================================================
# RAG AGENT NODE
# ============================================================================

async def rag_node(state: AgentState) -> AgentState:

    question = state["question"]
    messages = state.get("messages", [])
    
    if not messages:
        messages = [
            SystemMessage(content=(
                "You are a RAG assistant. "
                "Use the document_search tool to find relevant context. "
                "After receiving tool results, extract the answer and respond clearly. "
                "If sources are empty, say you do not know."
            )),
            HumanMessage(content=question)
        ]

    async with sse_client("http://localhost:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_list = await session.list_tools()
            lc_tools = []
            tools_map = {}

            for tool_info in tools_list.tools:
                if tool_info.name == "document_search":
                    tool_wrapper = mcp_tool_to_langchain(tool_info, session)
                    lc_tools.append(tool_wrapper)
                    tools_map[tool_info.name] = tool_wrapper

            llm_with_tools = llm.bind_tools(lc_tools)

            final_sources = []

            # Reasoning Loop

            while True:
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    return {
                        **state,
                        "messages": messages,
                        "response": {
                            "answer": response.content,
                            "sources": final_sources
                        }
                    }

                # Execute Tool Calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    if tool_name in tools_map:
                        try:
                            raw_result = await tools_map[tool_name].coroutine(**args)
                            parsed = json.loads(raw_result)

                            if "error" in parsed:
                                content = f"Retrieval Error: {parsed['error']}"

                            else:
                                answer = parsed.get("answer", "")
                                sources = parsed.get("sources", [])

                                final_sources = sources

                                content = (
                                    f"Retrieved Answer:\n{answer}\n\n"
                                    f"Sources:\n{sources}"
                                )

                        except Exception as e:
                            content = f"Tool Communication Error: {str(e)}"

                    else:
                        content = "Error: Tool not found."

                    tool_msg = ToolMessage(
                        tool_call_id=tool_call_id,
                        content=content,
                        name=tool_name
                    )

                    messages.append(tool_msg)


# ============================================================================
# FORMATTER NODE
# ============================================================================
def formatter_node(state: AgentState) -> AgentState:
    """
    Format agent responses into a consistent structure.

    Handles:
    - RAG output (answer + sources)
    - SQL output (tabular data)
    - Fallback for other response types
    """

    result = state.get("response", {})

    if "history" not in state or state["history"] is None:
        state["history"] = []

    formatted = {"answer": "", "citations": []}

    # ============================================================
    # ✅ RAG OUTPUT
    # ============================================================
    if isinstance(result, dict) and "answer" in result and "sources" in result:

        answer_text = result.get("answer", "").strip()
        raw_sources = result.get("sources", [])

        # Deduplicate source filenames
        unique_sources = []
        for s in raw_sources:
            if isinstance(s, dict):
                name = s.get("source")
            else:
                name = str(s)

            if name and name not in unique_sources:
                unique_sources.append(name)

        formatted["answer"] = answer_text
        formatted["citations"] = unique_sources

    # ============================================================
    # ✅ SQL OUTPUT
    # ============================================================
    elif isinstance(result, dict) and "columns" in result and "rows" in result:

        rows = result.get("rows", [])
        cols = result.get("columns", [])

        if not rows:
            answer_text = "No records found."

        elif len(cols) == 1 and len(rows) == 1:
            answer_text = f"{rows[0][0]}."

        else:
            answer_text = "Here are the results:\n\n"

            for row in rows[:5]:
                row_text = []
                for col, val in zip(cols, row):
                    row_text.append(f"{col}: {val}")
                answer_text += "- " + ", ".join(row_text) + "\n"

            if len(rows) > 5:
                answer_text += f"\n... and {len(rows) - 5} more rows."

        formatted["answer"] = answer_text
        formatted["citations"] = result.get(
            "sources", ["Northwind PostgreSQL Database"]
        )


    # ============================================================
    # ✅ FALLBACK
    # ============================================================
    else:
        formatted["answer"] = str(result)
        formatted["citations"] = []

    # ============================================================
    # ✅ HISTORY
    # ============================================================
    state["history"].append(
        {"role": "user", "content": state["question"]}
    )

    state["history"].append(
        {"role": "assistant", "content": formatted["answer"]}
    )

    return {**state, "response": formatted}
