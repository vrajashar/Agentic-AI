from langgraph.graph import StateGraph, END
from agent_graph.state import AgentState
from agent_graph.nodes import supervisor_node, sql_node, rag_node, guardrail_node, formatter_node
from langgraph.store.postgres import PostgresStore
import os


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("formatter", formatter_node)

    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {
            "sql": "sql_agent",
            "rag": "rag_agent",
            "end": END
        }
    )

    graph.add_edge("sql_agent", "formatter")
    graph.add_edge("rag_agent", "formatter")

    graph.set_finish_point("formatter")

    store = PostgresStore.from_conn_string(os.getenv("DB_URI"))

    return graph.compile(store=store)
