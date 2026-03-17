import streamlit as st
import asyncio
from agent_graph.graph import build_graph
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# -------------------------
# 1. Page Config
# -------------------------
st.set_page_config(page_title="Agentic Assistant", layout="centered")

# Initialize Langfuse
langfuse = get_client()
langfuse_handler = CallbackHandler()

# -------------------------
# 2. Load Graph & State
# -------------------------
@st.cache_resource
def load_graph():
    return build_graph()

app = load_graph()

# UI Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Backend Graph State
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "history": [],
        "session_id": "default_user"
    }

# -------------------------
# 3. Sidebar (Minimal)
# -------------------------
with st.sidebar:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.graph_state["history"] = []
        st.rerun()

# -------------------------
# 4. Main Chat Interface
# -------------------------
st.title("💬 Agentic Assistant")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render Sources/Context if available
        if "context" in msg and msg["context"]:
            with st.expander("View Sources"):
                # RAG Sources
                if "citations" in msg["context"] and msg["context"]["citations"]:
                    st.markdown("**📚 Documents:**")
                    for src in msg["context"]["citations"]:
                        st.write(f"- {src}")
                
                # SQL Query (if applicable)
                if "sql" in msg["context"] and msg["context"]["sql"]:
                    st.markdown("**🗄️ SQL Query:**")
                    st.code(msg["context"]["sql"], language="sql")

# Handle Input
if prompt := st.chat_input("Type your question..."):
    
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Run the Graph
            async def run_agent():
                st.session_state.graph_state["question"] = prompt
                return await app.ainvoke(
                    st.session_state.graph_state,
                    config={"callbacks": [langfuse_handler]}
                )

            final_state = asyncio.run(run_agent())
            st.session_state.graph_state = final_state # Sync memory
            
            # Extract Response Data
            response_data = final_state.get("response", {})
            answer = response_data.get("answer", "No answer generated.")
            
            # Extract Citations (RAG) or SQL (SQL Agent)
            # Note: You need to ensure your SQL node returns 'sql' in the response dict 
            # if you want to see the query here.
            context_data = {
                "citations": response_data.get("citations", []),
                "sql": response_data.get("sql", None) 
            }
            
            # 3. Display Answer
            st.markdown(answer)
            
            # 4. Display Sources (Immediate visibility)
            if context_data["citations"]:
                st.markdown("---")
                st.caption("📚 **Sources:**")
                for src in context_data["citations"]:
                    st.caption(f"• {src}")
            
            if context_data["sql"]:
                st.markdown("---")
                st.caption("🗄️ **Generated SQL:**")
                st.code(context_data["sql"], language="sql")

            # 5. Save to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "context": context_data
            })