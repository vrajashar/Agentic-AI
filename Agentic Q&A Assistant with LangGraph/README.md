# Multi-Source Agentic Q&A Assistant

A production-ready agentic chatbot that intelligently routes user queries between:

- 📊 Structured Data: Northwind PostgreSQL Database (SQL Analytics)
- 📚 Unstructured Data: PubMed / Biomedical PDFs (RAG with Vector Search)

The system is built using LangGraph for multi-agent orchestration and MCP (Model Context Protocol) with FastMCP for tool integration.

---

## 🚀 Key Features

- **LangGraph Supervisor Agent** for intelligent routing
- **Specialized SQL Agent** for database analytics
- **Specialized RAG Agent** for biomedical document search
- **MCP Server (FastMCP)** integrating SQL & RAG tools
- **Guardrails** for off-topic query detection
- **SQL Error Handling** with automatic query reformulation
- **RAG Retrieval Failure Handling** with graceful fallback
- **Persistent Conversation Memory** using PostgreSQL
- **Source Citations** for all responses

---

## 🛠 Tech Stack

- **Language:** Python 3.11+
- **Orchestration:** LangGraph, LangChain
- **Protocol:** FastMCP (Model Context Protocol)
- **Database:** PostgreSQL
- **Vector Store:** ChromaDB
- **LLM:** Groq
- **Frontend:** Streamlit

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
uv pip install -r requirements.txt
```

### 3️⃣ Environment Variables
Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/northwind
```

### 4️⃣ PostgreSQL Setup

**Step 1: Create the database**

```sql
CREATE DATABASE northwind;
```

(Make sure to import the standard Northwind schema data into this database)

**Step 2: Create memory table**

Execute the following SQL to set up persistent conversation history:

```sql
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 5️⃣ Start MCP Server
```bash
python mcp_server.py
```

### 6️⃣ Start Streamlit UI
```bash
streamlit run ui/app.py
```
