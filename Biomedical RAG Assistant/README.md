# 🧬 Biomedical Research Assistant (RAG)

A production-ready **Retrieval-Augmented Generation (RAG)** system for querying biomedical research PDFs (PubMed clinical trials).  
The system supports **OCR-based scanned documents**, **clean text-based PDFs**, semantic search using **open-source embeddings**, and an interactive **Streamlit UI** with source attribution.

---

## 🚀 Features

-  PDF ingestion (OCR for scanned PDFs, Unstructured parsing for clean PDFs)
-  Configurable chunking with metadata tracking
-  Grounded LLM responses (no hallucinations)
-  Source attribution (PDF name + page number)
-  Streamlit-based chat interface
-  Incremental document upload & indexing
-  Clean separation of ingestion, retrieval, and UI layers

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **ChromaDB**
- **Google Gemini (via LangChain)**
- **Streamlit**
- **Tesseract OCR**
- **Unstructured**

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure API Key
Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```



