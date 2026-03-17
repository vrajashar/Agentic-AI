from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from chunking import get_chunks_for_uploaded_pdf
from pathlib import Path


load_dotenv()

# ---------- Embeddings ----------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- Vector Store ----------
vectorstore = Chroma(
    collection_name="biomedical_chunks",
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

# ---------- Prompt ----------
PROMPT = PromptTemplate(
    template="""
You are a biomedical research assistant.

Use ONLY the provided context to answer the question.
If the answer is not present, say:
"I do not know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# ---------- LLM ----------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

# ---------- RAG Chain ----------
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# ---------- Public Function ----------
def answer_question(query: str):
    result = rag_chain.invoke({"query": query})

    answer = result["result"]
    sources = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in result["source_documents"]
    ]

    return answer, sources


PDF_DIR = Path("pdfs")

def index_uploaded_pdfs(pdf_paths):
    all_chunks = []

    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem
        chunks = get_chunks_for_uploaded_pdf(pdf_path, doc_id)
        all_chunks.extend(chunks)

    if all_chunks:
        vectorstore.add_documents(all_chunks)

    return len(all_chunks)


