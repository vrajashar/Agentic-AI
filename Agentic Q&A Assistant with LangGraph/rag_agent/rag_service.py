from pathlib import Path
from typing import List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from rag_agent.document_loader import load_documents
from rag_agent.chunking import chunk_documents

# ============================================================
# Embeddings
# ============================================================

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ============================================================
# Vector Store (Chroma)
# ============================================================

chroma_client = chromadb.PersistentClient(path="./chroma_db")
print("Chroma persist directory: ./chroma_db")

chroma_collection = chroma_client.get_or_create_collection(
    name="biomedical_chunks"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# ============================================================
# LLM
# ============================================================

llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Configure global settings for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model

# ============================================================
# Index
# ============================================================

index: VectorStoreIndex | None = None


def get_index() -> VectorStoreIndex:
    global index
    if index is None:
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )
    return index


# ============================================================
# Ingestion
# ============================================================
def get_indexed_doc_ids() -> set[str]:
    """
    Returns a set of document IDs already indexed in Chroma.
    """
    data = chroma_collection.get(include=["metadatas"])
    metadatas = data.get("metadatas", [])

    doc_ids = set()
    for meta in metadatas:
        if meta and "doc_id" in meta:
            doc_ids.add(meta["doc_id"])

    return doc_ids


def index_uploaded_pdfs(pdf_paths: List[Path]) -> int:
    indexed_doc_ids = get_indexed_doc_ids()
    all_nodes = []

    for pdf_path in pdf_paths:
        doc_id = pdf_path.stem

        if doc_id in indexed_doc_ids:
            continue  # already indexed

        documents = load_documents(pdf_path)

        nodes = chunk_documents(
            documents=documents,
            doc_id=doc_id,
            source_name=pdf_path.name,
        )

        all_nodes.extend(nodes)

    if not all_nodes:
        return 0

    idx = get_index()
    idx.insert_nodes(all_nodes)

    return len(all_nodes)

# ============================================================
# Querying
# ============================================================

def get_retriever(similarity_top_k: int = 5):
    """
    Creates a custom vector index retriever.
    """
    idx = get_index()
    
    retriever = VectorIndexRetriever(
        index=idx,
        similarity_top_k=similarity_top_k,
    )
    
    return retriever


def similarity_search(query: str, top_k: int = 5):
    """
    Performs similarity search and returns retrieved nodes.
    Useful for testing retrieval quality.
    """
    retriever = get_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    
    return [
        {
            "text": node.text,
            "score": node.score,
            "source": node.metadata.get("source"),
            "page": node.metadata.get("page"),
            "doc_id": node.metadata.get("doc_id"),
        }
        for node in nodes
    ]


def answer_question(query: str, similarity_top_k: int = 5, similarity_cutoff: float = 0.0):
    """
    Answer a question using RAG with custom retriever.
    
    Args:
        query: The question to answer
        similarity_top_k: Number of similar documents to retrieve
        similarity_cutoff: Minimum similarity score threshold (0.0-1.0)
    """
    idx = get_index()

    # Custom retriever with configurable parameters
    retriever = VectorIndexRetriever(
        index=idx,
        similarity_top_k=similarity_top_k,
    )
    
    # Build query engine with custom retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        ] if similarity_cutoff > 0 else None,
    )

    response = query_engine.query(query)

    return response.response, [
        {
            "text": node.text,
            "source": node.metadata.get("source"),
            "page": node.metadata.get("page"),
            "doc_id": node.metadata.get("doc_id"),
        }
        for node in response.source_nodes
    ]


# ============================================================
# Hybrid Retrieval (BM25 + Vector)
# ============================================================

def get_all_nodes():
    """
    Retrieves all nodes from ChromaDB for BM25 indexing.
    """
    from llama_index.core.schema import TextNode
    
    data = chroma_collection.get(include=["documents", "metadatas"])
    nodes = []
    
    for doc_id, doc_text, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
        if doc_text and len(doc_text.strip()) > 0:
            node = TextNode(
                text=doc_text,
                id_=doc_id,
                metadata=metadata or {}
            )
            nodes.append(node)
    
    return nodes


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5):
    """
    Performs hybrid retrieval combining BM25 keyword search with vector similarity.
    
    Args:
        query: The search query
        top_k: Number of results to return
        alpha: Weight for vector search (0.0 = pure BM25, 1.0 = pure vector)
    
    Returns:
        List of retrieved document chunks with scores
    """
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
        
        idx = get_index()
        
        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=idx,
            similarity_top_k=top_k,
        )
        
        # Get all nodes for BM25
        all_nodes = get_all_nodes()
        
        if not all_nodes:
            print("No nodes found for BM25, falling back to vector search")
            return similarity_search(query, top_k=top_k)
        
        # BM25 retriever (keyword-based) - pass nodes directly
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=top_k,
        )
        
        # Fusion retriever combines both
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,  # Don't generate query variations
            mode="reciprocal_rerank",  # RRF fusion
        )
        
        nodes = hybrid_retriever.retrieve(query)
        
        return [
            {
                "text": node.text,
                "score": node.score if node.score else 0.0,
                "source": node.metadata.get("source"),
                "page": node.metadata.get("page"),
                "doc_id": node.metadata.get("doc_id"),
            }
            for node in nodes
        ]
        
    except ImportError:
        # Fallback to simple vector search if BM25 not available
        print("BM25 retriever not available, falling back to vector search")
        return similarity_search(query, top_k=top_k)


def answer_question_hybrid(query: str, top_k: int = 5, alpha: float = 0.5):
    """
    Answer a question using hybrid RAG (BM25 + vector retrieval).
    
    Args:
        query: The question to answer
        top_k: Number of documents to retrieve
        alpha: Weight for vector vs BM25 (0.5 = equal weight)
    """
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
        
        idx = get_index()
        
        # Vector retriever
        vector_retriever = VectorIndexRetriever(
            index=idx,
            similarity_top_k=top_k,
        )
        
        # Get all nodes for BM25
        all_nodes = get_all_nodes()
        
        if not all_nodes:
            print("No nodes found for BM25, using standard retrieval")
            return answer_question(query, similarity_top_k=top_k)
        
        # BM25 retriever - pass nodes directly
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=top_k,
        )
        
        # Fusion retriever
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
        )
        
        # Build query engine with hybrid retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
        )
        
        response = query_engine.query(query)
        
        return response.response, [
            {
                "text": node.text,
                "source": node.metadata.get("source"),
                "page": node.metadata.get("page"),
                "doc_id": node.metadata.get("doc_id"),
            }
            for node in response.source_nodes
        ]
        
    except ImportError:
        # Fallback to standard retrieval
        print("Hybrid retrieval not available, using standard vector search")
        return answer_question(query, similarity_top_k=top_k)


