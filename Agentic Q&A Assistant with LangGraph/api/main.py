"""
FastAPI microservice for document management and Q&A.

Run with: uvicorn api.main:app --reload --port 8080
"""
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentStatus,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SourceDocument,
    RetrievalMode,
    ServiceHealth,
    DetailedHealth,
    ComponentStatus,
    MetricsResponse,
    ErrorResponse,
)

# ============================================================
# Service State
# ============================================================

class ServiceState:
    """Tracks service state and metrics."""
    def __init__(self):
        self.start_time = time.time()
        self.total_queries = 0
        self.query_times: List[float] = []
        self.processing_documents: dict = {}
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_query_time_ms(self) -> float:
        if not self.query_times:
            return 0.0
        return sum(self.query_times) / len(self.query_times)
    
    def record_query(self, time_ms: float):
        self.total_queries += 1
        self.query_times.append(time_ms)
        # Keep only last 1000 queries for average
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]


state = ServiceState()


# ============================================================
# Lifespan & App Setup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("Starting RAG API service...")
    # Initialize RAG service on startup
    try:
        from rag_agent.rag_service import get_index
        get_index()  # Pre-load index
        print("RAG index loaded successfully")
    except Exception as e:
        print(f"Warning: Could not pre-load index: {e}")
    
    yield
    
    print("Shutting down RAG API service...")


app = FastAPI(
    title="RAG Document Q&A API",
    description="""
    A FastAPI microservice for document management and Q&A using RAG (Retrieval-Augmented Generation).
    
    ## Features
    - **Document Upload**: Upload PDF, TXT, and DOCX files for indexing
    - **Document Management**: List and delete indexed documents
    - **Q&A**: Ask questions and get answers from your documents
    - **Search**: Perform similarity or hybrid search
    - **Streaming**: Stream responses for large outputs
    
    ## Retrieval Modes
    - **similarity**: Vector similarity search using embeddings
    - **hybrid**: Combines BM25 keyword search with vector similarity
    """,
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Document Endpoints
# ============================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    summary="Upload a document for indexing",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (PDF, TXT, or DOCX)"),
):
    """
    Upload a document for indexing into the RAG system.
    
    Supported formats:
    - PDF (.pdf)
    - Text (.txt)
    - Word (.docx)
    
    The document will be processed asynchronously and indexed for search.
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate document ID
    doc_id = Path(file.filename).stem
    
    # Save file
    file_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Track processing
    state.processing_documents[doc_id] = {
        "status": DocumentStatus.PENDING,
        "filename": file.filename,
    }
    
    # Add background task for indexing
    background_tasks.add_task(index_document_task, file_path, doc_id)
    
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        message="Document uploaded successfully. Indexing in progress.",
        chunks_indexed=0,
    )


async def index_document_task(file_path: Path, doc_id: str):
    """Background task to index a document."""
    try:
        state.processing_documents[doc_id]["status"] = DocumentStatus.PROCESSING
        
        from rag_agent.rag_service import index_uploaded_pdfs
        
        # Run indexing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        chunks_count = await loop.run_in_executor(
            None, 
            lambda: index_uploaded_pdfs([file_path])
        )
        
        state.processing_documents[doc_id]["status"] = DocumentStatus.INDEXED
        state.processing_documents[doc_id]["chunks_count"] = chunks_count
        state.processing_documents[doc_id]["indexed_at"] = datetime.utcnow()
        
        print(f"Indexed document {doc_id}: {chunks_count} chunks")
        
    except Exception as e:
        state.processing_documents[doc_id]["status"] = DocumentStatus.FAILED
        state.processing_documents[doc_id]["error"] = str(e)
        print(f"Failed to index document {doc_id}: {e}")


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all indexed documents",
)
async def list_documents():
    """
    List all documents that have been indexed in the system.
    
    Returns document metadata including status, chunk count, and indexing timestamp.
    """
    try:
        from rag_agent.rag_service import chroma_collection, get_indexed_doc_ids
        
        # Get indexed doc IDs from ChromaDB
        indexed_ids = get_indexed_doc_ids()
        
        # Get metadata for each document
        documents = []
        
        # Get chunk counts per document
        data = chroma_collection.get(include=["metadatas"])
        doc_chunks = {}
        for meta in data.get("metadatas", []):
            if meta and "source" in meta:
                source = meta["source"]
                doc_chunks[source] = doc_chunks.get(source, 0) + 1
        
        for source, count in doc_chunks.items():
            doc_id = Path(source).stem
            
            # Get processing info if available
            proc_info = state.processing_documents.get(doc_id, {})
            
            documents.append(DocumentInfo(
                document_id=doc_id,
                filename=source,
                status=proc_info.get("status", DocumentStatus.INDEXED),
                chunks_count=count,
                indexed_at=proc_info.get("indexed_at"),
                metadata={},
            ))
        
        return DocumentListResponse(
            total=len(documents),
            documents=documents,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    tags=["Documents"],
    summary="Delete a document",
)
async def delete_document(document_id: str):
    """
    Delete a document and its indexed chunks from the system.
    
    This removes all chunks associated with the document from the vector store.
    """
    try:
        from rag_agent.rag_service import chroma_collection
        
        # Find and delete all chunks with this doc_id
        data = chroma_collection.get(include=["metadatas"])
        ids_to_delete = []
        
        for i, meta in enumerate(data.get("metadatas", [])):
            if meta and meta.get("doc_id") == document_id:
                ids_to_delete.append(data["ids"][i])
            elif meta and document_id in meta.get("source", ""):
                ids_to_delete.append(data["ids"][i])
        
        if ids_to_delete:
            chroma_collection.delete(ids=ids_to_delete)
            
            # Clean up processing state
            if document_id in state.processing_documents:
                del state.processing_documents[document_id]
            
            return DocumentDeleteResponse(
                document_id=document_id,
                success=True,
                message=f"Deleted {len(ids_to_delete)} chunks",
            )
        else:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.get(
    "/documents/{document_id}",
    response_model=DocumentInfo,
    tags=["Documents"],
    summary="Get document details",
)
async def get_document(document_id: str):
    """
    Get detailed information about a specific document.
    """
    try:
        from rag_agent.rag_service import chroma_collection
        
        # Find chunks for this document
        data = chroma_collection.get(include=["metadatas"])
        chunks = []
        filename = None
        
        for i, meta in enumerate(data.get("metadatas", [])):
            if meta:
                if meta.get("doc_id") == document_id or document_id in meta.get("source", ""):
                    chunks.append(data["ids"][i])
                    if not filename:
                        filename = meta.get("source", document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        
        proc_info = state.processing_documents.get(document_id, {})
        
        return DocumentInfo(
            document_id=document_id,
            filename=filename or document_id,
            status=proc_info.get("status", DocumentStatus.INDEXED),
            chunks_count=len(chunks),
            indexed_at=proc_info.get("indexed_at"),
            metadata={},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


# ============================================================
# Query Endpoints
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Ask a question",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def query_documents(request: QueryRequest):
    """
    Ask a question and get an answer based on indexed documents.
    
    The system retrieves relevant documents and generates an answer using LLM.
    
    **Retrieval Modes:**
    - `similarity`: Pure vector similarity search
    - `hybrid`: Combines BM25 keyword matching with vector similarity
    """
    if request.stream:
        return StreamingResponse(
            stream_query(request),
            media_type="text/event-stream",
        )
    
    start_time = time.time()
    
    try:
        from rag_agent.rag_service import answer_question, answer_question_hybrid
        
        # Run query in thread pool
        loop = asyncio.get_event_loop()
        
        if request.retrieval_mode == RetrievalMode.HYBRID:
            answer, sources = await loop.run_in_executor(
                None,
                lambda: answer_question_hybrid(
                    request.query,
                    top_k=request.top_k,
                )
            )
        else:
            answer, sources = await loop.run_in_executor(
                None,
                lambda: answer_question(
                    request.query,
                    similarity_top_k=request.top_k,
                    similarity_cutoff=request.similarity_cutoff,
                )
            )
        
        processing_time = (time.time() - start_time) * 1000
        state.record_query(processing_time)
        
        return QueryResponse(
            query=request.query,
            answer=answer or "I could not find an answer to your question.",
            sources=[
                SourceDocument(
                    text=s.get("text", ""),
                    source=s.get("source"),
                    page=s.get("page"),
                    doc_id=s.get("doc_id"),
                )
                for s in sources
            ],
            retrieval_mode=request.retrieval_mode,
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


async def stream_query(request: QueryRequest) -> AsyncGenerator[str, None]:
    """Stream query response."""
    import json
    
    start_time = time.time()
    
    try:
        from rag_agent.rag_service import answer_question, answer_question_hybrid
        
        loop = asyncio.get_event_loop()
        
        # Send start event
        yield f"data: {json.dumps({'event': 'start', 'query': request.query})}\n\n"
        
        # Run query
        if request.retrieval_mode == RetrievalMode.HYBRID:
            answer, sources = await loop.run_in_executor(
                None,
                lambda: answer_question_hybrid(request.query, top_k=request.top_k)
            )
        else:
            answer, sources = await loop.run_in_executor(
                None,
                lambda: answer_question(
                    request.query,
                    similarity_top_k=request.top_k,
                    similarity_cutoff=request.similarity_cutoff,
                )
            )
        
        # Stream answer in chunks
        chunk_size = 50
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i+chunk_size]
            yield f"data: {json.dumps({'event': 'chunk', 'text': chunk})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        processing_time = (time.time() - start_time) * 1000
        state.record_query(processing_time)
        
        # Send sources
        yield f"data: {json.dumps({'event': 'sources', 'sources': sources})}\n\n"
        
        # Send end event
        yield f"data: {json.dumps({'event': 'end', 'processing_time_ms': processing_time})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Query"],
    summary="Search documents",
)
async def search_documents(request: SearchRequest):
    """
    Search for relevant documents without generating an answer.
    
    Returns retrieved document chunks with similarity scores.
    Useful for exploring what's in the index or debugging retrieval.
    """
    start_time = time.time()
    
    try:
        from rag_agent.rag_service import similarity_search, hybrid_search
        
        loop = asyncio.get_event_loop()
        
        if request.retrieval_mode == RetrievalMode.HYBRID:
            results = await loop.run_in_executor(
                None,
                lambda: hybrid_search(request.query, top_k=request.top_k)
            )
        else:
            results = await loop.run_in_executor(
                None,
                lambda: similarity_search(request.query, top_k=request.top_k)
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=[
                SourceDocument(
                    text=r.get("text", ""),
                    source=r.get("source"),
                    page=r.get("page"),
                    doc_id=r.get("doc_id"),
                    score=r.get("score"),
                )
                for r in results
            ],
            total_results=len(results),
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ============================================================
# Health & Monitoring Endpoints
# ============================================================

@app.get(
    "/health",
    response_model=ServiceHealth,
    tags=["Health"],
    summary="Health check",
)
async def health_check():
    """
    Simple health check endpoint.
    
    Returns service status and version.
    """
    return ServiceHealth(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@app.get(
    "/health/detailed",
    response_model=DetailedHealth,
    tags=["Health"],
    summary="Detailed health check",
)
async def detailed_health_check():
    """
    Detailed health check with component statuses.
    
    Checks:
    - ChromaDB connection
    - LLM availability
    - Embedding model
    """
    components = []
    overall_status = "healthy"
    
    # Check ChromaDB
    try:
        start = time.time()
        from rag_agent.rag_service import chroma_collection
        count = chroma_collection.count()
        latency = (time.time() - start) * 1000
        components.append(ComponentStatus(
            name="chromadb",
            status="up",
            latency_ms=latency,
            details=f"{count} chunks indexed",
        ))
    except Exception as e:
        overall_status = "unhealthy"
        components.append(ComponentStatus(
            name="chromadb",
            status="down",
            details=str(e),
        ))
    
    # Check embedding model
    try:
        start = time.time()
        from rag_agent.rag_service import embed_model
        _ = embed_model.get_text_embedding("test")
        latency = (time.time() - start) * 1000
        components.append(ComponentStatus(
            name="embedding_model",
            status="up",
            latency_ms=latency,
        ))
    except Exception as e:
        overall_status = "degraded"
        components.append(ComponentStatus(
            name="embedding_model",
            status="down",
            details=str(e),
        ))
    
    # Check LLM (optional - may be slow)
    components.append(ComponentStatus(
        name="llm",
        status="up",
        details="Groq API configured",
    ))
    
    return DetailedHealth(
        status=overall_status,
        version="1.0.0",
        uptime_seconds=state.uptime_seconds,
        components=components,
        timestamp=datetime.utcnow(),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Health"],
    summary="Service metrics",
)
async def get_metrics():
    """
    Get service metrics including query counts and timing.
    """
    try:
        from rag_agent.rag_service import chroma_collection, get_indexed_doc_ids
        
        doc_ids = get_indexed_doc_ids()
        chunk_count = chroma_collection.count()
        
        return MetricsResponse(
            total_documents=len(doc_ids),
            total_chunks=chunk_count,
            total_queries=state.total_queries,
            avg_query_time_ms=state.avg_query_time_ms,
            uptime_seconds=state.uptime_seconds,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with links to documentation."""
    return {
        "name": "RAG Document Q&A API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }
