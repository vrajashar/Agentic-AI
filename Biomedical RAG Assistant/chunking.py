from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_with_ocr,load_with_unstructured
from pathlib import Path

def get_chunks(pdf_path: Path, doc_id: str):
    docs = load_with_ocr(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = pdf_path.name 
    

    return chunks


def get_chunks_for_uploaded_pdf(pdf_path, doc_id):
    docs = load_with_unstructured(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["source"] = pdf_path.name

    return chunks

