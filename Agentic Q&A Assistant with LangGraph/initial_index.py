from pathlib import Path
from rag_agent.rag_service import index_uploaded_pdfs

PDF_DIR = Path("pdfs")

pdfs = list(PDF_DIR.glob("*.pdf"))

if not pdfs:
    print("No PDFs found.")
else:
    count = index_uploaded_pdfs(pdfs)
    print(f"Indexed {count} chunks.")
