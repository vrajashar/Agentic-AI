import streamlit as st
from pathlib import Path
from rag_service import answer_question, index_uploaded_pdfs

st.set_page_config(
    page_title="Biomedical RAG Assistant",
    layout="wide"
)

st.title("🧬 Biomedical Research Assistant")
st.markdown("Ask questions over biomedical research PDFs")

PDF_DIR = Path("pdfs")
PDF_DIR.mkdir(exist_ok=True)

# -----------------------
# Upload Section
# -----------------------
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    saved_paths = []

    for uploaded_file in uploaded_files:
        save_path = PDF_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(save_path)

    with st.spinner("Indexing uploaded documents..."):
        num_chunks = index_uploaded_pdfs(saved_paths)

    st.sidebar.success(f"Indexed {len(saved_paths)} PDFs ({num_chunks} chunks)")

# -----------------------
# Q&A Section
# -----------------------
query = st.text_input("Enter your question:")

if st.button("Ask") and query:
    with st.spinner("Searching documents and generating answer..."):
        answer, sources = answer_question(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    if sources:
        for src in sources:
            st.write(f"- **{src['source']}**, page {src['page']}")
    else:
        st.write("No sources found.")
