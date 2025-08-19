import streamlit as st
from rag_pipeline import multimodal_rag
from ingest_pdf import ingest_pdf
from config import FAISS_INDEX_PATH, TOP_K
import tempfile
import os

st.set_page_config(page_title="📚 Multimodal RAG with Gemini + FAISS", layout="wide")

st.title("📚 Multimodal RAG System")
st.markdown("Ask questions using **text, PDFs, or images**, powered by Gemini + FAISS 🚀")

# ----------------------------
# PDF UPLOAD + INGESTION
# ----------------------------
st.sidebar.header("📥 Knowledge Base")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_pdf.read())
        pdf_path = tmp_pdf.name
    st.sidebar.info(f"Ingesting: {uploaded_pdf.name}...")
    result_msg = ingest_pdf(pdf_path, faiss_index_path=FAISS_INDEX_PATH)
    st.sidebar.success(result_msg)


# ----------------------------
# QUERY INPUT (TEXT + IMAGE)
# ----------------------------
st.header("🔎 Ask a Question")
query_text = st.text_area("Enter your question (optional if using image):")

uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

# ----------------------------
# RUN PIPELINE
# ----------------------------
if st.button("Get Answer"):
    if not query_text and not uploaded_image:
        st.warning("Please provide a query (text or image).")
    else:
        image_bytes = uploaded_image.read() if uploaded_image else None
        with st.spinner("🔍 Retrieving and querying Gemini..."):
            answer, retrieved = multimodal_rag(
                text=query_text if query_text else None,
                image_bytes=image_bytes,
                faiss_index_path=FAISS_INDEX_PATH,
                top_k=TOP_K
            )

        # Display Gemini’s answer
        st.subheader("💡 Answer:")
        st.write(answer)

        # Display retrieved sources
        if retrieved:
            st.subheader("📂 Retrieved Sources")
            for r in retrieved:
                with st.expander(f"{r.get('source', 'unknown')} | Chunk {r.get('chunk_id')}"):
                    st.write(r["content"])
