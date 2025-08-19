multimodal_rag/
│── app.py                      # FastAPI main entry (API routes)
│── config.py                   # API keys, settings, constants
│── rag_pipeline.py              # Core RAG workflow (retrieval + Gemini query)
│── embeddings.py                # Text + Image embedding functions
│── vectorstore.py               # FAISS index: init, add, search, persist
│── retriever.py                 # Retrieval logic (top-k docs/images)
│── ingest_pdf.py                # PDF ingestion: parse, chunk, embed, store
│── gemini_api.py                # Wrapper for Gemini API calls
│
├── utils/
│   ├── preprocess.py            # Preprocessing for text/images
│   └── chunking.py              # Text chunking utilities
│
├── data/
│   ├── documents/               # Raw PDFs stored here
│   ├── images/                  # Optional: store uploaded images
│   └── faiss_index/             # Saved FAISS DB files
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
