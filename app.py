import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore

from rag.settings import load_settings
from rag.gemini_client import build_client
from rag.embeddings import GeminiEmbeddings
from rag.rag_pipeline import retrieve, answer_stream, answer_structured
from rag.observability import build_langfuse, start_trace, end_trace


settings = load_settings()
client = build_client(settings)
langfuse = build_langfuse(settings)


@st.cache_resource(show_spinner=True)
def load_faiss() -> FAISS:
    if not settings.faiss_dir.exists():
        raise FileNotFoundError(f"Vectorstore not found: {settings.faiss_dir}")
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)
    return FAISS.load_local(
        str(settings.faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=True)
def load_qdrant() -> QdrantVectorStore:
    if not settings.qdrant_path.exists():
        raise FileNotFoundError(f"Qdrant store not found: {settings.qdrant_path}")
    embeddings = GeminiEmbeddings(client=client, model=settings.embedding_model)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=str(settings.qdrant_path),
        collection_name=settings.qdrant_collection,
    )


st.title("IFC Annual Report 2024 — RAG (Gemini)")
st.caption("Gemini 2.0 Flash with Vertex AI auth, FAISS/Qdrant retrieval, Langfuse tracing.")

store_choice = st.selectbox("Vector store", ["FAISS", "Qdrant"])
top_k = st.slider("Top K", min_value=3, max_value=10, value=5, step=1)
show_context = st.checkbox("Show retrieved chunks", value=True)

q = st.text_input("Ask a question about the PDF")

if q:
    try:
        db = load_faiss() if store_choice == "FAISS" else load_qdrant()
    except Exception as exc:
        st.error(f"{exc}. Run: python ingest.py")
        st.stop()

    docs = retrieve(db, q, k=top_k)
    retrieved_chunks = [{"page": d.metadata.get("page"), "content": d.page_content} for d in docs]
    trace = start_trace(langfuse, "rag_query", {"query": q, "store": store_choice, "retrieved_chunks": retrieved_chunks})

    st.subheader("Answer (streaming)")
    answer_box = st.empty()
    streamed = ""
    for token in answer_stream(client, settings.gemini_model, q, docs):
        streamed += token
        answer_box.markdown(streamed)

    st.subheader("Answer (structured)")
    structured = answer_structured(client, settings.gemini_model, q, docs)
    st.json(structured)

    end_trace(trace, {"answer": structured, "raw_answer": streamed})

    if show_context:
        st.subheader("Top matches (with page numbers)")
        for i, d in enumerate(docs, 1):
            page = d.metadata.get("page", None)
            st.markdown(f"### Match {i} — page {page if page is not None else '?'}")
            st.write(d.page_content)
            st.divider()
