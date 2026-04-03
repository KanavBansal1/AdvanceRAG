import streamlit as st
import os

from ingestion.loader import load_documents
from ingestion.chunking import chunk_documents
from ingestion.embeddings import get_embeddings

from retrieval.vectorstore import create_vectorstore
from retrieval.retriever import get_retriever
from retrieval.hybrid_retriever import HybridRetriever

from llm.groq_llm import get_llm
from llm.rag_pipeline import create_rag

from evaluation.rag_eval import evaluate_rag


st.title("Advanced RAG Document Intelligence")

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Upload UI
uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True
)

if uploaded_files:

    for file in uploaded_files:

        with open(
            os.path.join("data", file.name),
            "wb"
        ) as f:

            f.write(file.getbuffer())

    st.success("Documents uploaded successfully")

    # Reset cache when new docs uploaded
    st.cache_resource.clear()


# Chat Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def initialize_rag():

    documents = load_documents("data")

    # Handle empty documents
    if len(documents) == 0:
        return None

    chunks = chunk_documents(documents)

    embeddings = get_embeddings()

    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    retriever = get_retriever(
        vectorstore
    )

    hybrid_retriever = HybridRetriever(
        retriever,
        chunks
    )

    llm = get_llm()

    qa = create_rag(
        llm,
        hybrid_retriever
    )

    return qa


data_path = "data"

# Create folder safely
os.makedirs(data_path, exist_ok=True)

# Only initialize if files exist
files = os.listdir(data_path)

if len(files) == 0:
    st.warning("Upload documents to start querying")
    st.stop()

qa = initialize_rag()


query = st.chat_input("Ask your question")

if query:

    result = qa(query)

    # Save chat
    st.session_state.chat_history.append({
        "question": query,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources": result["sources"],
        "contexts": result["contexts"]
    })

    # Evaluation
    # evaluation = evaluate_rag(
    #     query,
    #     result["answer"],
    #     result["contexts"]
    # )

    # st.write("### Evaluation")
    # st.write(evaluation)


# Display Chat
for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):

        st.write(chat["answer"])

        st.write("Confidence:", round(chat["confidence"], 3))

        with st.expander("Sources"):
            for source in chat["sources"]:
                st.write(source)