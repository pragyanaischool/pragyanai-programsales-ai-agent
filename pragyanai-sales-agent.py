"""
PragyanAI - Agentic Sales Chat Bot (LangChain + Groq + LLaMA + FAISS + MongoDB)

Features implemented in this single-file example:
- Ingest a folder of PDFs (student-facing program PDFs / brochures) into a FAISS vectorstore
- Build a Conversational Retrieval Chain using Groq (ChatGroq) as LLM and HuggingFace embeddings
- Streamlit web UI that acts as an "agentic" sales assistant: asks for Name, Email, Phone, College, Branch, Semester, Academic Score
- Recommends the best PragyanAI program (starter/intermediate/advanced) and explains why AI + why PragyanAI
- Stores prospect details + conversation snippets to MongoDB Atlas

Notes / TODOs before running:
1) Create a MongoDB Atlas cluster and get the connection string. Set env var MONGODB_URI.
2) Sign up for Groq, get GROQ_API_KEY and set it in env var GROQ_API_KEY.
3) Install required packages (example below).

Requirements (example):
    pip install langchain langchain-groq groq pymongo sentence-transformers faiss-cpu streamlit pypdf openai --upgrade

This example aims to be a high-quality starting point. You may need to adapt imports depending on langchain and provider package versions.
"""

import os
import time
from typing import List, Dict, Any

import streamlit as st
from pymongo import MongoClient

# LangChain / Groq / embeddings / vectorstore imports
# NOTE: package names/paths may change with versions; adjust if you see ImportError.
try:
    from langchain_groq import ChatGroq
except Exception:
    # fallback import path (older/newer installs)
    try:
        from langchain.groq import ChatGroq
    except Exception:
        ChatGroq = None

#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use SentenceTransformers embeddings via LangChain wrapper
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# ------------------------- Configuration --------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "YOUR_MONGODB_URI_HERE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pragyanai")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "leads")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDFS_FOLDER = os.getenv("PDFS_FOLDER", "pdfs/")  # local folder where uploaded PDFs are stored
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# ------------------------- MongoDB utilities -----------------------------
def get_mongo_collection(uri: str = MONGODB_URI):
    if not uri or uri == "YOUR_MONGODB_URI_HERE":
        raise ValueError("Please set MONGODB_URI environment variable to your MongoDB Atlas connection string")
    client = MongoClient(uri)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

# ------------------------- Ingestion ------------------------------------
def ingest_pdfs_to_faiss(pdf_folder: str = PDFS_FOLDER, index_path: str = FAISS_INDEX_PATH):
    """Loads PDFs, splits them, creates embeddings and stores in FAISS index on disk.
    Returns a LangChain retriever (FAISS wrapped)"""
    docs = []
    for fname in os.listdir(pdf_folder):
        if not fname.lower().endswith('.pdf'):
            continue
        full = os.path.join(pdf_folder, fname)
        loader = PyPDFLoader(full)
        loaded = loader.load()
        docs.extend(loaded)

    if len(docs) == 0:
        raise FileNotFoundError(f"No PDFs found in {pdf_folder}. Put your program PDFs there.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Build FAISS index
    if os.path.exists(index_path):
        # try to load existing index
        try:
            vectorstore = FAISS.load_local(index_path, embeddings)
        except Exception:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(index_path)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

# ------------------------- LLM / Chain ----------------------------------

def build_llm_and_chain(retriever):
    """Initializes Groq LLM (ChatGroq) and constructs a Conversational Retrieval Chain."""
    if ChatGroq is None:
        raise ImportError("ChatGroq import failed. Make sure langchain-groq and langchain are installed and up to date.")

    # Initialize LLM - choose a LLaMA/Groq model. Adjust model name as needed.
    llm = ChatGroq(model="llama-3.1-13b", temperature=0.0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True)
    return qa

# ------------------------- Recommendation logic -------------------------

def recommend_program(profile: Dict[str, Any], top_k: int = 3) -> str:
    """Simple rule-based recommendation to start with. Replace with LLM-driven logic if required."""
    sem = int(profile.get('semester', 1)) if profile.get('semester') else 1
    score = float(profile.get('academic_score') or 0)

    if sem <= 2 or score < 60:
        return "Starter: AI Fundamentals & Python for AI (Beginner) - 6 weeks"
    elif sem <= 6 and score < 75:
        return "Intermediate: Applied ML & NLP Projects - 8 weeks"
    else:
        return "Advanced: Deep Learning, Generative AI & Agentic Systems - 12 weeks"

# ------------------------- Streamlit App --------------------------------

def run_streamlit_app():
    st.set_page_config(page_title="PragyanAI Sales Agent", layout="centered")

    st.title("PragyanAI — Admissions Assistant")
    st.write("Hi! I'm PragyanAI's admissions assistant. I can answer questions about our programs and recommend which one suits you.")

    st.sidebar.header("Lead Capture")
    name = st.sidebar.text_input("Full name")
    email = st.sidebar.text_input("Email")
    phone = st.sidebar.text_input("Phone")
    college = st.sidebar.text_input("College")
    branch = st.sidebar.text_input("Branch")
    semester = st.sidebar.text_input("Semester (number)")
    academic_score = st.sidebar.text_input("Academic Score (percentage)")

    if st.sidebar.button("Register Lead"):
        try:
            coll = get_mongo_collection()
            lead = {
                "name": name,
                "email": email,
                "phone": phone,
                "college": college,
                "branch": branch,
                "semester": semester,
                "academic_score": academic_score,
                "created_at": time.time(),
            }
            coll.insert_one(lead)
            st.sidebar.success("Lead saved to MongoDB")
        except Exception as e:
            st.sidebar.error(f"Error saving lead: {e}")

    # Load or build the retriever
    try:
        retriever = ingest_pdfs_to_faiss()
    except Exception as e:
        st.error(f"Error ingesting PDFs or loading index: {e}. Make sure PDFS_FOLDER contains brochures.")
        return

    # Build chain
    try:
        qa_chain = build_llm_and_chain(retriever)
    except Exception as e:
        st.error(f"Error building LLM chain: {e}. Check langchain-groq installation and GROQ_API_KEY.")
        return

    # Chat UI
    if 'history' not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask me about PragyanAI programs, or type your question:")
    if st.button("Send") and user_input.strip():
        with st.spinner("Thinking..."):
            result = qa_chain({'question': user_input, 'chat_history': st.session_state.history})
            answer = result.get('answer') or result.get('result') or ""
            sources = result.get('source_documents', [])

            # show answer
            st.markdown("**Assistant:**")
            st.write(answer)

            # show sources (brief)
            if sources:
                st.markdown("**Source snippets:**")
                for i, doc in enumerate(sources[:3]):
                    text = doc.page_content.replace('\n', ' ')[:400]
                    st.write(f"- {text}...")

            # Save conversation locally and to MongoDB
            st.session_state.history.append((user_input, answer))
            try:
                coll = get_mongo_collection()
                coll.insert_one({
                    "lead_email": email,
                    "question": user_input,
                    "answer": answer,
                    "timestamp": time.time(),
                })
            except Exception:
                # don't block UI on DB error
                pass

    # Quick-recommendation button
    if st.button("Recommend Program For Me"):
        profile = {
            'semester': semester,
            'academic_score': academic_score,
        }
        rec = recommend_program(profile)
        st.success(f"Recommended: {rec}")
        st.info("Why AI is good: AI automates repetitive tasks, increases problem-solving capability, and opens high-growth career paths.\nWhy PragyanAI: practical projects, mentor support, and career-track curriculum tailored for students.")

    st.markdown("---")
    st.markdown("*This demo app ingests PDFs into a FAISS vectorstore and uses Groq (LLaMA variant) via LangChain as the LLM. Customize the model and index path as needed.*")

# ------------------------- Entrypoint -----------------------------------
if __name__ == '__main__':
    run_streamlit_app()
