"""
PragyanAI - Agentic Sales Chat Bot (LangChain + Groq + LLaMA + FAISS + MongoDB)

Features implemented in this single-file example:
- Ingest a folder of PDFs (student-facing program PDFs / brochures) into a FAISS vectorstore
- Build a Retrieval-Augmented-Generation (RAG) pipeline using modern LangChain 0.2+ APIs
  (`create_history_aware_retriever` + `create_retrieval_chain`), with Groq (ChatGroq) or OpenAI as fallback
- Streamlit web UI that acts as an "agentic" sales assistant: asks for Name, Email, Phone, College, Branch, Semester, Academic Score
- Recommends the best PragyanAI program (starter/intermediate/advanced) and explains why AI + why PragyanAI
- Stores prospect details + conversation snippets to MongoDB Atlas

Notes / TODOs before running:
1) Create a MongoDB Atlas cluster and get the connection string. Set env var MONGODB_URI.
2) Sign up for Groq, get GROQ_API_KEY and set it in env var GROQ_API_KEY (if using Groq).
3) Install required packages (example below).

Requirements (example):
    pip install -U langchain langchain-core langchain-community langchain-text-splitters langchain-groq langchain-openai pymongo sentence-transformers faiss-cpu streamlit pypdf

This file uses the modern LangChain modular API surface (0.2+). It attempts to import Groq's ChatGroq if available and falls back to OpenAI's ChatOpenAI.
"""

import os
import time
from typing import List, Dict, Any

import streamlit as st
from pymongo import MongoClient

# ------------------------- LangChain & Providers -------------------------
# Modern imports for LangChain 0.2+ modular packages
try:
    from langchain_groq import ChatGroq
except ImportError: # Use ImportError for module not found errors
    ChatGroq = None

# OpenAI fallback (if you prefer OpenAI)
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

# RAG / chain helpers - Standardized imports from langchain.chains
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Community packages for documents, embeddings and vectorstores
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    if not os.path.exists(pdf_folder):
        # Create the directory if it doesn't exist
        os.makedirs(pdf_folder, exist_ok=True)
        # Raise error if still no PDFs after creating folder,
        # or if the folder exists but is empty.
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}. Created it. Please put your program PDFs there.")

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

    # Build FAISS index (save/load)
    if os.path.exists(index_path):
        try:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True) # Added allow_dangerous_deserialization for newer FAISS
        except Exception as e:
            st.warning(f"Error loading existing FAISS index: {e}. Rebuilding index.")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(index_path)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(index_path)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

# ------------------------- Build LLM + RAG chain -------------------------

def build_llm_and_rag_chain(retriever):
    """Creates an LLM (ChatGroq preferred, else ChatOpenAI) and constructs a
    history-aware retriever + retrieval chain using modern LangChain APIs."""
    # Create LLM: prefer ChatGroq (Groq LLaMA), else ChatOpenAI
    llm = None
    if ChatGroq is not None:
        try:
            # Using a common Groq model name. Ensure GROQ_API_KEY is set.
            llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, api_key=GROQ_API_KEY)
        except Exception as e:
            st.error(f"Error initializing ChatGroq: {e}. Falling back to OpenAI (if configured).")
            llm = None
    if llm is None and ChatOpenAI is not None:
        # Fallback to OpenAI Chat model (requires OPENAI_API_KEY env var)
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) # Added model name for ChatOpenAI
        except Exception as e:
            st.error(f"Error initializing ChatOpenAI: {e}.")
            llm = None

    if llm is None:
        raise ImportError("No suitable LLM available. Install langchain_groq or langchain_openai and set credentials (GROQ_API_KEY or OPENAI_API_KEY).")

    # Build a contextualize (question rewriter) prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt - Removed extra quotes
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. Use at most three sentences and keep the answer concise.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # Important for keeping chat history in QA chain
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ------------------------- Recommendation logic -------------------------

def recommend_program(profile: Dict[str, Any], top_k: int = 3) -> str:
    """Simple rule-based recommendation to start with. Replace with LLM-driven logic if required."""
    try:
        sem = int(profile.get('semester', 1)) if profile.get('semester') else 1
    except ValueError: # Changed generic Exception to ValueError for int conversion
        sem = 1
    try:
        score = float(profile.get('academic_score') or 0)
    except ValueError: # Changed generic Exception to ValueError for float conversion
        score = 0.0

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
    except FileNotFoundError as e: # Specific error for file not found
        st.error(f"Error: {e}. Please ensure the '{PDFS_FOLDER}' folder exists and contains PDF brochures.")
        return
    except Exception as e:
        st.error(f"Error ingesting PDFs or loading index: {e}.")
        return

    # Build rag chain
    try:
        rag_chain = build_llm_and_rag_chain(retriever)
    except ImportError as e: # Specific error for LLM not found
        st.error(f"Error building RAG chain: {e}")
        return
    except Exception as e:
        st.error(f"Error building RAG chain: {e}. Check your LLM provider installations and credentials.")
        return

    # Chat UI
    if 'history' not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask me about PragyanAI programs, or type your question:")
    if st.button("Send") and user_input.strip():
        with st.spinner("Thinking..."):
            # rag_chain.invoke expects {"input": ..., "chat_history": [...]}
            try:
                # chat_history should be a list of tuples or LangChain Message objects
                # Converting st.session_state.history (list of tuples) to a format
                # acceptable by MessagesPlaceholder if necessary, but LangChain generally handles (human_text, ai_text) tuples.
                # For more robust chat history, consider converting to HumanMessage, AIMessage.
                # Example: from langchain_core.messages import HumanMessage, AIMessage
                # formatted_history = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a) for i, (q, a) in enumerate(st.session_state.history)]
                
                res = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.history})
            except Exception as e:
                st.error(f"RAG chain invocation failed: {e}")
                res = {}

            # robust extraction of answer and sources
            answer = res.get('answer') or res.get('result') or res.get('output') or 'Sorry, I could not process that request.'
            # some chains return 'source_documents' or 'sources'
            sources = res.get('source_documents') or res.get('sources') or []

            # show answer
            st.markdown("**Assistant:**")
            st.write(answer)

            # show sources (brief)
            if sources:
                st.markdown("**Source snippets:**")
                for i, doc in enumerate(sources[:3]):
                    # doc may be a Document object or dict
                    text = getattr(doc, 'page_content', None) or (doc.get('page_content') if isinstance(doc, dict) else str(doc))
                    if text:
                        text = text.replace('\n', ' ')[:400]
                        st.write(f"- {text}...")

            # Save conversation locally and to MongoDB
            st.session_state.history.append((user_input, answer)) # Storing as tuple (user_input, ai_response)
            try:
                # Ensure lead_email is captured if available, otherwise store conversation generally
                mongo_doc = {
                    "question": user_input,
                    "answer": answer,
                    "timestamp": time.time(),
                }
                if email and email.strip(): # Only add lead_email if available
                    mongo_doc["lead_email"] = email
                coll = get_mongo_collection()
                coll.insert_one(mongo_doc)
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
        st.info("Why AI is good: AI automates repetitive tasks, increases problem-solving capability, and opens high-growth career paths. Why PragyanAI: practical projects, mentor support, and career-track curriculum tailored for students.")

    st.markdown("---")
    st.markdown("*This demo app ingests PDFs into a FAISS vectorstore and uses Groq (LLaMA variant) or OpenAI via LangChain as the LLM. Customize the model and index path as needed.*")

# ------------------------- Entrypoint -----------------------------------
if __name__ == '__main__':
    run_streamlit_app()
  
