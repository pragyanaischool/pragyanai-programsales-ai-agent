"""
PragyanAI - Agentic Sales Chat Bot (LangChain + Groq + LLaMA + FAISS + MongoDB)

Features implemented in this single-file example:
- **MODIFIED**: Automatically summarizes all program PDFs into a table on startup.
- Ingest a folder of PDFs into a FAISS vectorstore for RAG.
- Build a Retrieval-Augmented-Generation (RAG) pipeline using modern LangChain 0.2+ APIs.
- Streamlit web UI that acts as an "agentic" sales assistant.
- Recommends the best PragyanAI program based on user input.
- Stores prospect details + conversation snippets to MongoDB Atlas.

Requirements:
    pip install -U langchain langchain-core langchain-community langchain-text-splitters langchain-groq langchain-openai pymongo sentence-transformers faiss-cpu streamlit pypdf pandas
"""

import os
import time
from typing import List, Dict, Any
import pandas as pd

import streamlit as st
from pymongo import MongoClient

# ------------------------- LangChain & Providers -------------------------
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------- Configuration --------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "YOUR_MONGODB_URI_HERE")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "pragyanai")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "leads")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PDFS_FOLDER = os.getenv("PDFS_FOLDER", "pdfs/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

# ------------------------- MongoDB utilities -----------------------------
def get_mongo_collection(uri: str = MONGODB_URI):
    if not uri or uri == "YOUR_MONGODB_URI_HERE":
        raise ValueError("Please set MONGODB_URI environment variable to your MongoDB Atlas connection string")
    client = MongoClient(uri)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

# ------------------------- Ingestion & Summarization ------------------------------------
@st.cache_data(show_spinner="Analyzing program brochures...")
def summarize_pdf(pdf_path: str, _llm) -> Dict[str, str]:
    """Loads a single PDF and uses an LLM to extract a brief summary."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_content = " ".join([doc.page_content for doc in docs])

        prompt_template = """
        You are an expert at summarizing academic and professional program brochures. 
        Analyze the following text from a program brochure and extract the key details.

        Document Text:
        {document}

        Provide a concise summary covering:
        1. Program Name: The official title of the program.
        2. Program Duration: The total length of the program (e.g., 6 weeks, 3 years).
        3. Key Topics: List 3-5 of the most important skills or subjects taught.

        Format your response as:
        Program Name: [Program Name]
        Duration: [Duration]
        Key Topics: [Topic 1, Topic 2, Topic 3]
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        summarizer_chain = prompt | _llm | StrOutputParser()
        
        summary_text = summarizer_chain.invoke({"document": text_content[:8000]}) # Use first 8k characters for speed
        
        # Parse the summary text
        summary_dict = {}
        for line in summary_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                summary_dict[key.strip()] = value.strip()
        
        summary_dict['source'] = os.path.basename(pdf_path)
        return summary_dict

    except Exception as e:
        return {
            "Program Name": "Error processing file",
            "Duration": "N/A",
            "Key Topics": str(e),
            "source": os.path.basename(pdf_path)
        }

def ingest_pdfs_to_faiss(pdf_folder: str = PDFS_FOLDER, index_path: str = FAISS_INDEX_PATH):
    """Loads PDFs, splits them, creates embeddings and stores in FAISS index on disk."""
    docs = []
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{pdf_folder}'. Put your program PDFs there.")

    for fname in pdf_files:
        full_path = os.path.join(pdf_folder, fname)
        loader = PyPDFLoader(full_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(index_path):
        try:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vectorstore.save_local(index_path)
    else:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(index_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------------- Build LLM + RAG chain -------------------------
def get_llm():
    """Initializes and returns the LLM, preferring Groq."""
    llm = None
    if ChatGroq and GROQ_API_KEY:
        try:
            llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, api_key=GROQ_API_KEY)
        except Exception as e:
            st.error(f"Error initializing ChatGroq: {e}. Falling back to OpenAI (if configured).")
    if llm is None and ChatOpenAI:
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        except Exception as e:
            st.error(f"Error initializing ChatOpenAI: {e}.")
    return llm

def build_rag_chain(retriever, llm):
    """Constructs the history-aware RAG chain."""
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

    qa_system_prompt = (
        "You are an expert admissions assistant for PragyanAI. Use the following pieces of retrieved context to answer the question. "
        "Be helpful, friendly, and professional. If you don't know the answer from the context, say that you don't have that information. "
        "Keep the answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ------------------------- Recommendation logic -------------------------
def recommend_program(profile: Dict[str, Any]) -> str:
    """Simple rule-based recommendation."""
    try:
        sem = int(profile.get('semester', 1)) if profile.get('semester') else 1
    except ValueError:
        sem = 1
    try:
        score = float(profile.get('academic_score') or 0)
    except ValueError:
        score = 0.0

    if sem <= 2 or score < 60:
        return "Starter: AI Fundamentals & Python for AI (Beginner) - 6 weeks"
    elif sem <= 6 and score < 75:
        return "Intermediate: Applied ML & NLP Projects - 8 weeks"
    else:
        return "Advanced: Deep Learning, Generative AI & Agentic Systems - 12 weeks"

# ------------------------- Streamlit App --------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="PragyanAI Sales Agent", layout="wide")
    st.title("PragyanAI Program Advisor")
    st.image("PragyanAI_Transperent.png") 
    # --- Lead Capture Sidebar ---
    with st.sidebar:
        st.header("Your Details")
        st.info("Provide your details for personalized advice and follow-up.")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        college = st.text_input("College")
        branch = st.text_input("Branch")
        semester = st.text_input("Semester (number)")
        academic_score = st.text_input("Academic Score (percentage)")

        if st.button("Register Interest"):
            if name and email and phone:
                try:
                    coll = get_mongo_collection()
                    lead = { "name": name, "email": email, "phone": phone, "college": college, 
                             "branch": branch, "semester": semester, "academic_score": academic_score, 
                             "created_at": time.time() }
                    coll.insert_one(lead)
                    st.success("Thank you! Your details have been saved.")
                except Exception as e:
                    st.error(f"Error saving lead: {e}")
            else:
                st.warning("Please fill in at least Name, Email, and Phone.")

    # --- Main Application Logic ---
    llm = get_llm()
    if llm is None:
        st.error("Could not initialize an LLM. Please check your API keys (GROQ_API_KEY or OPENAI_API_KEY).")
        return

    # --- Program Summaries Section ---
    pdf_files = []
    if os.path.exists(PDFS_FOLDER):
        pdf_files = [os.path.join(PDFS_FOLDER, f) for f in os.listdir(PDFS_FOLDER) if f.lower().endswith('.pdf')]

    if not pdf_files:
        st.warning(f"No PDF brochures found in the '{PDFS_FOLDER}' directory. Please add program PDFs to enable features.", icon="⚠️")
        return

    program_summaries = [summarize_pdf(pdf_path, llm) for pdf_path in pdf_files]
    with st.expander("**Click here to see a summary of our available programs**", expanded=True):
        if program_summaries:
            df = pd.DataFrame(program_summaries)
            st.table(df)
        else:
            st.info("No program summaries could be generated.")
            
    # --- Updated Introduction and Chat Interface ---
    st.markdown("---")
    st.markdown("### Being PragyanAI Program AI Assistant, how can I help you today?")
    st.write("I can answer questions about the programs listed above or help you choose the right one.")

    try:
        retriever = ingest_pdfs_to_faiss()
        rag_chain = build_rag_chain(retriever, llm)
    except Exception as e:
        st.error(f"Error setting up the RAG pipeline: {e}")
        return

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_input = st.chat_input("Ask me about program details, fees, or eligibility...")
    
    # Display chat history
    for author, message in st.session_state.history:
        with st.chat_message(author):
            st.markdown(message)

    if user_input:
        st.session_state.history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            try:
                res = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.history})
                answer = res.get('answer', 'Sorry, I could not process that request.')
                
                st.session_state.history.append(("assistant", answer))
                with st.chat_message("assistant"):
                    st.markdown(answer)

                # Save conversation snippet to MongoDB
                try:
                    mongo_doc = { "question": user_input, "answer": answer, "timestamp": time.time() }
                    if email: mongo_doc["lead_email"] = email
                    get_mongo_collection().insert_one(mongo_doc)
                except Exception:
                    pass # Non-critical, don't block UI

            except Exception as e:
                st.error(f"RAG chain invocation failed: {e}")

# ------------------------- Entrypoint -----------------------------------
if __name__ == '__main__':
    run_streamlit_app()
