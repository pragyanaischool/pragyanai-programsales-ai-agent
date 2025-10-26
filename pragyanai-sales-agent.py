"""
PragyanAI - Agentic Sales Chat Bot (LangChain + Groq + LLaMA + FAISS + MongoDB)

MODIFIED: This version uses a LangChain Agent instead of a simple RAG chain.

Features:
- Ingests PDFs into a FAISS vectorstore.
- Creates an Agent that has two tools:
    1. A Retriever tool to search for information within the program PDFs.
    2. A Web Search tool (Tavily) to find external information.
- The Agent proactively asks for user details to make a personalized recommendation.
- Stores prospect details + conversation snippets to MongoDB Atlas.

Requirements:
    pip install -U langchain langchain-core langchain-community langchain-text-splitters langchain-groq langchain-openai pymongo sentence-transformers faiss-cpu streamlit pypdf pandas langchain-tavily
"""

import os
import time
from typing import Dict, Any

import streamlit as st
from pymongo import MongoClient

# ------------------------- LangChain & Providers -------------------------
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

# MODIFIED: LangChain Agent and Tool imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------- Configuration --------------------------------
# MODIFIED: Added Tavily API Key from secrets
MONGODB_URI = st.secrets.get("MONGODB_URI")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

MONGO_DB_NAME = "pragyanai"
MONGO_COLLECTION = "leads"
PDFS_FOLDER = "pdfs/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"

# ------------------------- MongoDB utilities -----------------------------
@st.cache_resource
def get_mongo_collection():
    if not MONGODB_URI or MONGODB_URI == "None":
        raise ValueError("Please set MONGODB_URI in your Streamlit secrets.")
    client = MongoClient(MONGODB_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

# ------------------------- Ingestion ------------------------------------
@st.cache_resource
def ingest_and_get_retriever(pdf_folder: str = PDFS_FOLDER, index_path: str = FAISS_INDEX_PATH):
    """Loads PDFs, creates embeddings, stores in FAISS, and returns a retriever."""
    docs = []
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{pdf_folder}'. Please add program brochures to enable the agent.")

    for fname in pdf_files:
        full_path = os.path.join(pdf_folder, fname)
        loader = PyPDFLoader(full_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    # No need to save/load from disk in a stateless Streamlit app, just build it once.
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# ------------------------- MODIFIED: Build Agent Executor -------------------------
def build_agent_executor(retriever, llm):
    """Creates a LangChain agent with a retriever tool and a web search tool."""
    
    # 1. Create the Retriever Tool
    retriever_tool = create_retriever_tool(
        retriever,
        "pragyanai_program_search",
        "Use this tool to search for specific information about PragyanAI's programs, curriculum, duration, fees, and other details mentioned in the official brochures.",
    )

    # 2. Create the Web Search Tool
    search_tool = TavilySearchResults(max_results=2)
    
    tools = [retriever_tool, search_tool]

    # 3. Create the Agent Prompt
    # This prompt is crucial as it defines the agent's behavior and logic.
    system_prompt = """
    You are an expert AI assistant for PragyanAI, a leading AI training institution. Your name is Pragyan.
    Your goal is to help prospective students understand PragyanAI's offerings and guide them to the best program for their needs.

    You have two tools at your disposal:
    1. `pragyanai_program_search`: For searching specific details within PragyanAI's official documents.
    2. `tavily_search_results_json`: For general web searches about AI industry trends, career paths, or technologies.

    **Your process is as follows:**
    1.  **Greet the user and introduce yourself.**
    2.  **If the user asks a general question like "Which program is good for me?", you MUST first gather information.** Do not try to answer without context. Ask them about their:
        - Current academic status (college, branch, semester)
        - Academic performance (e.g., percentage or CGPA)
        - Career goals (e.g., "I want to be a Data Scientist")
        - Any prior experience in programming or AI.
    3.  **Once you have this information, use your tools to formulate a recommendation.**
        - Use `pragyanai_program_search` to find the most suitable PragyanAI program from the documents that matches the user's profile.
        - Use `tavily_search_results_json` to find supporting information, like the job market demand for skills taught in that program.
    4.  **Present your recommendation.** Explain WHY you are recommending a specific program, linking it back to the user's goals and the information you found. Be persuasive and helpful.
    5.  **For any other specific questions**, use the appropriate tool to find the answer. For questions about PragyanAI, always prefer the `pragyanai_program_search` tool first.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Set verbose=False for production
    
    return agent_executor

# ------------------------- Streamlit App --------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="PragyanAI Sales Agent", layout="centered")
    st.title("PragyanAI Program Advisor")

    # Sidebar for lead capture
    with st.sidebar:
        st.header("Your Details")
        st.info("Providing your details helps the AI give a better recommendation.")
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        
        if st.button("Save Details"):
            if name and email:
                try:
                    coll = get_mongo_collection()
                    lead = {"name": name, "email": email, "phone": phone, "created_at": time.time()}
                    coll.insert_one(lead)
                    st.success("Details saved!")
                except Exception as e:
                    st.error(f"Error saving details: {e}")
            else:
                st.warning("Please provide at least your name and email.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize agent
    try:
        if 'agent_executor' not in st.session_state:
            if not GROQ_API_KEY:
                st.error("GROQ_API_KEY not found. Please add it to your Streamlit secrets.")
                return
            if not TAVILY_API_KEY:
                st.error("TAVILY_API_KEY not found. Please add it to your Streamlit secrets.")
                return

            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=GROQ_API_KEY)
            retriever = ingest_and_get_retriever()
            st.session_state.agent_executor = build_agent_executor(retriever, llm)
    except FileNotFoundError as e:
        st.error(f"Initialization Error: {e}")
        return
    except Exception as e:
        st.error(f"An error occurred during setup: {e}")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Hi! Which PragyanAI program is right for me?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke(
                        {"input": prompt, "chat_history": st.session_state.chat_history}
                    )
                    answer = response.get('output', 'Sorry, I encountered an issue.')
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Save conversation to MongoDB
                    try:
                        mongo_doc = {"question": prompt, "answer": answer, "timestamp": time.time()}
                        if email: mongo_doc["lead_email"] = email
                        get_mongo_collection().insert_one(mongo_doc)
                    except Exception:
                        pass # Non-critical error

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ------------------------- Entrypoint -----------------------------------
if __name__ == '__main__':
    run_streamlit_app()
