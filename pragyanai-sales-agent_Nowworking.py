"""
🤖 PragyanAI - Multi-Agent Sales Chat Bot (Version 2.3)
--------------------------------------------------------
Fixes:
- Replaced deprecated `create_tool_calling_agent()` with `create_react_agent()`
- Added required `{tools}` and `{tool_names}` placeholders in prompts
- Fixed "Prompt missing required variables: {'tools', 'tool_names'}"
- Stable for LangChain >= 0.3 and Groq >= 0.5

Features:
- FAISS + MongoDB RAG pipeline
- Multi-Agent system: Program Info Agent, Market Research Agent, Sales Orchestrator
- Uses ChatGroq (LLaMA3) + Tavily Search
- Streamlit Frontend
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from pymongo import MongoClient
from datetime import datetime

# -----------------------------------------------------------
# 🧩 1. Streamlit UI Setup
# -----------------------------------------------------------

st.set_page_config(page_title="PragyanAI Sales Bot", layout="wide")
st.title("🤖 PragyanAI - Agentic Sales Chat Bot")

# -----------------------------------------------------------
# 🗝️ 2. API Keys and Secrets
# -----------------------------------------------------------

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    MONGO_URI = st.secrets["MONGODB_URI"]
except Exception:
    st.error("Missing Streamlit secrets! Please configure GROQ_API_KEY, TAVILY_API_KEY, and MONGO_URI.")
    st.stop()

# -----------------------------------------------------------
# 🧠 3. LLM Initialization
# -----------------------------------------------------------

llm = ChatGroq(
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile"
)

# -----------------------------------------------------------
# 🧩 4. MongoDB Setup
# -----------------------------------------------------------

client = MongoClient(MONGO_URI)
db = client["pragyanai_sales"]
collection = db["leads"]

# -----------------------------------------------------------
# 📄 5. Document Ingestion (FAISS + Embeddings)
# -----------------------------------------------------------

def ingest_pdfs(uploaded_files):
    all_docs = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")

    st.success("✅ Documents processed and indexed successfully!")
    return vectorstore

# -----------------------------------------------------------
# 🔍 6. RAG Retriever
# -----------------------------------------------------------

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        st.warning("⚠️ No FAISS index found. Please upload PDFs first.")
        return None

# -----------------------------------------------------------
# 🧩 7. Agent Prompts with ReAct-Compatible Variables
# -----------------------------------------------------------

# --- Program Info Agent ---
program_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are the Program Info Agent for PragyanAI.\n"
     "You specialize in understanding uploaded program brochures and course PDFs.\n"
     "You can use the following tools:\n{tools}\n\n"
     "Tool names: {tool_names}\n"
     "Answer accurately using document data."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# --- Market Research Agent ---
market_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Market Research Agent for PragyanAI.\n"
     "You research competitors, trends, and pricing.\n"
     "You can use the following tools:\n{tools}\n\n"
     "Tool names: {tool_names}\n"
     "Provide summarized insights."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# --- Sales Orchestrator Agent ---
orchestrator_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Sales Orchestrator Agent.\n"
     "You coordinate between program details and market data to generate persuasive sales messages.\n"
     "Use the following tools if needed:\n{tools}\n\n"
     "Tool names: {tool_names}\n"
     "Be professional, engaging, and data-driven."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# -----------------------------------------------------------
# 🧰 8. Tool Setup
# -----------------------------------------------------------

retriever_tool = Tool(
    name="Program Info Retriever",
    func=lambda q: get_vectorstore().as_retriever().get_relevant_documents(q),
    description="Useful for answering questions about the uploaded program PDFs."
)

tavily_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

# -----------------------------------------------------------
# 🤖 9. Agent Creation
# -----------------------------------------------------------

program_agent = create_react_agent(llm, [retriever_tool], program_prompt)
program_executor = AgentExecutor(agent=program_agent, tools=[retriever_tool], verbose=True)

market_agent = create_react_agent(llm, [tavily_tool], market_prompt)
market_executor = AgentExecutor(agent=market_agent, tools=[tavily_tool], verbose=True)

orchestrator_agent = create_react_agent(llm, [retriever_tool, tavily_tool], orchestrator_prompt)
orchestrator_executor = AgentExecutor(agent=orchestrator_agent, tools=[retriever_tool, tavily_tool], verbose=True)

# -----------------------------------------------------------
# 💬 10. Main Orchestration Logic
# -----------------------------------------------------------

def run_sales_pipeline(user_query):
    with st.spinner("🤔 Generating insights..."):
        prog_resp = program_executor.invoke({"input": user_query})
        market_resp = market_executor.invoke({"input": user_query})

        combined_input = f"""
        User query: {user_query}
        Program details: {prog_resp.get('output', '')}
        Market analysis: {market_resp.get('output', '')}
        """
        final_resp = orchestrator_executor.invoke({"input": combined_input})

    return {
        "program": prog_resp.get("output", ""),
        "market": market_resp.get("output", ""),
        "sales": final_resp.get("output", "")
    }

# -----------------------------------------------------------
# 📤 11. Streamlit UI Flow
# -----------------------------------------------------------

uploaded_files = st.file_uploader("📄 Upload course brochures or PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    ingest_pdfs(uploaded_files)

query = st.text_input("Ask about PragyanAI programs, competitors, or market insights:")
if st.button("Generate Sales Response") and query:
    result = run_sales_pipeline(query)
    st.subheader("🧾 Program Info")
    st.write(result["program"])
    st.subheader("📊 Market Insights")
    st.write(result["market"])
    st.subheader("💬 Sales Pitch")
    st.write(result["sales"])

# -----------------------------------------------------------
# 🧾 12. Lead Capture
# -----------------------------------------------------------

with st.expander("💡 Interested Lead Form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone")
    if st.button("Submit Lead"):
        collection.insert_one({
            "name": name,
            "email": email,
            "phone": phone,
            "timestamp": datetime.now()
        })
        st.success("✅ Lead saved successfully!")
