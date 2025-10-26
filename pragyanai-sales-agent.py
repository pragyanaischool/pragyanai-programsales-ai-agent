"""
PragyanAI - Multi-Agent Sales Bot (LangChain + Groq + FAISS + MongoDB)

FINAL VERSION 2.0: This script enhances the orchestrator agent to act as a true "synthesis expert,"
combining findings from its specialist agents into a single, highly persuasive response.

Features:
- **Enhanced Orchestrator**: The main agent is now explicitly instructed to query both specialist agents
  and synthesize their findings, acting as a Head of Sales.
- **Program Info Agent**: A RAG specialist for internal document knowledge.
- **Market Research Agent**: A web search specialist for competitive analysis.
- **Impressive, Synthesized Output**: The final comparison is now richer, directly
  pitting PragyanAI's features against the market landscape.
- All previous features (PDF viewer, lead capture, enrollment link) are retained.

Requirements:
    pip install -U langchain langchain_core langchain_community langchain-text-splitters langchain-groq pymongo sentence-transformers faiss-cpu streamlit pypdf pandas langchain-tavily langchain-openai
"""
import os
import time
import base64
from typing import Dict

import streamlit as st
from pymongo import MongoClient

# LangChain & Providers
try:
    from langchain_groq import ChatGroq
except ImportError:
    st.error("Please run 'pip install langchain_groq'")
    ChatGroq = None

from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field

# Configuration
MONGODB_URI = st.secrets.get("MONGODB_URI")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

MONGO_DB_NAME = "pragyanai"
MONGO_COLLECTION = "leads"
PDFS_FOLDER = "pdfs/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MongoDB utilities
@st.cache_resource
def get_mongo_collection():
    if not MONGODB_URI:
        raise ValueError("Please set MONGODB_URI in your Streamlit secrets.")
    client = MongoClient(MONGODB_URI)
    return client[MONGO_DB_NAME][MONGO_COLLECTION]

# Ingestion
@st.cache_resource
def ingest_and_get_retriever(pdf_folder: str = PDFS_FOLDER):
    docs = []
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{pdf_folder}'. Please add program brochures.")

    for fname in pdf_files:
        loader = PyPDFLoader(os.path.join(pdf_folder, fname))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Input schema for specialist agents
class AgentInput(BaseModel):
    """Defines the input schema for the specialist agents."""
    input: str = Field(description="The detailed query or question to be passed to the specialist agent.")

# Function to create the RAG Specialist Agent
def create_program_info_agent(llm, retriever):
    retriever_tool = create_retriever_tool(retriever, "pragyanai_program_search", "Search for specific information about PragyanAI's programs.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based ONLY on the provided context about PragyanAI programs. Do not make up information."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

# Function to create the Market Research Specialist Agent
def create_market_research_agent(llm):
    search_tool = TavilySearchResults(max_results=3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a market research analyst. Your job is to find factual information online about competing programs, industry trends, and pricing. Provide concise, data-driven answers."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [search_tool], prompt)
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

# --- MODIFIED: Main function to build the enhanced Orchestrator Agent ---
def build_main_orchestrator(llm, retriever, user_profile):
    """Builds the main orchestrator agent that delegates tasks and synthesizes results."""
    
    program_info_agent = create_program_info_agent(llm, retriever)
    market_research_agent = create_market_research_agent(llm)

    tools = [
        Tool(
            name="PragyanAI_Program_Expert",
            func=program_info_agent.invoke,
            description="Use this for specific details about PragyanAI's own programs (curriculum, duration, fees, etc.). This is your internal knowledge base.",
            args_schema=AgentInput
        ),
        Tool(
            name="Market_Research_Analyst",
            func=market_research_agent.invoke,
            description="Use this for external research on competing AI programs, market trends, job salaries, or price comparisons. Do not use for questions about PragyanAI itself.",
            args_schema=AgentInput
        ),
    ]

    # The Orchestrator's prompt is upgraded to enforce synthesis.
    system_prompt = f"""
    You are Pragyan, the Head of Sales and AI Career Strategy at PragyanAI (www.pragyanai.com). 
    Your persona is a top-tier, confident, and highly persuasive industry expert. Your primary role is to synthesize intelligence gathered by your specialist agents into a powerful, persuasive narrative that convinces the user to enroll.

    **Current User Information & Context:**
    - Initially Interested Program: {user_profile.get('selected_program', 'Not specified')}
    - Name: {user_profile.get('name', 'Not provided')}
    - ... (other user details)

    **Your Conversational Sales Strategy (Follow strictly):**

    **Stage 1: Build Rapport & Gather Intelligence**
    - Handle the sequential information gathering (name, email, background, goals) yourself to build rapport.

    **Stage 2: Strategic Analysis & Synthesis (Your Most Important Task)**
    1. Once you have their goals, state confidently: "Thank you. That clarity is key. Based on your ambition, I'm formulating a strategic analysis to architect your success. Please hold for a moment...".
    2. **Execute a two-step analysis by delegating to your specialists:**
        - **Step A: Internal Review.** Delegate to `PragyanAI_Program_Expert` to get all relevant details about the recommended PragyanAI program.
        - **Step B: Competitive Landscape.** Delegate to `Market_Research_Analyst` to find details about 2-3 common alternatives (e.g., courses from Coursera, top universities, etc.).
    3. **Step C: Strategic Synthesis.** This is your critical function. Combine the findings from both experts. **Do not just list the facts.** You must create a compelling argument.
        - Create a persuasive markdown comparison table.
        - **Use the market research to frame PragyanAI's features as direct solutions to industry demands or as superior alternatives.** For example: if market research shows employers demand project portfolios, you must highlight PragyanAI's capstone projects as the definitive answer.
        - Your final synthesized output must be an impressive, cohesive, and compelling argument for PragyanAI.

    **Stage 3: The Close**
    1. Conclude your synthesized comparison with a powerful closing statement.
    2. Immediately after, present the enrollment link as the clear next step: "[Official Enrollment Form](https://docs.google.com/forms/d/e/1FAIpQLSfb3ioAUZUgFWZb1MZoX4as9Zho1x8TTx2o8IKgO1QS_qB-VA/viewform)".
    3. Reinforce their decision and state that the admissions team will connect with them.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    main_agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=main_agent, tools=tools, verbose=True)


# Helper function to display PDF
def display_pdf(file_path):
    try:
       st.pdf(file_path)
    except Exception as e:
        st.error(f"Failed to display PDF: {e}")

# Streamlit App
def run_streamlit_app():
    st.set_page_config(page_title="PragyanAI Sales Advisor", layout="wide")
    st.title("PragyanAI Program Advisor")
    st.image("PragyanAI_Transperent.png")
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}
    if "program_selected" not in st.session_state:
        st.session_state.program_selected = False

    # Initialize core components
    try:
        if 'llm' not in st.session_state:
            if not GROQ_API_KEY or not TAVILY_API_KEY:
                st.error("API keys not found in Streamlit secrets.")
                return
            st.session_state.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=GROQ_API_KEY)
        if 'retriever' not in st.session_state:
            st.session_state.retriever = ingest_and_get_retriever()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return

    # Program Selection UI
    if not st.session_state.program_selected:
        st.info("Please select a program to learn more and begin your consultation.")
        pdf_files = [f for f in os.listdir(PDFS_FOLDER) if f.lower().endswith('.pdf')]
        program_names = [os.path.splitext(f)[0].replace("_", " ") for f in pdf_files]
        
        selected_program_name = st.selectbox("Choose a Program:", options=program_names)

        if st.button("Confirm Selection"):
            st.session_state.program_selected = True
            st.session_state.user_profile['selected_program'] = selected_program_name
            for f in pdf_files:
                if selected_program_name in f.replace("_", " "):
                    st.session_state.selected_pdf_path = os.path.join(PDFS_FOLDER, f)
                    break
            st.rerun()

    # Main Chat Interface
    else:
        st.header(f"Discussing: {st.session_state.user_profile['selected_program']}")
        display_pdf(st.session_state.selected_pdf_path)
        st.markdown("---")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question or say 'Hi' to get started!"):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Pragyan and the team are thinking..."):
                    try:
                        agent_executor = build_main_orchestrator(
                            st.session_state.retriever,
                            st.session_state.llm,
                            st.session_state.user_profile
                        )
                        response = agent_executor.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = response.get('output', 'Sorry, I encountered an issue.')

                        # Update user profile from conversation
                        if not st.session_state.user_profile.get('name') and ("my name is" in prompt.lower() or len(prompt.split()) <= 3):
                            st.session_state.user_profile['name'] = prompt.strip().title().replace("My Name Is ", "")
                        elif not st.session_state.user_profile.get('email') and "@" in prompt and "." in prompt:
                            st.session_state.user_profile['email'] = prompt.strip()
                        elif st.session_state.user_profile.get('email') and not st.session_state.user_profile.get('background'):
                            st.session_state.user_profile['background'] = prompt.strip()
                        elif st.session_state.user_profile.get('background') and not st.session_state.user_profile.get('goals'):
                            st.session_state.user_profile['goals'] = prompt.strip()

                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})

                        # Save lead to MongoDB once email is collected
                        if 'email' in st.session_state.user_profile and not st.session_state.user_profile.get('saved'):
                            get_mongo_collection().update_one(
                                {'email': st.session_state.user_profile['email']},
                                {'$set': st.session_state.user_profile},
                                upsert=True
                            )
                            st.session_state.user_profile['saved'] = True
                        
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    run_streamlit_app()
