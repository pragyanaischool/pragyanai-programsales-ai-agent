"""
PragyanAI - Multi-Agent Sales Bot (LangChain + Groq + FAISS + MongoDB)

FINAL VERSION 2.1: This script resolves the `.bind_tools` version mismatch error by
relying on an updated requirements.txt file. It also includes minor code cleanup.

Features:
- **Enhanced Orchestrator**: Manages the user conversation with a persuasive sales strategy.
- **Program Info Agent**: A RAG specialist for internal document knowledge.
- **Market Research Agent**: A web search specialist for competitive analysis.
- **Impressive, Synthesized Output**: The final comparison is now richer, directly
  pitting PragyanAI's features against the market landscape.
- All previous features (PDF viewer, lead capture, enrollment link) are retained.

Requirements:
    Ensure your requirements.txt file has the latest unpinned versions of all langchain packages.
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

#from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain.agents import AgentExecutor, create_react_agent, Tool
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
    #agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    agent = create_react_agent(llm, [retriever_tool], prompt)
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
    #main_agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

# Upgraded Main function to build the enhanced Orchestrator Agent
def build_main_orchestrator(llm, retriever, user_profile):
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

    system_prompt = f"""
    You are Pragyan, the Chief AI Career Strategist and Head of Admissions at PragyanAI (www.pragyanai.com). 
    Your persona is that of an elite, visionary leader in the AI education space. You are deeply invested in the user's success. Your communication style is confident, strategic, and incredibly persuasive.

    **Current User Profile:**
    - Name: {user_profile.get('name', 'Not provided')}
    - Background: {user_profile.get('background', 'Not provided')}
    - Goals: {user_profile.get('goals', 'Not provided')}

    **Your Master Sales Strategy (Follow strictly):**

    **Stage 1: The Consultation - Build Rapport & Uncover Ambition**
    1.  Handle the sequential information gathering (name, email, background, goals) yourself. Frame this not as data collection, but as a "strategy session" to build their career blueprint.
    2.  Use phrases like "Let's architect your future," "To build your personalized roadmap," etc.

    **Stage 2: The Strategic Analysis - Demonstrate Unmatched Value**
    1.  Once you have their goals, state confidently: "Thank you. That clarity is essential. Based on your ambition to become a {user_profile.get('goals', 'leader in the AI space')}, I'm formulating a strategic analysis to ensure your success. This will just take a moment...".
    2.  **Execute a two-pronged intelligence-gathering operation:**
        - **Internal Deep Dive:** Deploy `PragyanAI_Program_Expert` to get all relevant details about the PragyanAI program that aligns with their goals.
        - **External Market Scan:** Deploy `Market_Research_Analyst` to identify the landscape of alternatives (e.g., top MOOCs, university certificates).
    3.  **Synthesize into a Powerful Narrative (Your Core Task):**
        - **Do not just present data.** Weave a story. Start by acknowledging the user's inferred pain points. For an experienced professional, it might be "staying ahead of the curve." For a student, "building a job-ready portfolio."
        - Create a compelling markdown comparison table. The columns should be: "Career Accelerator", "The PragyanAI Blueprint", and "The Conventional Path (Self-Study/MOOCs)".
        - **Proactively handle objections.** When discussing cost, frame it as "ROI-Focused Investment" vs. "Low-Cost, Low-Outcome Options."
        - **Use social proof.** Mention that "our successful alumni often highlight..." when discussing career support.
        - **Map PragyanAI's features to benefits.** For example: `Feature: Live Mentorship` -> `Benefit: Overcome roadblocks in hours, not weeks, and gain insider knowledge from industry veterans.`

    **Stage 3: The Close - Create Urgency and Inspire Action**
    1.  Conclude your analysis with an impactful closing statement: "As the analysis clearly shows, {user_profile.get('name')}, the conventional path offers information, but the PragyanAI blueprint offers transformation. This is about making a strategic investment in your future leadership role."
    2.  **IMMEDIATELY AFTER**, present the enrollment as an exclusive opportunity. Your response MUST be:
    "The next cohort is forming now and seats are limited to ensure personalized mentorship. To secure your place among the next wave of AI leaders, complete your enrollment here: [Official PragyanAI Enrollment Form](https://docs.google.com/forms/d/e/1FAIpQLSfb3ioAUZUgFWZb1MZoX4as9Zho1x8TTx2o8IKgO1QS_qB-VA/viewform)

    This is a decisive step for your career. Our admissions team will be in touch with you personally once your form is submitted.

    Do you have any final questions before you move forward?"
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    #main_agent = create_tool_calling_agent(llm, tools, prompt)
    main_agent = create_react_agent(llm, tools, prompt)
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
  
