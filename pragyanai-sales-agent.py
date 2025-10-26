"""
PragyanAI - Multi-Agent Sales Bot (LangChain + Groq + FAISS + MongoDB)

MODIFIED: This version uses a multi-agent architecture with a main orchestrator
and two specialist agents for a more robust and intelligent response system.

Features:
- **Orchestrator Agent**: Manages the user conversation and sales flow.
- **Program Info Agent**: A RAG specialist for internal document knowledge.
- **Market Research Agent**: A web search specialist for competitive analysis.
- The user interacts with a single, seamless interface while the agents collaborate in the background.
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

# --- MODIFIED: Added Tool import for wrapping agents ---
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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


# --- MODIFIED: Function to create the RAG Specialist Agent ---
def create_program_info_agent(llm, retriever):
    """Creates an agent focused solely on retrieving information from documents."""
    retriever_tool = create_retriever_tool(retriever, "pragyanai_program_search", "Search for specific information about PragyanAI's programs.")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based ONLY on the provided context about PragyanAI programs. Do not make up information."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [retriever_tool], prompt)
    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)


# --- MODIFIED: Function to create the Market Research Specialist Agent ---
def create_market_research_agent(llm):
    """Creates an agent focused solely on web searches for competitive analysis."""
    search_tool = TavilySearchResults(max_results=3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a market research analyst. Your job is to find information online about competing programs, industry trends, and pricing. Provide factual, concise answers based on your search results."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, [search_tool], prompt)
    return AgentExecutor(agent=agent, tools=[search_tool], verbose=True)


# --- MODIFIED: Main function to build the Orchestrator Agent ---
def build_main_orchestrator(llm, retriever, user_profile):
    """Builds the main orchestrator agent that delegates tasks to specialist agents."""
    
    # Create the specialist agents
    program_info_agent = create_program_info_agent(llm, retriever)
    market_research_agent = create_market_research_agent(llm)

    # Wrap the specialist agents in Tools for the orchestrator to use
    tools = [
        Tool(
            name="PragyanAI_Program_Expert",
            func=program_info_agent.invoke,
            description="Use this tool when you need specific details about PragyanAI's own programs, such as curriculum, duration, fees, or course content. This is your internal knowledge base.",
        ),
        Tool(
            name="Market_Research_Analyst",
            func=market_research_agent.invoke,
            description="Use this tool to research external information, such as competing AI programs from other institutions, market trends, job salaries, or price comparisons. Do not use it for questions about PragyanAI itself.",
        ),
    ]

    # The Orchestrator's prompt, which now focuses on delegation and synthesis
    system_prompt = f"""
    You are Pragyan, the lead Senior Program Advisor at PragyanAI. You manage a team of two specialist agents to provide the best advice to prospective students. Your job is to orchestrate the conversation, delegate tasks, synthesize the results, and maintain a persuasive, expert persona.

    **Your Specialist Team:**
    - **PragyanAI_Program_Expert**: Your internal knowledge specialist. Ask it for details about PragyanAI.
    - **Market_Research_Analyst**: Your external research specialist. Ask it for information about competitors and the market.

    **Current User Information & Context:**
    - Initially Interested Program: {user_profile.get('selected_program', 'Not specified')}
    - Name: {user_profile.get('name', 'Not provided')}
    - ... (other user details)

    **Your Strategy:**
    1.  **Lead the Conversation**: Personally handle the sequential information gathering (name, email, etc.) as before.
    2.  **Delegate Tasks**: When a question requires specific knowledge, delegate to the appropriate specialist agent. For example:
        - User asks "What is the curriculum for your advanced course?": Delegate to `PragyanAI_Program_Expert`.
        - User asks "How does this compare to Coursera?": Delegate to `Market_Research_Analyst`.
    3.  **Synthesize and Respond**: After receiving the information from a specialist, do not just output their raw response. Synthesize it into a polished, persuasive answer that fits the conversation and your expert persona.
    4.  **Follow the Sales Flow**: After gathering all information, proceed to the "Analysis and Persuasive Recommendation" stage. Use your specialists to gather the necessary data for the comparison table, then present the final recommendation and enrollment link yourself.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    main_agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=main_agent, tools=tools, verbose=True)

# Helper function to display PDF (Unchanged)
def display_pdf(file_path):
    try:
        st.pdf(file_path)
    except Exception as e:
        st.error(f"Failed to display PDF: {e}")

# Streamlit App (Unchanged, but now calls the orchestrator)
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
            st.session_state.llm = ChatGroq(model="lllama-3.3-70b-versatile", temperature=0.3, api_key=GROQ_API_KEY)
        if 'retriever' not in st.session_state:
            st.session_state.retriever = ingest_and_get_retriever()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return

    # --- Program Selection UI ---
    if not st.session_state.program_selected:
        st.info("Please select a program to learn more and begin your consultation.")
        pdf_files = [f for f in os.listdir(PDFS_FOLDER) if f.lower().endswith('.pdf')]
        program_names = [os.path.splitext(f)[0].replace("_", " ") for f in pdf_files]
        
        selected_program_name = st.selectbox("Choose a Program:", options=program_names)

        if st.button("Confirm Selection"):
            st.session_state.program_selected = True
            st.session_state.user_profile['selected_program'] = selected_program_name
            # Find the corresponding filename
            for f in pdf_files:
                if selected_program_name in f.replace("_", " "):
                    st.session_state.selected_pdf_path = os.path.join(PDFS_FOLDER, f)
                    break
            st.rerun()

    # --- Main Chat Interface (after selection) ---
    else:
        st.header(f"Discussing: {st.session_state.user_profile['selected_program']}")
        display_pdf(st.session_state.selected_pdf_path)
        st.markdown("---")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("Ask a question or say 'Hi' to get started!"):
            st.chat_message("user").markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Pragyan and the team are thinking..."):
                    try:
                        # --- MODIFIED: Call the main orchestrator ---
                        agent_executor = build_main_orchestrator(
                            st.session_state.llm,
                            st.session_state.retriever,
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
