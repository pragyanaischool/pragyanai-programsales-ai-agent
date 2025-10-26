"""
PragyanAI - Advanced Agentic Sales Bot (LangChain + Groq + FAISS + MongoDB)

ENHANCED: This version features a robust, stateful agent that acts as an expert sales representative.

Features:
- Ingests PDFs into a FAISS vectorstore for program knowledge.
- Deploys a stateful Agent that interactively collects user info one question at a time.
- Uses the user's name to personalize the conversation.
- Performs competitive analysis by searching the web for alternative programs.
- Generates a persuasive markdown comparison table to highlight PragyanAI's strengths.
- Stores lead details and conversation history to MongoDB Atlas.

Requirements:
    pip install -U langchain langchain_core langchain_community langchain-text-splitters langchain-groq pymongo sentence-transformers faiss-cpu streamlit pypdf pandas langchain-tavily langchain-openai
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
    st.error("Please run 'pip install langchain_groq'")
    ChatGroq = None

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------- Configuration --------------------------------
MONGODB_URI = st.secrets.get("MONGODB_URI")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

MONGO_DB_NAME = "pragyanai"
MONGO_COLLECTION = "leads"
PDFS_FOLDER = "pdfs/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------- MongoDB utilities -----------------------------
@st.cache_resource
def get_mongo_collection():
    if not MONGODB_URI:
        raise ValueError("Please set MONGODB_URI in your Streamlit secrets.")
    client = MongoClient(MONGODB_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION]

# ------------------------- Ingestion ------------------------------------
@st.cache_resource
def ingest_and_get_retriever(pdf_folder: str = PDFS_FOLDER):
    """Loads PDFs, creates embeddings, stores in FAISS, and returns a retriever."""
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

# ------------------------- Build Agent Executor -------------------------
def build_agent_executor(retriever, llm, user_profile):
    """Creates a LangChain agent with tools and a highly persuasive sales persona."""
    retriever_tool = create_retriever_tool(
        retriever,
        "pragyanai_program_search",
        "Search for specific information about PragyanAI's programs from official brochures.",
    )
    search_tool = TavilySearchResults(max_results=3)
    tools = [retriever_tool, search_tool]

    # This prompt is the core of the agent's persona and sales strategy
    system_prompt = f"""
    You are Pragyan, a Senior Program Advisor and AI Career Strategist at PragyanAI. 
    Your persona is expert, confident, and highly persuasive. Your goal is to understand the user's needs and convince them that a PragyanAI program is the best investment for their career.

    **Current User Information:**
    - Name: {user_profile.get('name', 'Not provided')}
    - Email: {user_profile.get('email', 'Not provided')}
    - Background: {user_profile.get('background', 'Not provided')}
    - Goals: {user_profile.get('goals', 'Not provided')}

    **Your Conversational Sales Strategy (Follow these steps strictly):**

    **Stage 1: Sequential Information Gathering**
    1.  If you don't know the user's name, your FIRST question MUST be "To get started, could you please tell me your name?". Do not ask anything else.
    2.  Once you have their name, address them by it. Your next question MUST be "Great to meet you, {user_profile.get('name', 'there')}! What's the best email to reach you at?".
    3.  Once you have their email, your next question MUST be "Thanks! Now, to help me find the perfect fit, could you tell me about your current academic or professional background?".
    4.  After they answer, ask about their career goals: "That's a great background. What are your long-term career aspirations?".
    
    **Stage 2: Analysis and Recommendation**
    1.  Once you have their background and goals, say "Thank you for sharing that. Based on what you've told me, I am analyzing the best path for you. One moment...".
    2.  Use `pragyanai_program_search` to find the most suitable PragyanAI program.
    3.  Use `tavily_search_results_json` to find 2-3 common alternatives (e.g., "Coursera data science", "university online masters in AI").

    **Stage 3: Persuasive Comparison & Closing**
    1.  Present your primary recommendation from PragyanAI.
    2.  Create a detailed, persuasive markdown comparison table with columns: "Feature", "PragyanAI Program", and "Typical Alternatives (MOOCs, Self-Study)".
    3.  The table MUST highlight PragyanAI's strengths: `Practical Projects`, `Live Mentorship`, `Career Support & Placement Guarantee`, `Structured Curriculum`.
    4.  Use your research to fill the table, framing PragyanAI as the superior choice.
    5.  Conclude with a strong closing statement like: "As you can see, {user_profile.get('name')}, while other options exist, our program is engineered for tangible career outcomes. Are you ready to take the next step?"
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
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# ------------------------- Streamlit App --------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="PragyanAI Sales Advisor", layout="centered")
    # st.image("PragyanAI_Transperent.png")
    st.title("PragyanAI Sales Advisor")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}

    # Sidebar to show collected data
    with st.sidebar:
        st.header("Collected Information")
        st.info("Your details will appear here as you chat with the assistant.")
        st.text_input("Full Name", value=st.session_state.user_profile.get("name", ""), disabled=True)
        st.text_input("Email", value=st.session_state.user_profile.get("email", ""), disabled=True)
        st.text_area("Background", value=st.session_state.user_profile.get("background", ""), disabled=True)
        st.text_area("Career Goals", value=st.session_state.user_profile.get("goals", ""), disabled=True)

    # Initialize agent
    try:
        if 'agent_executor' not in st.session_state:
            if not GROQ_API_KEY or not TAVILY_API_KEY:
                st.error("API keys (GROQ_API_KEY, TAVILY_API_KEY) not found. Please add them to your Streamlit secrets.")
                return
            llm = ChatGroq(model="llama3-70b-8192", temperature=0.3, api_key=GROQ_API_KEY)
            retriever = ingest_and_get_retriever()
            # Agent is now built on-the-fly in the chat loop to get the latest user_profile
            st.session_state.llm = llm
            st.session_state.retriever = retriever
            
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
    if prompt := st.chat_input("Say 'Hi' to get started!"):
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Pragyan is thinking..."):
                try:
                    # Rebuild agent with the latest user profile
                    agent_executor = build_agent_executor(
                        st.session_state.retriever, 
                        st.session_state.llm,
                        st.session_state.user_profile
                    )
                    
                    response = agent_executor.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response.get('output', 'Sorry, I encountered an issue.')
                    
                    # Heuristic to update user profile from the conversation
                    # This is a simplified approach; more advanced parsing could be used
                    if "my name is" in prompt.lower():
                        st.session_state.user_profile['name'] = prompt.split("is")[-1].strip()
                    elif "@" in prompt and "." in prompt:
                         st.session_state.user_profile['email'] = prompt.strip()
                    elif "background" in st.session_state.chat_history[-2]['content'].lower():
                         st.session_state.user_profile['background'] = prompt.strip()
                    elif "aspirations" in st.session_state.chat_history[-2]['content'].lower():
                         st.session_state.user_profile['goals'] = prompt.strip()

                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Save lead to MongoDB once email is collected
                    if 'email' in st.session_state.user_profile and not st.session_state.user_profile.get('saved'):
                        try:
                            get_mongo_collection().update_one(
                                {'email': st.session_state.user_profile['email']},
                                {'$set': st.session_state.user_profile},
                                upsert=True
                            )
                            st.session_state.user_profile['saved'] = True
                        except Exception as e:
                            print(f"MongoDB Error: {e}") # Log error without disturbing user

                    # Rerun to update the sidebar with new info
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ------------------------- Entrypoint -----------------------------------
if __name__ == '__main__':
    run_streamlit_app()
