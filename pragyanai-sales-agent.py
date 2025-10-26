"""
PragyanAI - Advanced Agentic Sales Bot (LangChain + Groq + FAISS + MongoDB)

ENHANCED: This is the complete version with a program selection screen, an embedded PDF viewer,
and a more structured, context-aware conversational flow with an enrollment call-to-action.

Features:
- Lists programs from PDF files and allows user selection.
- Displays the selected PDF brochure directly in the app.
- Initiates a stateful, interactive conversation to capture lead details.
- Performs competitive analysis and generates persuasive comparisons.
- Presents a Google Form enrollment link as the final call-to-action.
- Stores lead details and conversation history to MongoDB Atlas.

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

from langchain.agents import AgentExecutor, create_tool_calling_agent
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

# Build Agent Executor
def build_agent_executor(retriever, llm, user_profile):
    retriever_tool = create_retriever_tool(retriever, "pragyanai_program_search", "Search for specific information about PragyanAI's programs.")
    search_tool = TavilySearchResults(max_results=3)
    tools = [retriever_tool, search_tool]

    system_prompt = f"""
    You are Pragyan, a Senior Program Advisor at PragyanAI. Your persona is expert, confident, and highly persuasive. Your goal is to convert the user into a lead by convincing them that a PragyanAI program is the best choice for their career.

    **Current User Information & Context:**
    - Initially Interested Program: {user_profile.get('selected_program', 'Not specified')}
    - Name: {user_profile.get('name', 'Not provided')}
    - Email: {user_profile.get('email', 'Not provided')}
    - Background: {user_profile.get('background', 'Not provided')}
    - Goals: {user_profile.get('goals', 'Not provided')}

    **Your Conversational Sales Strategy (Follow these steps strictly):**

    **Stage 1: Sequential Information Gathering**
    1. If you don't know the user's name, your FIRST question MUST be "I can certainly help you with the '{user_profile.get('selected_program', 'program')}'. To get started, could you please tell me your full name?". Do not ask anything else.
    2. Once you have their name, address them by it. Your next question MUST be "Great to meet you, {user_profile.get('name', 'there')}! What's the best email to reach you at for further details?".
    3. Once you have their email, your next question MUST be "Thanks! Now, to ensure this program is the perfect fit, could you tell me about your current academic or professional background?".
    4. After they answer, ask about their career goals: "That's a great background. What are your long-term career aspirations?".
    
    **Stage 2: Analysis and Persuasive Recommendation**
    1. Once you have their background and goals, say "Thank you for sharing that. Based on your goal to become a {user_profile.get('goals', 'specialist in your field')}, I am running an analysis to confirm the best path for you. One moment...".
    2. Use `pragyanai_program_search` to find details about the selected program and confirm it's a good fit.
    3. Use `tavily_search_results_json` to find 2-3 common alternatives (e.g., "Coursera data science", "university online masters in AI").
    4. Present your recommendation for the PragyanAI program. Create a detailed, persuasive markdown comparison table with columns: "Feature", "PragyanAI Program", and "Typical Alternatives (MOOCs, Self-Study)".
    5. The table MUST highlight PragyanAI's strengths: `Practical Projects`, `Live Mentorship`, `Career Support & Placement Guarantee`, `Structured Curriculum`.

    **Stage 3: Closing and Enrollment**
    1. Conclude the comparison with a strong closing statement like: "As you can see, {user_profile.get('name')}, while other options exist, our program is engineered for tangible career outcomes and represents a direct investment in your future."
    2. **IMMEDIATELY AFTER** the closing statement, present the enrollment link as the clear next step. Your response MUST be:
    "If you are keen to enroll, you can take the first step by filling out our enrollment form here: [Enrollment Form](https://docs.google.com/forms/d/e/1FAIpQLSfb3ioAUZUgFWZb1MZoX4as9Zho1x8TTx2o8IKgO1QS_qB-VA/viewform)

    This is a very good decision for your career. Once you fill out the form, our admissions team will connect with you for the next steps.

    Do you have any other questions I can help with?"
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Helper function to display PDF
def display_pdf(file_path):
    try:
        #with open(file_path, "rb") as f:
       #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
       # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
       # st.markdown(pdf_display, unsafe_allow_html=True)
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
                with st.spinner("Pragyan is thinking..."):
                    try:
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
