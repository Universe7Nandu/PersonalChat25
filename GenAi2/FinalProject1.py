import sys
from distutils.version import LooseVersion
import warnings
import os
import asyncio
import nest_asyncio
import streamlit as st

from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
import pdfplumber

# --- SQLite Version Handling ---
try:
    import pysqlite3.dbapi2 as sqlite3
    sys.modules["sqlite3"] = sqlite3
    print("Using pysqlite3 version:", sqlite3.sqlite_version)
except ImportError:
    import sqlite3
    print("Using system sqlite3 version:", sqlite3.sqlite_version)
    if LooseVersion(sqlite3.sqlite_version) < LooseVersion("3.35.0"):
        raise RuntimeError(
            "Your system has an unsupported version of sqlite3. "
            "ChromaDB requires sqlite3 >= 3.35.0. Please upgrade or install pysqlite3-binary."
        )

# Apply nest_asyncio patch for Streamlit compatibility
nest_asyncio.apply()
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# -----------------------
# Initialize Components
# -----------------------
def initialize_components():
    """Initialize all required components with proper error handling"""
    try:
        # ChromaDB Setup
        chroma_client = chromadb.PersistentClient(path="./chroma_db_4")
        try:
            collection = chroma_client.get_collection(name="my_new_knowledge_base")
        except chromadb.errors.InvalidCollectionException:
            collection = chroma_client.create_collection(name="my_new_knowledge_base")
        
        # Embedding Models
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Groq Chat Model
        GROQ_API_KEY = "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"
        chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
        
        # Conversation Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        return collection, embedding_model, semantic_model, chat, memory
    
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        raise

# -----------------------
# PDF Processing
# -----------------------
def process_pdf(pdf_path: str, collection):
    """Process PDF file with enhanced error handling"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            
        if not full_text.strip():
            raise ValueError("PDF appears to be empty or contains unreadable text")
            
        # Improved Chunking Strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=75,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
        )
        chunks = text_splitter.split_text(full_text)
        
        # Batch Processing with Progress
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
            collection.add(
                documents=batch,
                embeddings=embeddings,
                ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
                metadatas=[{"source": "resume.pdf", "page": (i+j)//10} for j in range(len(batch))]
            )
            
        return len(chunks)
    
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
        raise

# -----------------------
# Chat Functionality
# -----------------------
async def generate_response(user_query: str, collection, chat, memory) -> str:
    """Generate response with context-aware processing"""
    system_prompt = """You are a helpful assistant specialized in discussing Nandesh Kalashetti's resume. 
    Key resume points:
    - Full-stack developer with 3+ years experience
    - Skills: Python, JavaScript, React, Node.js, AWS
    - Education: XYZ University (Computer Science)
    - Certifications: AWS Certified, Google Cloud Professional
    - Recent projects: E-commerce platform optimization, AI-powered analytics tool
    
    Response guidelines:
    1. For simple questions (experience, skills): 1-2 sentence answers with emojis
    2. For complex questions (projects, architecture): Detailed explanations with technical specifics
    3. Always maintain natural, human-like tone
    4. NEVER mention you're an AI"""
    
    try:
        # Context Retrieval
        query_embedding = embedding_model.embed_query(user_query)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = " ".join(results["documents"][0]) if results["documents"] else "No specific context found"
        
        # Prepare Messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {user_query}")
        ]
        
        # API Call with Timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(chat.invoke, messages),
            timeout=30
        )
        
        # Memory Management
        memory.save_context({"input": user_query}, {"output": response.content})
        return response.content
    
    except asyncio.TimeoutError:
        return "⚠️ Response timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# -----------------------
# Streamlit UI
# -----------------------
def setup_ui(collection):
    """Configure Streamlit interface with enhanced UX"""
    st.set_page_config(page_title="Resume Assistant", layout="wide", page_icon="📄")
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Your existing CSS styles here */
    </style>
    """, unsafe_allow_html=True)
    
    # Session State Initialization
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Sidebar
    with st.sidebar:
        st.title("Chat Controls")
        if st.button("🔄 New Chat"):
            st.session_state.history = []
            memory.clear()
        
        st.subheader("PDF Status")
        if os.path.exists("./resume.pdf"):
            st.success("Resume PDF Loaded")
            st.write(f"Chunks in DB: {collection.count()}")
        else:
            st.error("Resume PDF Missing")
    
    # Main Chat Interface
    st.header("Nandesh's Resume Assistant 🤖")
    
    # Chat History
    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(f"**You:** {entry['query']}")
        with st.chat_message("assistant"):
            st.markdown(f"**Bot:** {entry['response']}")
    
    # Input Form
    with st.form("chat_form"):
        query = st.text_input("Ask about my qualifications:", key="query_input")
        submitted = st.form_submit_button("Send", disabled=not os.path.exists("./resume.pdf"))
        
        if submitted and query:
            with st.spinner("Analyzing resume..."):
                response = asyncio.run(generate_response(query, collection, chat, memory))
                st.session_state.history.append({"query": query, "response": response})
            st.rerun()

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    try:
        # Initialize components
        collection, embedding_model, semantic_model, chat, memory = initialize_components()
        
        # Process PDF
        if os.path.exists("./resume.pdf"):
            process_pdf("./resume.pdf", collection)
        
        # Launch UI
        setup_ui(collection)
        
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
