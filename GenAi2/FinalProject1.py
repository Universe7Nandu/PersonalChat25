# FinalProject1_updated_v3.py
import sys
import os

# 1. SQLITE3 PATCH (MUST BE FIRST)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    raise RuntimeError("Install pysqlite3-binary: pip install pysqlite3-binary")

# 2. IMPORTS (AFTER SQLITE PATCH)
import asyncio
import nest_asyncio
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# 3. CONFIGURATION
GROQ_API_KEY = "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"  # Replace if needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_SETTINGS = {
    "persist_directory": "resume_db",
    "collection_name": "resume_collection"
}

# 4. ASYNC SETUP
nest_asyncio.apply()

# 5. CORE FUNCTIONS
def initialize_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_SETTINGS["persist_directory"],
        embedding_function=embeddings,
        collection_name=CHROMA_SETTINGS["collection_name"]
    )

def process_pdf(file):
    try:
        pdf = PdfReader(file)
        return "\n".join(page.extract_text() for page in pdf.pages)
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
        return ""

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# 6. STREAMLIT UI
def main():
    st.set_page_config(
        page_title="Nandesh's AI Resume Assistant", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Modern Glassmorphic CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #1d2b64, #f8cdda);
        font-family: 'Roboto', sans-serif;
    }
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 40px;
    }
    header {
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    h1 {
        font-size: 3em;
        color: #fff;
        margin: 0;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
        color: #fff;
        padding: 20px;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffdd57;
    }
    [data-testid="stSidebar"] a {
        color: #ffdd57;
        text-decoration: none;
    }
    /* Chat Bubble Styles */
    .chat-box {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .user-message {
        color: #007BFF;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .bot-message {
        color: #333;
        line-height: 1.6;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff7e5f, #feb47b);
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        color: #fff;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.03);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .process-btn {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<header><h1>AI Resume Assistant 🤖</h1></header>", unsafe_allow_html=True)
    
    # Main layout container with two columns
    col1, col2 = st.columns([1, 2])
    
    # Left Column: Resume Section
    with col1:
        st.subheader("Resume Upload & Processing")
        uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf", key="resume_pdf")
        if uploaded_file:
            st.session_state.uploaded_resume = uploaded_file
            if "resume_processed" not in st.session_state:
                st.session_state.resume_processed = False
            if not st.session_state.resume_processed:
                if st.button("Process Resume", key="process_btn", help="Click to extract and index your resume"):
                    with st.spinner("Processing resume..."):
                        text = process_pdf(uploaded_file)
                        if text:
                            chunks = chunk_text(text)
                            vector_store = initialize_vector_store()
                            vector_store.add_texts(chunks)
                            st.session_state.resume_processed = True
                            st.success(f"Processed {len(chunks)} resume sections")
            else:
                st.info("Resume processed successfully!")
        else:
            st.info("Upload your resume PDF to enrich your chat responses.")
    
    # Right Column: Chat Section (always available)
    with col2:
        st.subheader("Chat with AI")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        user_query = st.text_input("Your message:")
        
        if user_query:
            with st.spinner("Generating response..."):
                # If resume has been processed, include context from the resume
                if st.session_state.get("resume_processed", False):
                    vector_store = initialize_vector_store()
                    docs = vector_store.similarity_search(user_query, k=3)
                    context = "\n".join([d.page_content for d in docs])
                    prompt = f"Context: {context}\nQuestion: {user_query}"
                else:
                    prompt = user_query
                llm = ChatGroq(
                    temperature=0.7,
                    groq_api_key=GROQ_API_KEY,
                    model_name="mixtral-8x7b-32768"
                )
                response = asyncio.run(llm.ainvoke([{
                    "role": "user",
                    "content": prompt
                }]))
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": response.content
                })
        
        # Display Chat History in stylish chat bubbles
        for chat in st.session_state.chat_history:
            st.markdown(f"""
            <div class="chat-box">
                <p class="user-message">You: {chat['question']}</p>
                <p class="bot-message">AI: {chat['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
