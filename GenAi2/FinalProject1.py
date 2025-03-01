# FinalProject1_updated.py
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
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
    /* Global Styles */
    body {
        background: #f4f7f6;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #343a40;
    }
    /* Header Styles */
    header, .stHeader {
        background: linear-gradient(135deg, #0062E6, #33AEFF);
        color: #fff;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: #343a40;
        color: #fff;
        padding: 20px;
    }
    [data-testid="stSidebar"] a {
        color: #ffdd57;
        text-decoration: none;
    }
    [data-testid="stSidebar"] a:hover {
        text-decoration: underline;
    }
    /* Chat Box Styles */
    .chat-box {
        background: #fff;
        border: 1px solid #dfe6e9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Message Styles */
    .user-message {
        color: #007BFF;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .bot-message {
        color: #343a40;
        line-height: 1.5;
    }
    /* Input Styles */
    div[data-baseweb="input"] > div {
        border-radius: 8px;
        border: 1px solid #ced4da;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    /* Button Styles */
    button[kind] {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #33AEFF, #0062E6);
        color: #fff;
        border: none;
        padding: 10px 20px;
        transition: transform 0.2s ease;
    }
    button[kind]:hover {
        transform: translateY(-2px);
    }
    /* Scrollbar Styles */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #b3b3b3;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with About info and resume update option
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Nandesh Kalashetti**  
        GenAI Developer & Full-Stack Engineer  
        [LinkedIn](https://linkedin.com/in/nandesh-kalashetti) | 
        [GitHub](https://github.com/Universe7Nandu)
        """)
        st.file_uploader("Update Resume", type="pdf", key="resume_update")

    st.header("AI Resume Assistant 🤖")
    
    # Resume File Upload and Processing Section
    uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf", key="resume_pdf")
    if uploaded_file:
        # Save file in session_state
        st.session_state.uploaded_resume = uploaded_file
        if "resume_processed" not in st.session_state:
            st.session_state.resume_processed = False
        if not st.session_state.resume_processed:
            if st.button("Process Resume"):
                with st.spinner("Processing resume..."):
                    text = process_pdf(uploaded_file)
                    if text:
                        chunks = chunk_text(text)
                        vector_store = initialize_vector_store()
                        vector_store.add_texts(chunks)
                        st.session_state.resume_processed = True
                        st.success(f"Processed {len(chunks)} resume sections")
        else:
            st.info("Resume has been processed. You can now ask questions about your qualifications.")
    
    # Chat Interface Section
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_query = st.text_input("Ask about my qualifications:")
    
    if user_query:
        if not st.session_state.get("resume_processed", False):
            st.warning("Please upload and process your resume first!")
        else:
            with st.spinner("Generating response..."):
                vector_store = initialize_vector_store()
                llm = ChatGroq(
                    temperature=0.7,
                    groq_api_key=GROQ_API_KEY,
                    model_name="mixtral-8x7b-32768"
                )
                # Retrieve context from resume embeddings
                docs = vector_store.similarity_search(user_query, k=3)
                context = "\n".join([d.page_content for d in docs])
                # Generate response using async function
                response = asyncio.run(llm.ainvoke([{
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {user_query}"
                }]))
                st.session_state.chat_history.append({
                    "question": user_query,
                    "answer": response.content
                })
    
    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.container():
            st.markdown(f"""
            <div class="chat-box">
                <p class="user-message">You: {chat['question']}</p>
                <p class="bot-message">AI: {chat['answer']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
