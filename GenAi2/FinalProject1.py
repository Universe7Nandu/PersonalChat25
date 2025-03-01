
# FinalProject1.py
import sys
import os

# 1. CRITICAL SQLITE3 FIX - MUST BE FIRST
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Configuration (Direct API Key Inclusion)
GROQ_API_KEY = "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"  # Replace with actual key
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def process_pdf(file):
    """Extract text from PDF with error handling"""
    try:
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
        return ""

def create_retriever(text):
    """Create ChromaDB vector store from text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    return Chroma.from_texts(
        texts=docs,
        embedding=embeddings,
        persist_directory="resume_db"
    ).as_retriever()

def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
    
    # UI Header
    st.header("📄 Smart Resume Analyzer", divider="rainbow")
    
    # File Upload Section
    uploaded_file = st.file_uploader("Upload resume PDF", type="pdf")
    
    if uploaded_file:
        # Processing Section
        with st.status("Analyzing resume...", expanded=True) as status:
            # Step 1: PDF Processing
            st.write("🔍 Extracting text from PDF...")
            text = process_pdf(uploaded_file)
            
            if not text:
                st.error("Failed to extract text")
                return

            # Step 2: Vector DB Creation
            st.write("🧠 Creating knowledge base...")
            retriever = create_retriever(text)
            
            # Step 3: AI Setup
            st.write("🤖 Initializing AI engine...")
            groq_chat = ChatGroq(
                temperature=0.2,
                groq_api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768"
            )
            
            # Step 4: QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=retriever
            )
            
            status.update(label="Analysis Complete!", state="complete")

        # Results Display
        st.subheader("Resume Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Text Length", f"{len(text):,} chars")
            st.metric("Chunks Created", len(retriever.vectorstore.get()['documents']))
        
        with col2:
            st.metric("AI Model", "Mixtral-8x7B")
            st.metric("Embeddings", EMBEDDING_MODEL.split("/")[-1])
        
        # Query Section
        st.divider()
        user_query = st.text_input("Ask questions about the resume:")
        
        if user_query:
            with st.spinner("Generating response..."):
                result = qa_chain.invoke({"query": user_query})
                st.markdown(f"**Answer:** {result['result']}")

        # Debug Info (Hidden by default)
        with st.expander("Technical Details"):
            st.code(f"SQLite Version: {sys.modules['sqlite3'].sqlite_version}")
            st.write("Vector DB Path:", os.path.abspath("resume_db"))

if __name__ == "__main__":
    main()
