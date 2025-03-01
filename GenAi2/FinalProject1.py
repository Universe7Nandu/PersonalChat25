# FinalProject1.py
# Requirements: Use the corrected requirements.txt from previous answer

import sys
import warnings
import os
import asyncio
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

# ----- Critical SQLite3 Fix -----
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now import ChromaDB
import chromadb
from chromadb.config import Settings

# Rest of imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
import pdfplumber

# -----------------------
# Suppress warnings
# -----------------------
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# -----------------------
# 1. Initialize Components
# -----------------------
def initialize_components():
    """Initialize all core components with error handling"""
    global chroma_client, collection, embedding_model, semantic_model, chat
    
    # ChromaDB setup
    chroma_client = chromadb.PersistentClient(path="./chroma_db_5")
    try:
        collection = chroma_client.get_collection(name="my_new_knowledge_base")
    except chromadb.errors.InvalidCollectionException:
        collection = chroma_client.create_collection(name="my_new_knowledge_base")
    
    # Model initialization
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Groq chat setup
    GROQ_API_KEY = "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"
    chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# -----------------------
# 2. Core Functions
# -----------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_recent_chat_history(n=8):
    """Return formatted chat history"""
    history = memory.load_memory_variables({}).get("chat_history", [])
    return "\n".join([f"{msg.type}: {msg.content}" for msg in history[-n:]])

def retrieve_context(query, top_k=3):
    """Enhanced context retrieval with fallback"""
    try:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return " ".join(results["documents"][0]) if results else "No context found."
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "Error retrieving context"

def evaluate_response(generated, context):
    """Improved similarity scoring with error handling"""
    try:
        gen_embed = semantic_model.encode(generated, convert_to_tensor=True)
        ctx_embed = semantic_model.encode(context, convert_to_tensor=True)
        return util.pytorch_cos_sim(gen_embed, ctx_embed).item()
    except Exception as e:
        print(f"Evaluation error: {e}")
        return 0.0

# -----------------------
# 3. PDF Processing & Chat
# -----------------------
def process_pdf(pdf_path):
    """End-to-end PDF processing pipeline"""
    if not os.path.exists(pdf_path):
        return "PDF file not found"
    
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    
    if not text.strip():
        return "No text extracted"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = [embedding_model.embed_query(c) for c in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"source": "resume.pdf"}] * len(batch)
        )
    
    return f"Processed {len(chunks)} chunks from PDF"

async def generate_response(user_query):
    """Optimized async response generation"""
    system_prompt = """..."""  # Keep your existing prompt
    
    try:
        context = retrieve_context(user_query)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context: {context}\nQuestion: {user_query}")
        ]
        
        response = await asyncio.to_thread(chat.invoke, messages)
        if not response.content.strip():
            return "I couldn't generate a proper response. Please try again."
        
        memory.save_context({"input": user_query}, {"output": response.content})
        score = evaluate_response(response.content, context)
        print(f"Evaluation Score: {score:.2f}")
        
        return response.content
        
    except Exception as e:
        print(f"Generation error: {e}")
        return "Sorry, I'm having trouble responding right now."

# -----------------------
# 4. Streamlit Interface
# -----------------------
def main():
    """Streamlit UI configuration"""
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    
    initialize_components()
    process_pdf("resume.pdf")  # Initial PDF processing
    
    st.title("Nandesh's AI Assistant")
    st.image("photo2.jpg", width=150)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    user_query = st.chat_input("Ask me about Nandesh's background:")
    
    if user_query:
        with st.spinner("Thinking..."):
            response = asyncio.run(generate_response(user_query))
            st.session_state.chat_history.append({
                "user": user_query,
                "assistant": response
            })
        
        for msg in st.session_state.chat_history[-5:]:
            with st.chat_message("user"):
                st.write(msg["user"])
            with st.chat_message("assistant"):
                st.write(msg["assistant"])

if __name__ == "__main__":
    main()
