import sys
import sqlite3
from distutils.version import LooseVersion

# --- SQLite Patch ---
if LooseVersion(sqlite3.sqlite_version) < LooseVersion("3.35.0"):
    try:
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        raise RuntimeError(
            "Your system has an unsupported version of SQLite3. "
            "ChromaDB requires SQLite3 >= 3.35.0. Please install pysqlite3-binary."
        )
import sys
import sqlite3
from distutils.version import LooseVersion
import warnings
import os
import asyncio
import nest_asyncio  # To allow nested asyncio loops in Streamlit
import streamlit as st

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
import pdfplumber

# Apply nest_asyncio patch
nest_asyncio.apply()

# --- Suppress warnings ---
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# -----------------------
# 1. Initialize ChromaDB, Embeddings, and Chat Model
# -----------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db_5")
try:
    collection = chroma_client.get_collection(name="my_new_knowledge_base")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(name="my_new_knowledge_base")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Replace with your actual Groq API key
GROQ_API_KEY = "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"
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# -----------------------
# 2. Conversation Memory
# -----------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------
# 3. Helper Functions
# -----------------------
def get_recent_chat_history(n=8):
    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    return past_chat_history[-n:] if past_chat_history else ["No past conversation history."]

def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    if results and results.get("documents"):
        docs = results.get("documents", [[]])[0]
        if isinstance(docs, list):
            return " ".join(docs)
        return docs
    return "No relevant context found."

def evaluate_response(user_query, generated_response, context):
    if isinstance(context, list):
        context = " ".join(context)
    if not context or context.strip() == "" or context == "No relevant context found.":
        return 0.0
    response_embedding = semantic_model.encode(generated_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()
    return similarity_score

def send_evaluation_to_arize(user_query, generated_response, evaluation_score):
    print(f"[DEBUG] Arize Log | Query: {user_query} | Score: {evaluation_score:.2f}")

def chunk_document(document_text, chunk_size=200, chunk_overlap=50, batch_size=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(document_text)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"chunk_index": i+j} for j in range(len(batch))]
        )
    return f"Upserted {len(chunks)} chunks to the database."

async def query_llama3_async(user_query):
    system_prompt = """
## **Goal**  
I want a chatbot that references a PDF about Nandesh Kalashetti’s background, skills, projects, and certifications. It should provide **short answers** (with emojis) for simple questions, and **detailed explanations** (referencing PDF insights) for more complex questions. The chatbot must maintain a warm, human tone and never reveal any non-human origins.

## **Return Format**  
1. **Concise Responses (Simple Queries):**  
   - Under six words.  
   - Use relevant emojis (e.g., 😊, 🚀, 👍).

2. **Detailed Explanations (In-Depth Queries):**  
   - Integrate key insights from the PDF about Nandesh Kalashetti’s educational background, technical skills, and certifications.  
   - Provide a warm, empathetic, and structured explanation.
   
3. **Tone & Style:**  
   - Human-like, empathetic, and helpful.  
   - Avoid revealing any AI or non-human nature.

## **Context Dump**  
- **Primary Source:** Contains details on Nandesh Kalashetti’s education, skills, projects, and certifications.
- **Behavioral Directives:**  
   - Present yourself as a knowledgeable human expert.
   - Never break the “no AI disclosure” rule.
---
"""
    print("[DEBUG] query_llama3_async invoked with user_query:", user_query)
    past_chat_history = get_recent_chat_history()
    retrieved_context = retrieve_context(user_query)
    combined_context = f"🗂 Past Chat: {past_chat_history}\n📖 DB Context: {retrieved_context}"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\n📝 Question: {user_query}")
    ]
    try:
        print("[DEBUG] Sending request to Groq LLaMA...")
        response = await asyncio.to_thread(chat.invoke, messages)
        print("[DEBUG] Groq response object:", response)
        if response:
            memory.save_context({"input": user_query}, {"output": response.content})
            evaluation_score = evaluate_response(user_query, response.content, retrieved_context)
            send_evaluation_to_arize(user_query, response.content, evaluation_score)
            print("[DEBUG] Response content:", response.content)
            if response.content.strip():
                return response.content
            else:
                return "⚠️ No content in the response."
        else:
            print("[DEBUG] No response object received.")
            return "⚠️ No response received from the model."
    except Exception as e:
        print("[DEBUG] Exception in query_llama3_async:", e)
        return f"⚠️ API Error: {str(e)}"

# -----------------------
# 4. PDF Extraction and Ingestion
# -----------------------
def extract_text_from_pdf(pdf_path):
    try:
        # Use a straightforward extraction without extra parameters
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print("Error extracting text from PDF:", e)
        return ""

def ingest_pdf_into_chromadb(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF file not found at: {pdf_path}")
        return
    text = extract_text_from_pdf(pdf_path)
    if text.strip():
        print(f"[DEBUG] Extracted {len(text)} characters from PDF.")
        result = chunk_document(text, chunk_size=200, chunk_overlap=50)
        print(result)
    else:
        print("⚠️ No text found in the PDF!")

# -----------------------
# 5. Streamlit UI
# -----------------------
def add_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body {
            margin: 0;
            padding: 0;
            background: #121212;
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }
        header[data-testid="stHeader"], footer {
            display: none !important;
        }
        .css-1cpxqw2 {
            width: 260px !important;
            background: #1e1e2f !important;
        }
        .css-1cpxqw2 > div {
            color: #fff !important;
        }
        .chat-header {
            padding: 1rem 1.5rem 0.5rem 1.5rem;
        }
        .chat-subheader {
            padding: 0 1.5rem 1rem 1.5rem;
            color: #aaa;
            font-size: 0.9rem;
        }
        .chat-messages {
            flex: 1;
            padding: 1rem 1.5rem;
            overflow-y: auto;
        }
        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }
        .chat-messages::-webkit-scrollbar-track {
            background: #1e1e2f;
        }
        .chat-messages::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 4px;
        }
        .message-bubble {
            padding: 1rem 1.2rem;
            margin: 0.75rem 0;
            max-width: 80%;
            border-radius: 16px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            animation: fadeIn 0.4s ease forwards;
            transition: transform 0.2s, background-color 0.2s;
        }
        .message-bubble:hover {
            transform: scale(1.01);
        }
        .user-message {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: #000;
            margin-left: auto;
            align-self: flex-end;
        }
        .bot-message {
            background: #2c2c3e;
            color: #fff;
            margin-right: auto;
            align-self: flex-start;
        }
        .message-bubble strong {
            display: block;
            margin-bottom: 0.3rem;
            font-weight: 600;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .chat-input-area {
            width: 100%;
            padding: 1rem;
            background: #1e1e2f;
            border-top: 1px solid #444;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .chat-input {
            flex: 1;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            border: none;
            background: #2c2e3e;
            color: #e0e0e0;
            font-size: 1rem;
        }
        .chat-input:focus {
            outline: none;
            box-shadow: 0 0 0 2px #00d2ff;
        }
        .send-button {
            padding: 0.8rem 1.5rem;
            background: #00d2ff;
            color: #000;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }
        .send-button:hover {
            background: #3a7bd5;
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def chatgpt_like_ui():
    st.set_page_config(page_title="PersonalChatbot-GenAI", page_icon="🤖", layout="wide")
    add_custom_css()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_query_input" not in st.session_state:
        st.session_state.user_query_input = ""
    with st.sidebar:
        st.title("New Chat")
        if st.button("Start New Chat"):
            st.session_state.chat_history = []
            memory.clear()
        st.subheader("Previous Chats")
        if st.session_state.chat_history:
            for i, ch in enumerate(st.session_state.chat_history):
                st.write(f"**{i+1}.** {ch['query']}")
        st.markdown("---")
       
        st.write("**Nandesh Kalashetti**")
        st.write("GenAi Developer And Full-stack Web-Developer")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/nandesh-kalashetti-333a78250/)")
        st.markdown("[GitHub](https://github.com/Universe7Nandu/)")
        st.markdown("[Email](mailto:nandeshkalshetti1@gmail.com)")
        st.write("Developed by Nandesh Kalashetti")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='chat-header'><h2>🤖 PersonalChatbot-GenAI</h2></div>
        <div class='chat-subheader'>Ask me about Nandesh's background, skills, projects, and more!</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
    for chat_item in st.session_state.chat_history:
        st.markdown(
            f"""
            <div class='message-bubble user-message'>
                <strong>You:</strong> {chat_item['query']}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class='message-bubble bot-message'>
                <strong>Bot:</strong> {chat_item['response']}
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("", key="user_query_input", placeholder="Ask me anything...")
        submit_button = st.form_submit_button(label="Send")
        if submit_button and user_query.strip():
            send_message(user_query)
    st.markdown("</div>", unsafe_allow_html=True)

def send_message(user_query):
    with st.spinner("Generating response..."):
        response = asyncio.run(query_llama3_async(user_query))
    st.session_state.chat_history.append({"query": user_query, "response": response})
    # Clear the input field explicitly
    st.session_state.user_query_input = ""

def main():
    pdf_path = "./resume.pdf"
    ingest_pdf_into_chromadb(pdf_path)
    chatgpt_like_ui()

if __name__ == "__main__":
    main()
