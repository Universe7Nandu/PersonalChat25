import sys
import warnings
import os
import asyncio
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx  
from groq import Groq
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# Suppress Streamlit warnings (this works in most cases)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# ------------------------------------------------
# 1. Initialize ChromaDB, Embeddings, and Chat Model
# ------------------------------------------------
# Use a persistent path for ChromaDB; adjust the path as needed
chroma_client = chromadb.PersistentClient(path="./chroma_db_4")

# Use get_collection if available, else create the collection
try:
    collection = chroma_client.get_collection(name="ai_knowledge_base")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(name="ai_knowledge_base")

# Initialize the Hugging Face embedding model and SentenceTransformer model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Set your Groq API key (provided key)
GROQ_API_KEY = "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"
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------
def get_recent_chat_history(n=8):
    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])
    return past_chat_history[-n:] if past_chat_history else ["No past conversation history."]

def get_memory_usage():
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    return len(chat_history)

def retrieve_context(query, top_k=3):
    """Retrieve relevant documents from ChromaDB using embeddings."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    if results and results.get("documents"):
        # If documents is a list, join them into a string
        docs = results.get("documents", [[]])[0]
        if isinstance(docs, list):
            return " ".join(docs)
        return docs
    return "No relevant context found."

def evaluate_response(user_query, generated_response, context):
    """Evaluate the response by comparing it with the retrieved context."""
    # If context is a list, join it into a string
    if isinstance(context, list):
        context = " ".join(context)
    if not context or context.strip() == "" or context == "No relevant context found.":
        return 0.0
    response_embedding = semantic_model.encode(generated_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()
    return similarity_score

def send_evaluation_to_arize(user_query, generated_response, evaluation_score):
    # Placeholder for Arize integration; here we just print the score.
    print(f"Arize Log | Query: {user_query} | Score: {evaluation_score:.2f}")

def chunk_document(document_text, chunk_size=200, chunk_overlap=50, batch_size=10):
    """
    Split a document into chunks and upsert them into ChromaDB.
    Useful for ingesting large documents (e.g., PDF resumes).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(document_text)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"chunk_index": i+j} for j in range(len(batch))]
        )
    
    return f"Upserted {len(chunks)} chunks to the database."

# ------------------------------------------------
# 3. Asynchronous Core Chat Function
# ------------------------------------------------
async def query_llama3_async(user_query):
    system_prompt = """
## **Goal**  
I want a chatbot that references a PDF about Nandesh Kalashetti’s background, skills, projects, and certifications. It should provide **short answers** (with emojis) for simple questions, and **detailed explanations** (referencing PDF insights) for more complex questions. The chatbot must maintain a warm, human tone and never reveal any non-human origins.

## **Return Format**  
1. **Concise Responses (Simple Queries):**  
   - Under six words.  
   - Use relevant emojis (e.g., 😊, 🚀, 👍) to convey friendliness and clarity.  

2. **Detailed Explanations (In-Depth Queries):**  
   - Integrate key insights from the PDF about Nandesh Kalashetti’s educational background, technical skills, and certifications.  
   - Provide a warm, empathetic, and structured explanation.  
   - Always maintain precision and clarity, referencing relevant data from the PDF.  

3. **Tone & Style:**  
   - Human-like, empathetic, and helpful.  
   - Avoid revealing any AI or non-human nature.  
   - Adapt answer length based on the query’s complexity.

## **Warnings**  
- Never disclose any AI-related details or mention you’re a system.  
- Remain consistent with Nandesh Kalashetti’s persona—an expert full-stack web developer with advanced problem-solving skills.  
- Ensure all references to the PDF are correct and accurately represent Nandesh’s background.

## **Context Dump**  
- **Primary Source (Uploaded PDF):** Contains comprehensive details on Nandesh Kalashetti’s education, technical skills, projects, certifications, and overall professional expertise.  
- **Key Capabilities:**  
  - Dynamically adapt response length (short vs. long) based on the user’s query.  
  - Insert emojis to add warmth and clarity.  
  - Provide real-world applicability in every solution.  
- **Behavioral Directives:**  
  - Present yourself as a knowledgeable human expert.  
  - Never break the “no AI disclosure” rule.  
  - Prioritize user understanding and real-world context.  

---

**Usage Example:**  
1. **Short Query:** “What are Nandesh’s top skills?”  
   - **Short Answer** (≤6 words, with emojis)  
2. **Complex Query:** “Tell me more about his advanced projects and how they integrate with cloud platforms.”  
   - **Detailed Explanation** referencing PDF data (projects, certifications, advanced solutions), with structured insights and an empathetic tone.

"""
    past_chat_history = get_recent_chat_history()
    retrieved_context = retrieve_context(user_query)
    combined_context = f"🗂 Past Chat: {past_chat_history}\n📖 DB Context: {retrieved_context}"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\n📝 Question: {user_query}")
    ]
    try:
        # Run the synchronous chat.invoke method asynchronously using asyncio.to_thread
        response = await asyncio.to_thread(chat.invoke, messages)
        if response:
            memory.save_context({"input": user_query}, {"output": response.content})
            evaluation_score = evaluate_response(user_query, response.content, retrieved_context)
            send_evaluation_to_arize(user_query, response.content, evaluation_score)
            print(f"💾 Memory Usage: {get_memory_usage()} past interactions")
            print(f"Evaluation Score (Semantic Similarity): {evaluation_score:.2f}")
            return response.content
        return "⚠️ No response received."
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# ------------------------------------------------
# 4. Streamlit Web UI
# ------------------------------------------------
def add_custom_css():
    st.markdown(
        """
   <style>
/* Import a professional font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
  margin: 0;
  padding: 0;
  background: linear-gradient(135deg,rgb(227, 230, 235) 0%, #243B55 100%);
  font-family: 'Inter', sans-serif;
}

/* Central chat container */
.chat-container {
  width: 100%;
  max-width: 800px;
  margin: 3rem auto;
  background: #fff; /* A crisp, clean container background */
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  overflow: hidden;
  position: relative;
}

/* Optional header for your chatbot */
.chat-header {
  text-align: center;
  background: #f5f6fa;
  padding: 1.5rem 1rem;
  border-bottom: 1px solid #eaeaea;
  font-size: 1.75rem;
  font-weight: 700;
  color: #333;
}

/* Main area for messages */
.chat-messages {
  padding: 1.5rem;
  background: #fafbfc;
  min-height: 300px; /* Ensures some height even with few messages */
}

/* Shared bubble styles */
.message-bubble {
  padding: 1rem 1.5rem;
  margin: 1rem 0;
  max-width: 70%;
  border-radius: 16px;
  box-shadow: 0 5px 15px rgba(223, 33, 33, 0.1);
  animation: fadeIn 0.4s ease forwards;
  transition: transform 0.2s ease-in-out;
}

/* Hover effect to add a little interactivity */
.message-bubble:hover {
  transform: scale(1.02);
}

/* Distinguish user messages with a sleek gradient */
.user-message {
  align-self: flex-end;
  background: linear-gradient(135deg, #3A7BD5, #00D2FF);
  color: black;
  text-align: right;
}

/* Bot messages in a softer, neutral tone */
.bot-message {
  align-self: flex-start;
  background: #ECEFF1;
  color: black;
  border: 1px solid #dde1e4;
}

/* Fade-in animation for message bubbles */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Sticky or pinned area at the bottom for input (if used that way) */
.chat-input-area {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: #f5f6fa;
  border-top: 1px solid #eaeaea;
}

/* Text input with a more professional look */
.chat-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 30px;
  font-size: 1rem;
  background: #fff;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.chat-input:focus {
  outline: none;
  border-color: #3A7BD5;
  box-shadow: 0 0 0 3px rgba(58,123,213,0.2);
}

/* Send button with a refined gradient and subtle hover effect */
.send-button {
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 30px;
  background: linear-gradient(135deg, #3A7BD5, #00D2FF);
  color: #fff;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.send-button:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
</style>

        """,
        unsafe_allow_html=True
    )

def streamlit_chat():
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
    add_custom_css()
    st.title("🤖 PersonalChatbot-GenAI")
    st.write("An advanced chatbot powered by RAG, prompt engineering, and optimized inference.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("Ask a question:")
        submit_button = st.form_submit_button(label="Send ✈️")

    if submit_button and user_query:
        with st.spinner("Generating response..."):
            response = asyncio.run(query_llama3_async(user_query))
        st.session_state.chat_history.append({"query": user_query, "response": response})

        # Limit chat history for performance
        MAX_CHAT_HISTORY = 100
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history.pop(0)

    for chat_item in st.session_state.chat_history:
        st.markdown(f"<div class='message-bubble user-message'><strong>You:</strong> {chat_item['query']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='message-bubble bot-message'><strong>Bot:</strong> {chat_item['response']}</div>", unsafe_allow_html=True)

# ------------------------------------------------
# 5. Main Execution
# ------------------------------------------------
if __name__ == "__main__":
    streamlit_chat()