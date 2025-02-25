import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# 1. Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db_4")
try:
    collection = chroma_client.get_collection(name="ai_knowledge_base")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(name="ai_knowledge_base")

# 2. Initialize Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Function to Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 4. Function to Chunk and Upsert into ChromaDB
def chunk_and_upsert(document_text, chunk_size=200, chunk_overlap=50, batch_size=10):
    """
    Split a document into chunks and upsert them into ChromaDB.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document_text)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = [embedding_model.embed_query(chunk) for chunk in batch]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=[f"doc_chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"chunk_index": i+j} for j in range(len(batch))]
        )
    return f"Upserted {len(chunks)} chunks to the database."

# 5. Main Function to Ingest PDF
if __name__ == "__main__":
    pdf_path = "./resume.pdf"  # <-- Make sure the PDF is in the same folder or provide the full path
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF file not found at: {pdf_path}")
    else:
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            result = chunk_and_upsert(text, chunk_size=200, chunk_overlap=50)
            print(result)
        else:
            print("⚠️ No text found in the PDF!")
