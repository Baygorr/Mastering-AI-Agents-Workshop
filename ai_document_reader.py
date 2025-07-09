import streamlit as st
import faiss
import numpy as np
import PyPDF2
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Constants
VECTOR_DIM = 384  # for all-MiniLM-L6-v2
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3  # number of chunks to retrieve

# Initialize components
llm = OllamaLLM(model="mistral")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(VECTOR_DIM)
text_chunks = []
summary_text = ""

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

# Store text in FAISS
def store_in_faiss(text):
    global text_chunks, index
    st.info("🔍 Processing and storing text...")

    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(text)
    vectors = embeddings.embed_documents(chunks)

    # Ensure correct dtype
    faiss_vectors = np.array(vectors, dtype=np.float32)
    index.add(faiss_vectors)
    text_chunks.extend(chunks)
    return f"✅ Stored {len(chunks)} chunks."

# Generate summary
def generate_summary(text):
    global summary_text
    st.info("🧠 Generating summary...")
    summary_text = llm.invoke(f"Summarize this document:\n\n{text[:3000]}")
    return summary_text

# Question answering
def retrieve_and_answer(query):
    if index.ntotal == 0:
        return "⚠️ No documents stored yet."

    query_vec = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_vec, k=min(TOP_K, len(text_chunks)))

    context = ""
    for i in indices[0]:
        if i < len(text_chunks):
            context += text_chunks[i] + "\n"

    if not context.strip():
        return "🤖 No relevant information found."

    return llm.invoke(f"Use this context to answer the question:\n\n{context}\n\nQuestion: {query}")

# Enable summary download
def download_summary():
    if summary_text:
        st.download_button(
            label="📥 Download Summary",
            data=summary_text,
            file_name="AI_Summary.txt",
            mime="text/plain"
        )

# Streamlit UI
st.title("📄 AI Document Reader & Q&A Bot")
st.write("Upload a PDF, receive a summary, and ask document-based questions.")

uploaded_file = st.file_uploader("📂 Upload a PDF Document", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if text:
        message = store_in_faiss(text)
        st.success(message)

        # Generate and display summary
        summary = generate_summary(text)
        st.subheader("📝 AI Summary")
        st.write(summary)
        download_summary()

query = st.text_input("❓ Ask a question based on the uploaded document:")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("🤖 Answer")
    st.write(answer)
