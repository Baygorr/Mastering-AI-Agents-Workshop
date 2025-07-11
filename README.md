# Mastering-AI-Agents-Workshop

1. pip install streamlit faiss-cpu numpy PyPDF2 langchain langchain-community langchain-huggingface langchain-ollama sentence-transformers


2. ollama pull mistral
ollama run mistral


3. streamlit run ai_document_reader.py


| Import                  | Used For                                              |
| ----------------------- | ----------------------------------------------------- |
| `streamlit`             | Web UI (upload files, display results)                |
| `faiss`                 | Vector similarity search                              |
| `numpy`                 | Converting vectors to float32 arrays                  |
| `PyPDF2`                | Extracting text from PDF                              |
| `OllamaLLM`             | Generating responses/summaries from local LLM         |
| `HuggingFaceEmbeddings` | Converting text chunks to vector embeddings           |
| `FAISS` (LangChain)     | Optional wrapper to use FAISS with LangChain          |
| `CharacterTextSplitter` | Splitting long text into chunks                       |
| `Document`              | Data structure to represent documents (optional here) |
