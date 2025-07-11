# Mastering-AI-Agents-Workshop

1. pip install requests beautifulsoup4 streamlit langchain_ollama

2. ollama pull mistral
ollama run mistral


3. streamlit run ai_scraper_app.py


| Import Statement                                            | Purpose                                                                       |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `import requests`                                           | Sends HTTP requests to fetch website content (used for web scraping).         |
| `from bs4 import BeautifulSoup`                             | Parses and extracts text from HTML (used to scrape `<p>` tags from webpages). |
| `import streamlit as st`                                    | Builds an interactive web UI for your AI app (inputs, display, etc.).         |
| `import faiss`                                              | Handles fast similarity search (used for storing and retrieving embeddings).  |
| `import numpy as np`                                        | Supports numerical operations, especially on vectors (used with FAISS).       |
| `from langchain_ollama import OllamaLLM`                    | Loads and runs local AI models via Ollama (like Mistral, LLaMA3).             |
| `from langchain_huggingface import HuggingFaceEmbeddings`   | Converts text into vector embeddings using HuggingFace models.                |
| `from langchain_community.vectorstores import FAISS`        | Connects LangChain to FAISS for vector search and retrieval.                  |
| `from langchain.text_splitter import CharacterTextSplitter` | Splits large text into smaller chunks for better embedding and storage.       |
| `from langchain.schema import Document`                     | Defines document structure in LangChain (used internally by chains/stores).   |
