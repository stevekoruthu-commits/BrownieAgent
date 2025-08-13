from functools import lru_cache
from langchain_ollama import OllamaEmbeddings
import os

# Set the OLLAMA_MODELS environment variable
os.environ["OLLAMA_MODELS"] = r"D:\Data\OLLAMA\.ollama"

@lru_cache(maxsize=1)
def get_embedding_function():
    """Get an embedding function using nomic-embed-text for semantic search"""
    return OllamaEmbeddings(
        model="nomic-embed-text",  # Using proper embedding model
        base_url="http://localhost:11434",
        # Add timeout settings
        request_timeout=120,  # 2 minutes timeout
        show_progress=False,
    )

# Alternative: Use faster embedding models
@lru_cache(maxsize=1)
def get_fast_embedding_function():
    """Use faster embedding model"""
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
        request_timeout=60,  # 1 minute timeout
        show_progress=False,
    )
