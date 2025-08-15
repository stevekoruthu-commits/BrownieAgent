"""
Background Service Manager for RAG App
Keeps Ollama models warm and pre-loads everything for instant startup
"""
import os
import time
import threading
import json
from datetime import datetime
import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

class RAGServiceManager:
    def __init__(self):
        self.status = {
            "ollama_ready": False,
            "llm_model_ready": False,
            "embedding_model_ready": False,
            "chroma_ready": False,
            "company_docs_ready": False,
            "last_heartbeat": None,
            "startup_time": None
        }
        self.models = {
            "llm": "llama3.2:1b",
            "embedding": "nomic-embed-text"
        }
        self.collections = {}
        self.embedding_function = None
        self.chroma_client = None
        
    def start_services(self):
        """Start all background services and warm up models"""
        print("🚀 Starting RAG services...")
        start_time = time.time()
        
        # Start Ollama and warm up models
        self._warm_ollama()
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Pre-load company documents
        self._preload_company_docs()
        
        # Start heartbeat
        self._start_heartbeat()
        
        self.status["startup_time"] = time.time() - start_time
        print(f"✅ All services ready in {self.status['startup_time']:.2f} seconds")
        self._save_status()
        
    def _warm_ollama(self):
        """Warm up Ollama models"""
        try:
            print("🔥 Warming up Ollama models...")
            
            # Test Ollama connection
            models = ollama.list()
            self.status["ollama_ready"] = True
            print("✅ Ollama connected")
            
            # Warm up LLM model with a quick query
            response = ollama.generate(
                model=self.models["llm"],
                prompt="Hello",
                options={"num_predict": 1}
            )
            self.status["llm_model_ready"] = True
            print(f"✅ LLM model {self.models['llm']} warmed up")
            
            # Warm up embedding model
            embedding_response = ollama.embeddings(
                model=self.models["embedding"],
                prompt="test"
            )
            self.status["embedding_model_ready"] = True
            print(f"✅ Embedding model {self.models['embedding']} warmed up")
            
        except Exception as e:
            print(f"❌ Error warming Ollama: {e}")
            
    def _init_chromadb(self):
        """Initialize ChromaDB with pre-configured collections"""
        try:
            print("📊 Initializing ChromaDB...")
            
            # Create embedding function
            self.embedding_function = OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name=self.models["embedding"]
            )
            
            # Initialize persistent client
            self.chroma_client = chromadb.PersistentClient(path="./chroma_vector_db")
            
            # Pre-create collections
            for collection_name in ["company_docs", "temp_docs"]:
                collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                self.collections[collection_name] = collection
                
            self.status["chroma_ready"] = True
            print("✅ ChromaDB initialized with collections")
            
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            
    def _preload_company_docs(self):
        """Pre-load company documents if they exist"""
        try:
            print("📚 Checking company documents...")
            
            data_folder = "./data"
            if os.path.exists(data_folder):
                pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
                if pdf_files:
                    collection = self.collections.get("company_docs")
                    if collection:
                        docs = collection.get()
                        doc_count = len(docs["documents"]) if docs["documents"] else 0
                        print(f"✅ Company docs ready: {doc_count} chunks from {len(pdf_files)} PDFs")
                        self.status["company_docs_ready"] = True
                    else:
                        print("⚠️ ChromaDB collection not ready")
                else:
                    print("ℹ️ No PDF files in data/ folder")
            else:
                print("ℹ️ No data/ folder found")
                
        except Exception as e:
            print(f"❌ Error checking company docs: {e}")
            
    def _start_heartbeat(self):
        """Start background heartbeat to keep models warm"""
        def heartbeat():
            while True:
                try:
                    # Quick health check every 5 minutes
                    time.sleep(300)
                    
                    # Ping Ollama to keep models in memory
                    ollama.generate(
                        model=self.models["llm"],
                        prompt="ping",
                        options={"num_predict": 1}
                    )
                    
                    self.status["last_heartbeat"] = datetime.now().isoformat()
                    self._save_status()
                    
                except Exception as e:
                    print(f"⚠️ Heartbeat error: {e}")
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        print("💓 Heartbeat started (keeps models warm)")
        
    def _save_status(self):
        """Save status to file for other processes to read"""
        try:
            with open("service_status.json", "w") as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save status: {e}")
            
    def get_collection(self, name):
        """Get a pre-initialized collection"""
        return self.collections.get(name)
        
    def is_ready(self):
        """Check if all services are ready"""
        return (self.status["ollama_ready"] and 
                self.status["chroma_ready"] and
                self.embedding_function is not None)

def main():
    """Run as background service"""
    service = RAGServiceManager()
    service.start_services()
    
    # Keep running to maintain warm models
    try:
        print("🔄 Service running... Press Ctrl+C to stop")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 Service stopped")

if __name__ == "__main__":
    main()
