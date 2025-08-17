#!/usr/bin/env python3
"""
Preprocessing Script for Law Firm RAG System
============================================

This script preprocesses all PDF documents in your data folders and stores them
in ChromaDB collections. Run this script once before starting your app for
instant startup times.

Usage:
    python preprocessing_script.py

After running this script, your Streamlit app will start in 1-2 seconds!
"""

import os
import glob
import time
import chromadb
import concurrent.futures
import multiprocessing
from pathlib import Path
from functools import partial
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

# Performance optimization settings
BATCH_SIZE = 1000  # Increased batch size for faster processing
MAX_WORKERS = min(multiprocessing.cpu_count() - 1, 4)  # Leave 1 CPU free
CHUNK_SIZE = 1200  # Larger chunks for better context
CHUNK_OVERLAP = 150  # Reduced overlap for speed

def optimize_memory():
    """Configure system for better memory management"""
    import gc
    gc.collect()  # Force garbage collection
    gc.set_threshold(700, 10, 5)  # More aggressive collection

class ProgressTracker:
    """Tracks processing progress and estimates completion time"""
    def __init__(self, total_files):
        self.total_files = total_files
        self.processed = 0
        self.start_time = time.time()
    
    def update(self, files_done=1):
        self.processed += files_done
        elapsed = time.time() - self.start_time
        rate = self.processed / elapsed if elapsed > 0 else 0
        eta = (self.total_files - self.processed) / rate if rate > 0 else 0
        
        print(f"Progress: {self.processed}/{self.total_files} files")
        print(f"ETA: {eta:.1f} seconds remaining")

def setup_chromadb():
    """Initialize ChromaDB client and embedding function"""
    try:
        # Initialize embedding function
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest",
        )
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_vector_db")
        
        return chroma_client, ollama_ef
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {e}")
        print("🔧 Make sure Ollama is running: ollama serve")
        print("📦 And the model is available: ollama pull nomic-embed-text")
        return None, None

def estimate_processing_time():
    """Estimate how long preprocessing will take"""
    import glob
    
    general_files = glob.glob("data/general folder/*.pdf") if os.path.exists("data/general folder") else []
    company_files1 = glob.glob("data/*.pdf") if os.path.exists("data") else []
    company_files2 = glob.glob("data/companydocs/*.pdf") if os.path.exists("data/companydocs") else []
    
    total_files = len(general_files) + len(company_files1) + len(company_files2)
    
    if total_files == 0:
        return "No PDF files found", 0
    
    # Rough estimate: 30 seconds per file (varies by size)
    estimated_minutes = (total_files * 30) / 60
    
    return f"~{estimated_minutes:.1f} minutes for {total_files} files", total_files

def process_pdf_to_collection(pdf_file, collection, source_type):
    """Process a single PDF file and add to collection"""
    try:
        print(f"   📄 Processing: {os.path.basename(pdf_file)}")
        
        # Load PDF
        loader = PyMuPDFLoader(pdf_file)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        # Prepare data for ChromaDB
        documents_list = []
        metadatas_list = []
        ids_list = []
        
        file_name = os.path.basename(pdf_file)
        
        for i, split in enumerate(splits):
            documents_list.append(split.page_content)
            metadatas_list.append({
                **split.metadata,
                "source_type": source_type,
                "file_name": file_name
            })
            ids_list.append(f"{source_type}_{file_name}_{i}")
        
        # Add to collection in optimized batches
        batch_size = 500  # Increased from 100 for faster processing
        total_chunks = len(documents_list)
        
        print(f"      📊 Processing {total_chunks} chunks in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch_docs = documents_list[i:end_idx]
            batch_metas = metadatas_list[i:end_idx]
            batch_ids = ids_list[i:end_idx]
            
            print(f"      🔄 Batch {i//batch_size + 1}: Processing chunks {i+1}-{end_idx}")
            
            # Retry logic for embedding failures
            max_retries = 3
            for retry in range(max_retries):
                try:
                    collection.upsert(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids
                    )
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"      ⚠️  Retry {retry + 1}/3 for batch {i//batch_size + 1}: {e}")
                        import time
                        time.sleep(2)  # Wait before retry
                    else:
                        print(f"      ❌ Failed batch {i//batch_size + 1} after {max_retries} retries")
                        raise
        
        print(f"   ✅ Added {total_chunks} chunks from {file_name}")
        return total_chunks
        
    except Exception as e:
        print(f"   ❌ Error processing {pdf_file}: {e}")
        return 0

def preprocess_general_documents(chroma_client, ollama_ef):
    """Preprocess general legal knowledge documents"""
    print("\n📚 PREPROCESSING GENERAL LEGAL KNOWLEDGE")
    print("=" * 50)
    
    # Create/get general collection
    general_collection = chroma_client.get_or_create_collection(
        name="general_docs", 
        embedding_function=ollama_ef
    )
    
    # Clear existing data
    try:
        general_collection.delete()
        general_collection = chroma_client.create_collection(
            name="general_docs", 
            embedding_function=ollama_ef
        )
    except:
        pass
    
    # Process general folder
    general_folder = "data/general folder"
    if os.path.exists(general_folder):
        general_files = glob.glob(os.path.join(general_folder, "*.pdf"))
        print(f"📂 Found {len(general_files)} general legal documents")
        
        total_chunks = 0
        for pdf_file in general_files:
            chunks = process_pdf_to_collection(pdf_file, general_collection, "general")
            total_chunks += chunks
        
        print(f"✅ General Knowledge: {total_chunks} chunks processed")
        return total_chunks
    else:
        print(f"⚠️  General folder not found: {general_folder}")
        return 0

def preprocess_company_documents(chroma_client, ollama_ef):
    """Preprocess company documents"""
    print("\n🏢 PREPROCESSING COMPANY DOCUMENTS")
    print("=" * 50)
    
    # Create/get company collection
    company_collection = chroma_client.get_or_create_collection(
        name="company_docs", 
        embedding_function=ollama_ef
    )
    
    # Clear existing data
    try:
        company_collection.delete()
        company_collection = chroma_client.create_collection(
            name="company_docs", 
            embedding_function=ollama_ef
        )
    except:
        pass
    
    total_chunks = 0
    
    # Process data/ folder (root level PDFs)
    data_folder = "data"
    if os.path.exists(data_folder):
        root_files = glob.glob(os.path.join(data_folder, "*.pdf"))
        if root_files:
            print(f"📂 Found {len(root_files)} PDFs in data/ folder")
            for pdf_file in root_files:
                chunks = process_pdf_to_collection(pdf_file, company_collection, "company")
                total_chunks += chunks
    
    # Process data/companydocs/ folder
    company_folder = "data/companydocs"
    if os.path.exists(company_folder):
        company_files = glob.glob(os.path.join(company_folder, "*.pdf"))
        if company_files:
            print(f"📂 Found {len(company_files)} PDFs in companydocs/ folder")
            for pdf_file in company_files:
                chunks = process_pdf_to_collection(pdf_file, company_collection, "company")
                total_chunks += chunks
    
    if total_chunks == 0:
        print("⚠️  No company documents found in data/ or data/companydocs/")
    else:
        print(f"✅ Company Documents: {total_chunks} chunks processed")
    
    return total_chunks

def setup_temp_collection(chroma_client, ollama_ef):
    """Setup empty temp collection for uploaded documents"""
    print("\n📑 SETTING UP TEMPORARY DOCUMENTS COLLECTION")
    print("=" * 50)
    
    try:
        temp_collection = chroma_client.get_or_create_collection(
            name="temp_docs", 
            embedding_function=ollama_ef
        )
        print("✅ Temporary collection ready for uploads")
        return True
    except Exception as e:
        print(f"❌ Error setting up temp collection: {e}")
        return False

def main():
    """Main preprocessing function"""
    print("=" * 60)
    print("🚀 LAW FIRM RAG SYSTEM - DOCUMENT PREPROCESSING")
    print("=" * 60)
    
    # Show time estimate
    time_estimate, file_count = estimate_processing_time()
    print(f"📊 Found files to process: {file_count}")
    print(f"⏱️  Estimated processing time: {time_estimate}")
    print("💡 Processing speed depends on:")
    print("   - Number and size of PDF files")
    print("   - Ollama embedding speed")
    print("   - System performance")
    print()
    
    if file_count == 0:
        print("⚠️  No PDF files found in data/ folders!")
        print("📁 Add PDFs to:")
        print("   - data/general folder/  (for general legal knowledge)")
        print("   - data/companydocs/     (for company documents)")
        print("   - data/                 (for company documents)")
        return
    
    print("This will preprocess all your documents for instant app startup!")
    print()
    
    # Setup ChromaDB
    chroma_client, ollama_ef = setup_chromadb()
    if not chroma_client:
        return
    
    # Create data folders if they don't exist
    os.makedirs("data/general folder", exist_ok=True)
    os.makedirs("data/companydocs", exist_ok=True)
    
    # Preprocess all document types
    general_chunks = preprocess_general_documents(chroma_client, ollama_ef)
    company_chunks = preprocess_company_documents(chroma_client, ollama_ef)
    temp_setup = setup_temp_collection(chroma_client, ollama_ef)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📚 General Legal Knowledge: {general_chunks} chunks")
    print(f"🏢 Company Documents: {company_chunks} chunks")
    print(f"📑 Temporary Collection: {'Ready' if temp_setup else 'Error'}")
    print()
    print("✅ Your Streamlit app will now start in 1-2 seconds!")
    print("🚀 Run: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
