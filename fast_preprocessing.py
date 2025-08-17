#!/usr/bin/env python3
"""
Fast Parallel Preprocessing Script
==================================

This version uses parallel processing to speed up document preprocessing
by processing multiple files simultaneously.
"""

import os
import glob
import chromadb
import concurrent.futures
from multiprocessing import cpu_count
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

def setup_chromadb():
    """Initialize ChromaDB client and embedding function"""
    try:
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text:latest",
        )
        chroma_client = chromadb.PersistentClient(path="./chroma_vector_db")
        return chroma_client, ollama_ef
    except Exception as e:
        print(f"❌ Error setting up ChromaDB: {e}")
        return None, None

def process_single_pdf(args):
    """Process a single PDF file (for parallel execution)"""
    pdf_file, source_type, ollama_ef = args
    
    try:
        print(f"🔄 Processing: {os.path.basename(pdf_file)}")
        
        # Load and split PDF
        loader = PyMuPDFLoader(pdf_file)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        # Prepare data
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
        
        return {
            "documents": documents_list,
            "metadatas": metadatas_list,
            "ids": ids_list,
            "file": pdf_file,
            "source_type": source_type,
            "count": len(documents_list)
        }
        
    except Exception as e:
        print(f"❌ Error processing {pdf_file}: {e}")
        return None

def fast_parallel_preprocessing():
    """Main parallel preprocessing function"""
    print("🚀 PARALLEL PROCESSING MODE")
    print("=" * 50)
    
    # Setup
    chroma_client, ollama_ef = setup_chromadb()
    if not chroma_client:
        return
    
    # Collect all files
    general_files = glob.glob("data/general folder/*.pdf") if os.path.exists("data/general folder") else []
    company_files1 = glob.glob("data/*.pdf") if os.path.exists("data") else []
    company_files2 = glob.glob("data/companydocs/*.pdf") if os.path.exists("data/companydocs") else []
    
    # Prepare arguments for parallel processing
    tasks = []
    for pdf in general_files:
        tasks.append((pdf, "general", ollama_ef))
    for pdf in company_files1 + company_files2:
        tasks.append((pdf, "company", ollama_ef))
    
    if not tasks:
        print("⚠️ No PDF files found!")
        return
    
    print(f"📊 Processing {len(tasks)} files in parallel...")
    print(f"🔧 Using {min(4, cpu_count())} CPU cores")
    
    # Process files in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, cpu_count())) as executor:
        future_to_file = {executor.submit(process_single_pdf, task): task[0] for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:
                results.append(result)
                print(f"✅ {os.path.basename(result['file'])}: {result['count']} chunks")
    
    # Batch insert into collections
    print("\n📚 Storing in ChromaDB collections...")
    
    # Separate by collection type
    general_data = [r for r in results if r['source_type'] == 'general']
    company_data = [r for r in results if r['source_type'] == 'company']
    
    # Store general docs
    if general_data:
        general_collection = chroma_client.get_or_create_collection(
            name="general_docs", 
            embedding_function=ollama_ef
        )
        for data in general_data:
            general_collection.upsert(
                documents=data['documents'],
                metadatas=data['metadatas'],
                ids=data['ids']
            )
        print(f"✅ General: {sum(d['count'] for d in general_data)} chunks")
    
    # Store company docs
    if company_data:
        company_collection = chroma_client.get_or_create_collection(
            name="company_docs", 
            embedding_function=ollama_ef
        )
        for data in company_data:
            company_collection.upsert(
                documents=data['documents'],
                metadatas=data['metadatas'],
                ids=data['ids']
            )
        print(f"✅ Company: {sum(d['count'] for d in company_data)} chunks")
    
    print("\n🎉 PARALLEL PREPROCESSING COMPLETE!")

if __name__ == "__main__":
    fast_parallel_preprocessing()
