#!/usr/bin/env python3
"""
Recreate FAISS database with the correct llama3.2:1b model using existing PDF files
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from get_embedding_function import get_embedding_function

DATA_PATH = "data"
FAISS_PATH = "faiss_index"

def main():
    print("🔄 Recreating FAISS database with correct model...")
    
    # Remove old database if it exists
    if os.path.exists(FAISS_PATH):
        import shutil
        shutil.rmtree(FAISS_PATH)
        print("🗑️ Removed old database")
    
    # Load all PDF files from data directory
    documents = []
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in data/ directory")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files: {pdf_files}")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_PATH, pdf_file)
        print(f"📖 Loading {pdf_file}...")
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Add source metadata
        for doc in docs:
            doc.metadata['source'] = pdf_file
        
        documents.extend(docs)
    
    print(f"✅ Loaded {len(documents)} pages total")
    
    # Split documents into chunks
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Get embedding function (using llama3.2:1b)
    print("🧮 Getting embedding function...")
    embeddings = get_embedding_function()
    
    # Create FAISS database
    print("🔗 Creating FAISS database...")
    db = FAISS.from_documents(chunks, embeddings)
    
    # Save database
    print("💾 Saving database...")
    db.save_local(FAISS_PATH)
    
    print(f"🎉 Successfully created FAISS database with {len(chunks)} chunks!")
    print(f"📍 Saved to: {FAISS_PATH}")
    
    # Test the database
    print("\n🧪 Testing database...")
    test_query = "What is this document about?"
    results = db.similarity_search(test_query, k=3)
    print(f"✅ Test search returned {len(results)} results")
    for i, result in enumerate(results, 1):
        source = result.metadata.get('source', 'Unknown')
        content_preview = result.page_content[:100].replace('\n', ' ')
        print(f"   {i}. [{source}] {content_preview}...")

if __name__ == "__main__":
    main()
