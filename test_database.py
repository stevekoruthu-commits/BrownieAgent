#!/usr/bin/env python3
"""
Test script to verify the FAISS database is working
"""

import os
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import FAISS

FAISS_PATH = "faiss_index"

def test_database():
    print("🧪 Testing FAISS database...")
    
    if not os.path.exists(FAISS_PATH):
        print("❌ FAISS database not found!")
        return False
    
    try:
        # Load the database
        embeddings = get_embedding_function()
        db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ Database loaded successfully")
        
        # Test queries
        test_queries = [
            "What is Monopoly?",
            "How do you play Ticket to Ride?",
            "What are the rules?",
            "How many players can play?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = db.similarity_search(query, k=3)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    source = result.metadata.get('source', 'Unknown')
                    content_preview = result.page_content[:150].replace('\n', ' ')
                    print(f"   {i}. [{source}] {content_preview}...")
            else:
                print("❌ No results found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database: {e}")
        return False

if __name__ == "__main__":
    success = test_database()
    if success:
        print("\n🎉 Database test completed successfully!")
        print("✅ Your RAG system should now work with the Company Knowledge Base")
    else:
        print("\n💥 Database test failed!")
        print("❌ You may need to recreate the database")
