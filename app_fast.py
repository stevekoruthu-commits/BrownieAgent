"""
Fast-loading RAG App
Uses pre-warmed services for instant@st.cache_data(ttl=300)
def fast_query_collection(prompt: str, search_scope: str = "both"):
    """Fast cached query optimized for legal document analysis"""
    try:
        if search_scope == "both":
            # Query both collections - optimized for law firm needs
            company_collection = get_fast_collection("company_docs")
            temp_collection = get_fast_collection("temp_docs")
            
            results_company = company_collection.query(query_texts=[prompt], n_results=7)
            results_temp = temp_collection.query(query_texts=[prompt], n_results=3)""
import os
import json
import tempfile
import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Global service objects (reused across requests)
if "service_ready" not in st.session_state:
    st.session_state.service_ready = False
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = None

def check_service_status():
    """Check if background services are running"""
    try:
        with open("service_status.json", "r") as f:
            status = json.load(f)
        return status.get("ollama_ready", False) and status.get("chroma_ready", False)
    except:
        return False

@st.cache_resource
def init_fast_services():
    """Initialize services quickly using pre-warmed components"""
    try:
        # Use pre-warmed embedding function
        embedding_function = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        
        # Use existing ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_vector_db")
        
        return chroma_client, embedding_function
    except Exception as e:
        st.error(f"Service initialization error: {e}")
        return None, None

def get_fast_collection(collection_name: str):
    """Get collection from pre-initialized ChromaDB"""
    if st.session_state.chroma_client is None:
        st.session_state.chroma_client, st.session_state.embedding_function = init_fast_services()
    
    if st.session_state.chroma_client:
        return st.session_state.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=st.session_state.embedding_function
        )
    return None

@st.cache_data(ttl=300)
def fast_query_collection(prompt: str, search_scope: str = "both"):
    """Fast cached query using pre-warmed models"""
    try:
        if search_scope == "both":
            # Query both collections
            company_collection = get_fast_collection("company_docs")
            temp_collection = get_fast_collection("temp_docs")
            
            results_company = company_collection.query(query_texts=[prompt], n_results=3)
            results_temp = temp_collection.query(query_texts=[prompt], n_results=3)
            
            # Combine results
            all_documents = results_company["documents"][0] + results_temp["documents"][0]
            all_metadatas = results_company["metadatas"][0] + results_temp["metadatas"][0]
            all_distances = results_company["distances"][0] + results_temp["distances"][0]
            
            return {
                "documents": [all_documents],
                "metadatas": [all_metadatas],
                "distances": [all_distances]
            }
        else:
            collection_name = "company_docs" if search_scope == "company" else "temp_docs"
            collection = get_fast_collection(collection_name)
            return collection.query(query_texts=[prompt], n_results=5)
            
    except Exception as e:
        st.error(f"Query error: {e}")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

def fast_llm_call(context: str, prompt: str):
    """Fast LLM call using pre-warmed model"""
    try:
        # Use the pre-warmed model with improved prompt
        full_prompt = f"""You are analyzing documents from a knowledge base. Based on the following document excerpts, please answer the question accurately and helpfully.

DOCUMENT CONTENT:
{context}

QUESTION: {prompt}

Please analyze the documents and provide a detailed answer based on the information provided. This is from a legitimate document collection."""

        response = ollama.generate(
            model="llama3.2:1b",
            prompt=full_prompt,
            stream=True,
            options={
                "temperature": 0.3,  # Lower = faster, more focused
                "top_p": 0.8,        # Reduced for speed
                "top_k": 20,         # Limit vocabulary for speed
                "num_predict": 300,  # Shorter responses = faster
                "num_ctx": 2048,     # Smaller context window
                "repeat_penalty": 1.1
            }
        )
        
        for chunk in response:
            if chunk.get('response'):
                yield chunk['response']
                
    except Exception as e:
        yield f"Error: {e}"

def process_document_fast(uploaded_file: UploadedFile) -> list[Document]:
    """Fast document processing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        
        try:
            loader = PyMuPDFLoader(temp_file.name)
            docs = loader.load()
            
            if not docs:
                st.error("Could not load PDF content")
                return []
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            
            return text_splitter.split_documents(docs)
            
        finally:
            os.unlink(temp_file.name)

def add_to_collection_fast(all_splits: list[Document], file_name: str):
    """Fast document addition to collection"""
    collection = get_fast_collection("temp_docs")
    if not collection:
        st.error("Collection not available")
        return
        
    try:
        documents = [doc.page_content for doc in all_splits]
        metadatas = [{"source": file_name, "chunk_id": i} for i, doc in enumerate(all_splits)]
        ids = [f"{file_name}_{i}" for i in range(len(all_splits))]
        
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        
        st.success(f"Added {len(all_splits)} chunks from {file_name}")
        
    except Exception as e:
        st.error(f"Error adding documents: {e}")

# Main App UI
def main():
    st.set_page_config(page_title="⚡ Fast RAG App", layout="wide")
    
    # Check service status
    services_ready = check_service_status()
    
    if not services_ready:
        st.warning("🔄 Background services not detected. Starting in standard mode...")
        st.info("💡 To enable fast startup, run: `python service_manager.py` in another terminal")
    else:
        st.success("⚡ Fast mode enabled - using pre-warmed services!")
    
    # Initialize services on first run
    if not st.session_state.service_ready:
        with st.spinner("Initializing fast services..."):
            st.session_state.chroma_client, st.session_state.embedding_function = init_fast_services()
            st.session_state.service_ready = True
    
    # UI Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📚 Document Sources")
        
        # Company docs status
        st.subheader("🏢 Company Knowledge")
        company_collection = get_fast_collection("company_docs")
        if company_collection:
            company_docs = company_collection.get()
            company_count = len(company_docs["documents"]) if company_docs["documents"] else 0
            st.info(f"📊 {company_count} chunks ready")
        
        st.divider()
        
        # Fast upload
        st.subheader("📑 Quick Upload")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        if uploaded_file:
            if f"fast_processed_{uploaded_file.name}" not in st.session_state:
                with st.spinner("⚡ Fast processing..."):
                    splits = process_document_fast(uploaded_file)
                    if splits:
                        add_to_collection_fast(splits, uploaded_file.name)
                        st.session_state[f"fast_processed_{uploaded_file.name}"] = True
                        st.rerun()
        
        # Temp docs status
        temp_collection = get_fast_collection("temp_docs")
        if temp_collection:
            temp_docs = temp_collection.get()
            temp_count = len(temp_docs["documents"]) if temp_docs["documents"] else 0
            st.info(f"📄 {temp_count} temp chunks")
            
            if temp_count > 0 and st.button("🗑️ Clear"):
                temp_collection.delete(ids=temp_docs["ids"])
                st.success("Cleared!")
                st.rerun()
    
    with col2:
        st.header("⚡ Fast RAG Chat")
        
        # Search scope
        search_scope = st.selectbox(
            "Search in:",
            ["both", "company", "temp"],
            format_func=lambda x: {
                "both": "🔍 All Documents",
                "company": "🏢 Company Docs", 
                "temp": "📑 Uploaded Docs"
            }[x]
        )
        
        # Fast input
        col_input, col_button = st.columns([10, 1])
        
        with col_input:
            prompt = st.text_input(
                "**Ask your question:**",
                placeholder="Type and press Enter for instant results..."
            )
        
        with col_button:
            ask = st.button("⚡", help="Fast search")

        # Fast processing
        if prompt and prompt.strip() and (ask or prompt != st.session_state.get("last_fast_prompt", "")):
            st.session_state["last_fast_prompt"] = prompt
            
            with st.spinner("⚡ Searching..."):
                results = fast_query_collection(prompt, search_scope)
            
            if not results.get("documents")[0]:
                st.warning("No documents found. Upload some PDFs first!")
            else:
                doc_count = len(results.get('documents')[0])
                st.success(f"⚡ Found {doc_count} relevant chunks instantly!")
                
                # Combine context
                context = "\n\n".join(results.get("documents")[0][:5])  # Top 5 results
                
                # Fast LLM response
                st.write("**Answer:**")
                response_container = st.empty()
                
                full_response = ""
                for chunk in fast_llm_call(context, prompt):
                    full_response += chunk
                    response_container.write(full_response + "▊")
                
                response_container.write(full_response)

if __name__ == "__main__":
    main()
