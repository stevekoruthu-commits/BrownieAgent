"""
Law Firm Optimized RAG App
Fast-loading with detailed legal document analysis
Optimized for comprehensive case research and document review
"""
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
def init_law_firm_services():
    """Initialize services for law firm use"""
    try:
        embedding_function = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="nomic-embed-text"
        )
        
        chroma_client = chromadb.PersistentClient(path="./chroma_vector_db")
        
        return chroma_client, embedding_function
    except Exception as e:
        st.error(f"Service initialization error: {e}")
        return None, None

def get_law_collection(collection_name: str):
    """Get collection optimized for legal document analysis"""
    if st.session_state.chroma_client is None:
        st.session_state.chroma_client, st.session_state.embedding_function = init_law_firm_services()
    
    if st.session_state.chroma_client:
        return st.session_state.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=st.session_state.embedding_function
        )
    return None

@st.cache_data(ttl=600)  # 10 minute cache for law firm stability
def legal_document_query(prompt: str, search_scope: str = "both"):
    """Comprehensive legal document query with detailed results"""
    try:
        if search_scope == "both":
            # Query both collections with more results for thorough analysis
            company_collection = get_law_collection("company_docs")
            temp_collection = get_law_collection("temp_docs")
            
            results_company = company_collection.query(query_texts=[prompt], n_results=8)
            results_temp = temp_collection.query(query_texts=[prompt], n_results=4)
            
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
            collection = get_law_collection(collection_name)
            return collection.query(query_texts=[prompt], n_results=10)
            
    except Exception as e:
        st.error(f"Query error: {e}")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

def legal_llm_analysis(context: str, prompt: str):
    """Comprehensive LLM analysis for legal documents"""
    try:
        full_prompt = f"""You are a legal document analysis assistant. Based on the following legal document excerpts, please provide a comprehensive and detailed analysis.

LEGAL DOCUMENT CONTENT:
{context}

QUESTION: {prompt}

Please provide a thorough legal analysis based on the document content. Include:
1. Key legal points and findings
2. Relevant case details and parties involved
3. Important dates, procedures, and legal citations
4. Any procedural or substantive legal issues
5. Context and background information

This analysis is for legitimate legal research purposes using publicly available court documents."""

        response = ollama.generate(
            model="llama3.2:1b",
            prompt=full_prompt,
            stream=True,
            options={
                "temperature": 0.2,  # Lower for more consistent legal analysis
                "top_p": 0.8,
                "num_predict": 800   # More tokens for detailed legal analysis
            }
        )
        
        for chunk in response:
            if chunk.get('response'):
                yield chunk['response']
                
    except Exception as e:
        yield f"Error in legal analysis: {e}"

def process_legal_document(uploaded_file: UploadedFile) -> list[Document]:
    """Process legal documents with optimized chunking"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.flush()
        
        try:
            loader = PyMuPDFLoader(temp_file.name)
            docs = loader.load()
            
            if not docs:
                st.error("Could not load PDF content")
                return []
            
            # Optimized chunking for legal documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,     # Larger chunks for legal context
                chunk_overlap=300,   # More overlap for legal continuity
                length_function=len,
                is_separator_regex=False,
            )
            
            return text_splitter.split_documents(docs)
            
        finally:
            os.unlink(temp_file.name)

def add_legal_document(all_splits: list[Document], file_name: str):
    """Add legal document to collection with proper metadata"""
    collection = get_law_collection("temp_docs")
    if not collection:
        st.error("Collection not available")
        return
        
    try:
        documents = [doc.page_content for doc in all_splits]
        metadatas = [{"source": file_name, "chunk_id": i, "doc_type": "legal"} for i, doc in enumerate(all_splits)]
        ids = [f"{file_name}_{i}" for i in range(len(all_splits))]
        
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        
        st.success(f"✅ Added {len(all_splits)} chunks from {file_name}")
        
    except Exception as e:
        st.error(f"Error adding legal document: {e}")

# Main Law Firm App UI
def main():
    st.set_page_config(
        page_title="⚖️ Law Firm Document Analysis", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with law firm branding
    st.title("⚖️ Legal Document Analysis System")
    st.markdown("*Comprehensive case research and document review platform*")
    
    # Check service status
    services_ready = check_service_status()
    
    if not services_ready:
        st.warning("🔄 Background services not detected. Running in standard mode...")
        st.info("💡 For optimal performance, run: `python service_manager.py` in another terminal")
    else:
        st.success("⚡ Enhanced performance mode - using pre-warmed services!")
    
    # Initialize services
    if not st.session_state.service_ready:
        with st.spinner("Initializing legal document analysis system..."):
            st.session_state.chroma_client, st.session_state.embedding_function = init_law_firm_services()
            st.session_state.service_ready = True
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Case files (company docs)
        st.subheader("📚 Case Database")
        company_collection = get_law_collection("company_docs")
        if company_collection:
            company_docs = company_collection.get()
            company_count = len(company_docs["documents"]) if company_docs["documents"] else 0
            st.info(f"📊 {company_count} case documents indexed")
        
        st.divider()
        
        # Upload new documents
        st.subheader("📄 Upload New Documents")
        uploaded_file = st.file_uploader(
            "Upload legal documents (PDF)", 
            type=["pdf"],
            help="Upload case files, court documents, contracts, etc."
        )
        
        if uploaded_file:
            if f"legal_processed_{uploaded_file.name}" not in st.session_state:
                with st.spinner("📑 Processing legal document..."):
                    splits = process_legal_document(uploaded_file)
                    if splits:
                        add_legal_document(splits, uploaded_file.name)
                        st.session_state[f"legal_processed_{uploaded_file.name}"] = True
                        st.rerun()
        
        # Current session documents
        temp_collection = get_law_collection("temp_docs")
        if temp_collection:
            temp_docs = temp_collection.get()
            temp_count = len(temp_docs["documents"]) if temp_docs["documents"] else 0
            st.info(f"📋 {temp_count} session documents")
            
            if temp_count > 0 and st.button("🗑️ Clear Session", help="Clear temporarily uploaded documents"):
                temp_collection.delete(ids=temp_docs["ids"])
                st.success("Session documents cleared!")
                st.rerun()
    
    # Main analysis area
    st.header("🔍 Legal Document Analysis")
    
    # Search scope selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_scope = st.selectbox(
            "📋 Search Scope:",
            ["both", "company", "temp"],
            format_func=lambda x: {
                "both": "🔍 All Documents (Case DB + Session)",
                "company": "📚 Case Database Only", 
                "temp": "📄 Session Documents Only"
            }[x]
        )
    
    with col2:
        st.metric("Total Documents", 
                 f"{company_count + temp_count if 'company_count' in locals() and 'temp_count' in locals() else 'Loading...'}")
    
    # Query input
    st.subheader("💬 Legal Research Query")
    
    # Enhanced input area
    col_input, col_button = st.columns([10, 1])
    
    with col_input:
        prompt = st.text_area(
            "**Enter your legal research question:**",
            placeholder="e.g., 'Analyze the case details for Dr. Rajesh Kumar Gupta' or 'What are the key legal issues in this matter?' or 'Summarize the procedural history'",
            height=100
        )
    
    with col_button:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask = st.button("🔍", help="Analyze documents")

    # Analysis results
    if prompt and prompt.strip() and (ask or prompt != st.session_state.get("last_legal_query", "")):
        st.session_state["last_legal_query"] = prompt
        
        with st.spinner("🔍 Analyzing legal documents..."):
            results = legal_document_query(prompt, search_scope)
        
        if not results.get("documents")[0]:
            st.warning("⚠️ No relevant documents found. Please upload legal documents or check your query.")
        else:
            doc_count = len(results.get('documents')[0])
            st.success(f"📋 Found {doc_count} relevant document sections")
            
            # Show document sources
            with st.expander("📄 Document Sources", expanded=False):
                metadatas = results.get("metadatas")[0]
                sources = list(set([meta.get("source", "Unknown") for meta in metadatas if meta]))
                st.write("**Sources analyzed:**")
                for source in sources:
                    st.write(f"• {source}")
            
            # Combine context with relevance scoring
            context_pieces = results.get("documents")[0][:6]  # Top 6 most relevant
            context = "\n\n--- Document Section ---\n".join(context_pieces)
            
            # Legal analysis
            st.subheader("⚖️ Legal Analysis")
            
            analysis_container = st.empty()
            full_analysis = ""
            
            for chunk in legal_llm_analysis(context, prompt):
                full_analysis += chunk
                analysis_container.markdown(full_analysis + "▊")
            
            analysis_container.markdown(full_analysis)
            
            # Additional analysis options
            st.subheader("🔧 Additional Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Case Summary"):
                    summary_prompt = f"Provide a comprehensive case summary based on: {prompt}"
                    st.session_state["last_legal_query"] = summary_prompt
                    st.rerun()
            
            with col2:
                if st.button("📅 Timeline Analysis"):
                    timeline_prompt = f"Create a chronological timeline of events for: {prompt}"
                    st.session_state["last_legal_query"] = timeline_prompt
                    st.rerun()
            
            with col3:
                if st.button("🎯 Key Issues"):
                    issues_prompt = f"Identify and analyze key legal issues in: {prompt}"
                    st.session_state["last_legal_query"] = issues_prompt
                    st.rerun()

if __name__ == "__main__":
    main()
