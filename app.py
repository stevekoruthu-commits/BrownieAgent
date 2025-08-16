import os
import tempfile

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Add caching for faster repeated queries
@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_query_collection(prompt: str, search_scope: str = "both"):
    """Cached version of query_collection for faster repeated queries"""
    return query_collection(prompt, search_scope=search_scope)

def analyze_question_type(question: str) -> dict:
    """Analyzes the question to determine appropriate response style.
    
    Categorizes questions based on keywords and structure to determine
    if the user wants a short, direct answer or a detailed explanation.
    
    Args:
        question: The user's question string
    
    Returns:
        dict: Contains response_type, max_tokens, and instructions
    """
    question_lower = question.lower().strip()
    
    # Keywords that indicate short, direct answers
    short_indicators = [
        "what is", "who is", "when is", "where is", "how much", "how many",
        "define", "meaning of", "name", "list", "yes or no", "true or false",
        "which", "what does", "give me", "tell me briefly"
    ]
    
    # Keywords that indicate detailed explanations
    detailed_indicators = [
        "explain", "describe", "how does", "why", "elaborate", "detail",
        "process", "procedure", "steps", "analysis", "compare", "difference",
        "overview", "summary", "comprehensive", "in depth", "thoroughly"
    ]
    
    # Keywords that indicate medium-length responses
    medium_indicators = [
        "benefits", "advantages", "disadvantages", "features", "types",
        "examples", "uses", "applications", "importance", "significance"
    ]
    
    # Check for question length and complexity
    word_count = len(question.split())
    has_multiple_parts = any(sep in question for sep in [" and ", " also ", " additionally ", "?", ";"])
    
    # Determine response type
    if any(indicator in question_lower for indicator in short_indicators) and word_count <= 5:
        response_type = "short"
    elif any(indicator in question_lower for indicator in detailed_indicators):
        response_type = "detailed"
    elif any(indicator in question_lower for indicator in medium_indicators):
        response_type = "medium"
    elif word_count <= 3 and not has_multiple_parts:
        response_type = "short"
    elif word_count > 12 or has_multiple_parts:
        response_type = "detailed"
    else:
        response_type = "detailed"  # Default to detailed instead of medium
    
    # Configure response parameters based on type - INCREASED TOKEN LIMITS FOR COMPLETE ANSWERS
    response_configs = {
        "short": {
            "max_tokens": 600,
            "temperature": 0.1,
            "instruction": "Provide a clear, focused answer with key details. Be informative but concise."
        },
        "medium": {
            "max_tokens": 1000,
            "temperature": 0.2,
            "instruction": "Provide a well-structured answer with good detail and examples where relevant."
        },
        "detailed": {
            "max_tokens": 1500,
            "temperature": 0.3,
            "instruction": "Provide a comprehensive explanation with detailed information and context."
        }
    }
    
    config = response_configs[response_type]
    config["response_type"] = response_type
    
    return config

def get_adaptive_system_prompt(response_type: str) -> str:
    """Returns an adaptive system prompt based on the question type."""
    
    base_prompt = """You are an AI assistant that analyzes documents from a knowledge base and provides accurate, helpful answers. You have access to legitimate document content including legal documents, reports, and other informational materials. Always base your responses on the provided context."""
    
    type_specific_prompts = {
        "short": """
Instructions:
- Provide clear, direct answers based on the document content
- Focus on the key information that answers the question
- Use 1-2 paragraphs with essential details
- Reference the document content appropriately
- Be concise but informative
""",
        "medium": """
Instructions:
- Provide comprehensive answers based on the document content
- Use multiple paragraphs to explain thoroughly
- Include relevant background information from the documents
- Explain concepts clearly with supporting details from the context
- Organize information logically
""",
        "detailed": """
Instructions:
- Provide thorough, detailed explanations based on the document content
- Use multiple paragraphs with comprehensive structure
- Include extensive details, context, and background from the documents
- Explain concepts thoroughly from multiple angles when the documents allow
- Use bullet points, numbered lists where helpful
- Provide comprehensive analysis of the available information
"""
    }
    
    return base_prompt + type_specific_prompts.get(response_type, type_specific_prompts["medium"])

system_prompt = """
You are an AI assistant that provides clear, direct answers based on the given context.

Instructions:
- Answer the question directly using the provided information
- Use clear, simple language  
- Organize with bullet points or short paragraphs
- Be comprehensive but concise
- Don't mention document sources
- Focus on what's most relevant to the user's question

Provide a well-structured response that directly addresses the user's question.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()  # Close the file before using it
    
    splits = []
    try:
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        
        # Check if documents were loaded
        if not docs:
            st.error("Could not load content from the PDF file. The file might be corrupted or empty.")
            return []
        
        # Check if there's any text content
        total_content = "".join([doc.page_content for doc in docs])
        if not total_content.strip():
            st.error("The PDF appears to contain no readable text content.")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,   # Optimized for speed
            chunk_overlap=100, # Reduced overlap for faster processing
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        splits = text_splitter.split_documents(docs)
        
        # Filter out empty splits
        splits = [split for split in splits if split.page_content.strip()]
        
    except Exception as e:
        st.error(f"Error processing PDF file: {str(e)}")
        return []
    finally:
        # Clean up temp file with error handling
        try:
            os.unlink(temp_file.name)
        except (OSError, PermissionError):
            # If we can't delete it immediately, it will be cleaned up later
            pass
    
    return splits


def get_vector_collection(collection_name: str = "rag_app") -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Args:
        collection_name: Name of the collection (e.g., "company_docs" or "temp_docs")

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./chroma_vector_db")
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    # Check if we have any splits to process
    if not all_splits:
        st.error("No content could be extracted from the uploaded file. Please check if it's a valid PDF with readable text.")
        return
    
    collection = get_vector_collection("temp_docs")  # Use temp collection for uploads
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        # Check if the split has content
        if split.page_content and split.page_content.strip():
            documents.append(split.page_content)
            metadatas.append(split.metadata)
            ids.append(f"{file_name}_{idx}")

    # Check if we have any valid documents to add
    if not documents:
        st.error("No readable text content found in the uploaded file.")
        return

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success(f"Successfully added {len(documents)} document chunks to the vector store!")


def load_company_documents():
    """Loads all PDF documents from the data/ folder into the company collection.
    
    Processes all PDF files in the data/ folder and stores them in a separate
    company documents collection for permanent access.
    
    Returns:
        int: Number of documents processed
    """
    import glob
    
    data_folder = "data"
    if not os.path.exists(data_folder):
        return 0
        
    pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not pdf_files:
        return 0
    
    collection = get_vector_collection("company_docs")
    
    # Clear existing company docs
    existing_docs = collection.get()
    if existing_docs["documents"]:
        collection.delete(ids=existing_docs["ids"])
    
    total_chunks = 0
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,   # Optimized for speed
                chunk_overlap=100, # Reduced overlap for faster processing
                separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            )
            splits = text_splitter.split_documents(docs)
            
            # Process in batches to avoid timeout
            file_name = os.path.basename(pdf_file).replace(".pdf", "").translate(
                str.maketrans({"-": "", ".": "", " ": "_"})
            )
            
            batch_size = 20  # Process 20 chunks at a time
            for i in range(0, len(splits), batch_size):
                batch_splits = splits[i:i + batch_size]
                documents, metadatas, ids = [], [], []
                
                for idx, split in enumerate(batch_splits):
                    global_idx = i + idx
                    documents.append(split.page_content)
                    metadatas.append({**split.metadata, "source_type": "company"})
                    ids.append(f"company_{file_name}_{global_idx}")
                
                # Upsert batch with retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            st.error(f"Failed to process batch after {max_retries} retries: {str(e)}")
                            continue
                        st.warning(f"Batch processing failed, retrying... ({retry + 1}/{max_retries})")
                        import time
                        time.sleep(2)  # Wait 2 seconds before retry
                
            total_chunks += len(splits)
            
        except Exception as e:
            st.error(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return total_chunks


def query_collection(prompt: str, n_results: int = 5, search_scope: str = "both"):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 15.
        search_scope: Which collections to search ("company", "temp", or "both").

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    results = {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    
    if search_scope in ["company", "both"]:
        company_collection = get_vector_collection("company_docs")
        try:
            company_results = company_collection.query(query_texts=[prompt], n_results=n_results)
            results["documents"][0].extend(company_results["documents"][0])
            results["distances"][0].extend(company_results["distances"][0])
            results["metadatas"][0].extend(company_results["metadatas"][0])
        except Exception:
            pass  # Company collection might be empty
    
    if search_scope in ["temp", "both"]:
        temp_collection = get_vector_collection("temp_docs")
        try:
            temp_results = temp_collection.query(query_texts=[prompt], n_results=n_results)
            results["documents"][0].extend(temp_results["documents"][0])
            results["distances"][0].extend(temp_results["distances"][0])
            results["metadatas"][0].extend(temp_results["metadatas"][0])
        except Exception:
            pass  # Temp collection might be empty
    
    # Sort by distance (lower is better)
    if results["documents"][0]:
        combined = list(zip(results["documents"][0], results["distances"][0], results["metadatas"][0]))
        combined.sort(key=lambda x: x[1])
        
        # Limit to n_results
        combined = combined[:n_results]
        
        results["documents"][0] = [x[0] for x in combined]
        results["distances"][0] = [x[1] for x in combined]
        results["metadatas"][0] = [x[2] for x in combined]
    
    return results


def call_llm(context: str, prompt: str):
    """Adaptive LLM call that adjusts response style based on question analysis.

    Uses Ollama to stream responses with adaptive formatting that provides
    short, medium, or detailed responses based on the question type.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    # Analyze the question to determine response style
    question_analysis = analyze_question_type(prompt)
    response_type = question_analysis["response_type"]
    max_tokens = question_analysis["max_tokens"]
    temperature = question_analysis["temperature"]
    instruction = question_analysis["instruction"]
    
    # Get adaptive system prompt
    adaptive_system_prompt = get_adaptive_system_prompt(response_type)
    
    # Create user message with specific instructions
    user_message = f"""DOCUMENT CONTENT:
{context}

QUESTION: {prompt}

{instruction}

Please analyze the document content above and provide a comprehensive answer to the question."""
    
    response = ollama.chat(
        model="llama3.2:1b",
        stream=True,
        options={
            "temperature": temperature,
            "top_p": 0.8,  # Reduced for faster, more focused responses
            "num_predict": max_tokens,
            "top_k": 40,   # Limit vocabulary for speed
        },
        messages=[
            {
                "role": "system",
                "content": adaptive_system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_documents(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    """Enhanced document ranking and context formatting for unified responses.
    
    Takes the top documents from ChromaDB results and formats them in a way
    that encourages the LLM to provide unified, synthesized answers rather than
    fragmented responses from multiple sources.
    
    Args:
        documents: List of document strings already ranked by ChromaDB similarity.
        prompt: User's question for context-aware formatting.
    
    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Well-formatted context for unified responses
            - relevant_text_ids (list[int]): List of indices for the top ranked documents
    """
    relevant_text = ""
    relevant_text_ids = []
    
    # Take top 4 documents for better speed
    top_k = min(3, len(documents))
    
    if top_k > 0:
        # Create a unified context without document separators
        relevant_text = "KNOWLEDGE BASE INFORMATION:\n\n"
        
        for idx in range(top_k):
            if idx < len(documents) and documents[idx].strip():
                # Remove redundant information and clean up text
                clean_text = documents[idx].strip()
                if clean_text:
                    relevant_text += f"{clean_text}\n\n"
                    relevant_text_ids.append(idx)
        
        # Add instruction for synthesis
        relevant_text += f"\nUSER QUESTION: {prompt}\n"
        relevant_text += "\nINSTRUCTION: Use all the above information to provide one comprehensive, well-structured answer to the user's question."
    
    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    st.set_page_config(page_title="RAG Question Answer", layout="wide")
    
    # Initialize company documents on startup with error handling
    if "company_docs_loaded" not in st.session_state:
        try:
            with st.spinner("Loading company documents..."):
                chunks = load_company_documents()
                st.session_state.company_docs_loaded = True
                if chunks > 0:
                    st.success(f"Loaded {chunks} chunks from company documents in data/ folder")
                else:
                    st.info("No PDF files found in data/ folder")
        except Exception as e:
            st.error(f"Error loading company documents: {str(e)}")
            st.warning("Continuing without company documents. You can still upload temporary files.")
            st.session_state.company_docs_loaded = True  # Mark as loaded to avoid retrying
    
    # Two column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📚 Document Sources")
        
        # Company Documents Status
        st.subheader("🏢 Company Knowledge Base")
        company_collection = get_vector_collection("company_docs")
        company_docs = company_collection.get()
        company_count = len(company_docs["documents"]) if company_docs["documents"] else 0
        st.info(f"Company documents: {company_count} chunks loaded")
        
        if st.button("🔄 Reload Company Docs"):
            chunks = load_company_documents()
            st.success(f"Reloaded {chunks} chunks from data/ folder")
            st.rerun()
        
        st.divider()
        
        # Temporary Upload Area
        st.subheader("📑 Temporary Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF for one-time queries", 
            type=["pdf"], 
            accept_multiple_files=False
        )
        
        # Auto-process when file is uploaded
        if uploaded_file:
            # Check if this file hasn't been processed yet
            if f"processed_{uploaded_file.name}" not in st.session_state:
                with st.spinner("Processing uploaded document..."):
                    normalize_uploaded_file_name = uploaded_file.name.translate(
                        str.maketrans({"-": "", ".": "", " ": "_"})
                    )
                    all_splits = process_document(uploaded_file)
                    add_to_vector_collection(all_splits, normalize_uploaded_file_name)
                    st.session_state[f"processed_{uploaded_file.name}"] = True
                    st.rerun()  # Refresh to show updated counts
        
        # Show temp documents status
        temp_collection = get_vector_collection("temp_docs")
        temp_docs = temp_collection.get()
        temp_count = len(temp_docs["documents"]) if temp_docs["documents"] else 0
        st.info(f"Temporary documents: {temp_count} chunks")
        
        if temp_count > 0 and st.button("🗑 Clear Temp Docs"):
            temp_collection.delete(ids=temp_docs["ids"])
            st.success("Cleared temporary documents")
            st.rerun()
    
    with col2:
        # Question and Answer Area
        st.header("🗣 RAG Question Answer")
        
        # Search scope selection
        search_scope = st.selectbox(
            "Search in:",
            ["both", "company", "temp"],
            format_func=lambda x: {
                "both": "🔍 Both Company & Temporary Docs",
                "company": "🏢 Company Documents Only", 
                "temp": "📑 Temporary Documents Only"
            }[x]
        )
        
        # Create input row with text area and button
        col_input, col_button = st.columns([10, 1])
        
        with col_input:
            prompt = st.text_input(
                "*Ask a question related to your documents:*",
                key="question_input",
                placeholder="Type your question and press Enter or click the arrow..."
            )
        
        with col_button:
            ask = st.button("➤", help="Ask question", key="ask_button")

        # Process question if button clicked or Enter pressed (text_input submits on Enter)
        if prompt and prompt.strip() and (ask or prompt != st.session_state.get("last_prompt", "")):
            st.session_state["last_prompt"] = prompt
            
            # Analyze question type for adaptive response
            question_analysis = analyze_question_type(prompt)
            response_type = question_analysis["response_type"]
            
            # Visual indicators for response type
            response_type_colors = {
                "short": "🟢",
                "medium": "🟡", 
                "detailed": "🔵"
            }
            
            response_type_descriptions = {
                "short": "Quick Answer",
                "medium": "Balanced Response",
                "detailed": "Detailed Explanation"
            }
            
            # Show response type indicator
            st.info(f"{response_type_colors[response_type]} *{response_type_descriptions[response_type]}* mode detected for your question")
            
            with st.spinner("Searching documents..."):
                # Use cached query for faster repeated questions
                results = cached_query_collection(prompt, search_scope)
            
            if not results.get("documents")[0]:
                if search_scope == "temp":
                    st.warning("No documents found in temporary collection. Please upload and process a document first.")
                elif search_scope == "company": 
                    st.warning("No documents found in company collection. Please add PDFs to the data/ folder and reload.")
                else:
                    st.warning("No documents found in either collection. Please upload documents or check the data/ folder.")
            else:
                st.success(f"Found {len(results.get('documents')[0])} relevant chunks from {search_scope} collection(s)")
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_documents(context, prompt)
                response = call_llm(context=relevant_text, prompt=prompt)
                st.write_stream(response)