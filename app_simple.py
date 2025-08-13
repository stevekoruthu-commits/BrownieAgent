import streamlit as st
import subprocess
import socket
import os
import hashlib
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# Set Ollama models path to your D drive location
os.environ['OLLAMA_MODELS'] = r'D:\Data\OLLAMA\.ollama'

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize query cache
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}

FAISS_PATH = "faiss_index"

# --- Start Ollama server in background if not running ---
def is_ollama_running(host="localhost", port=11434):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except Exception:
        return False

if not is_ollama_running():
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.info("Starting Ollama server in the background...")
    except Exception as e:
        st.warning(f"Could not start Ollama server automatically: {e}")

st.set_page_config(page_title="Game Rules Assistant", layout="wide")

# --- Custom CSS for ChatGPT-style look ---
st.markdown("""
<style>
    .block-container { max-width: 900px !important; margin: auto; }
    section[data-testid="stSidebar"] {
        background: #202123;
        color: #ececf1;
        border-right: 1px solid #1e1e25;
    }
    .stChatMessage { max-width: 800px; margin: 0 auto; }
    .stButton>button, .stTextInput>div>input {
        border-radius: 6px !important;
    }
    .stTextInput>div>input {
        background: #1e1e25 !important; color: #ececf1 !important;
    }
    .stChatInputContainer { background: #1e1e25 !important; }
    .stChatInputContainer textarea { color: #ececf1 !important; background: #1e1e25 !important; }
    .stChatMessage { background: #343541 !important; }
    .stChatMessage.user { background: #343541 !important; }
    .stChatMessage.assistant { background: #444654 !important; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Chats")
    if st.button("New Chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.session_state.query_cache.clear()
        st.rerun()
    
    # Model selection
    st.header("Model Settings")
    available_models = ["llama3.2:1b", "llama3.1:8b", "llama3:8b", "mistral:latest"]
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "llama3.2:1b"
    
    selected_model = st.selectbox("Choose Model:", available_models, 
                                index=available_models.index(st.session_state.selected_model))
    st.session_state.selected_model = selected_model
    
    # Company Knowledge Base status
    st.header("Knowledge Base")
    st.success("✅ Company Knowledge Base")
    st.info("📄 Contains: monopoly.pdf, ticket_to_ride.pdf")
    
    # Chat history
    if st.session_state.chat_history:
        st.subheader("Recent Questions")
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {chat['question'][:30]}{'...' if len(chat['question'])>30 else ''}")

# --- Main Functions ---
def get_ollama_model():
    try:
        model_name = st.session_state.get('selected_model', 'llama3.2:1b')
        return OllamaLLM(model=model_name)
    except Exception as e:
        st.warning(f"Could not load model, using llama3.2:1b")
        return OllamaLLM(model="llama3.2:1b")

def load_main_db():
    try:
        if os.path.exists(FAISS_PATH):
            return FAISS.load_local(FAISS_PATH, get_embedding_function(), allow_dangerous_deserialization=True)
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load database: {e}")
        return None

def rebuild_faiss_from_data():
    """Rebuild FAISS index from PDFs in data/ using current embedding function."""
    data_path = "data"
    if not os.path.isdir(data_path):
        st.error("❌ 'data' folder not found. Cannot rebuild database.")
        return None

    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        st.error("❌ No PDF files found in 'data' folder. Cannot rebuild database.")
        return None

    st.info(f"🔧 Rebuilding knowledge base from {len(pdf_files)} PDF(s): {', '.join(pdf_files)}")

    # Load all PDFs
    documents = []
    for fname in pdf_files:
        loader = PyPDFLoader(os.path.join(data_path, fname))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = fname
        documents.extend(docs)

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                              separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""])
    chunks = splitter.split_documents(documents)
    st.info(f"✂️ Created {len(chunks)} chunks. Embedding...")

    # Build FAISS
    embeddings = get_embedding_function()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)
    st.success("✅ Rebuilt knowledge base successfully.")
    return db

def test_model():
    """Test if the model is working"""
    try:
        model = get_ollama_model()
        response = model.invoke("Hello")
        return True, response
    except Exception as e:
        return False, str(e)

# Load the main database
try:
    st.info("🔄 Loading Company Knowledge Base...")
    db_main = load_main_db()
    if not db_main:
        st.error("❌ Company Knowledge Base not found. Please run recreate_faiss_db.py first.")
        st.info("To fix this, run: `python recreate_faiss_db.py`")
        st.stop()
    else:
        st.success("✅ Company Knowledge Base loaded successfully!")
        
    # Test the model
    model_works, test_result = test_model()
    if not model_works:
        st.error(f"❌ Model test failed: {test_result}")
        st.info("Make sure Ollama is running and llama3.2:1b model is available.")
    else:
        st.success("✅ AI model is working!")
        
except Exception as e:
    st.error(f"❌ Error loading database: {e}")
    st.error(f"Error type: {type(e).__name__}")
    st.info("To fix this, try running: `python recreate_faiss_db.py`")
    st.stop()

# --- Main chat area ---
st.markdown("<h1 style='text-align:center; margin-bottom:0.5em;'>🎲 Game Rules Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Ask questions about Monopoly and Ticket to Ride!</p>", unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.info("🎯 **Welcome to the Game Rules Assistant!**\n\n"
           "Ask me anything about:\n"
           "• 🏠 **Monopoly** - Rules, strategies, gameplay\n"
           "• 🚂 **Ticket to Ride** - How to play, scoring, tips\n\n"
           "Just type your question below to get started!")

# Show chat history
for chat in st.session_state.chat_history:
    st.chat_message("user").write(chat["question"])
    st.chat_message("assistant").write(chat["answer"])

# Chat input
query = st.chat_input("Ask about game rules...")

if query and query.strip():
    # Show user's message
    st.chat_message("user").write(query)
    
    # Create query hash for caching
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    
    # Check cache first
    if query_hash in st.session_state.query_cache:
        cached_result = st.session_state.query_cache[query_hash]
        st.chat_message("assistant").write(cached_result)
        st.session_state.chat_history.append({"question": query, "answer": cached_result})
        st.info("✅ Used cached response")
    else:
        try:
            # Search for relevant documents
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    results = db_main.similarity_search(query, k=3)
                except AssertionError as ae:
                    # Likely embedding dimension mismatch; rebuild and retry once
                    st.warning("⚠️ Detected index/embedding mismatch. Rebuilding knowledge base...")
                    db_rebuilt = rebuild_faiss_from_data()
                    if db_rebuilt is not None:
                        db_main = db_rebuilt
                        results = db_main.similarity_search(query, k=3)
                    else:
                        raise
            
            if results:
                # Build context from search results
                context_parts = []
                sources_used = set()
                for result in results:
                    source = result.metadata.get('source', 'Game Rules')
                    sources_used.add(source)
                    content = result.page_content.strip()
                    context_parts.append(f"[{source}]\n{content}")
                
                context_text = "\n\n".join(context_parts)
                
                # Show which sources are being used
                st.info(f"📚 Found information in: {', '.join(sources_used)}")
                
                # Create prompt
                prompt = f"""Based on the following game rules and information, please provide a helpful and accurate answer.

Context:
{context_text}

Question: {query}

Please provide a clear, detailed answer based on the information provided. If the question is about game rules, explain them step by step. If it's about strategy or tips, be specific and helpful.

Answer:"""
                
                # Generate response
                model = get_ollama_model()
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Generating response..."):
                        response_text = ""
                        try:
                            for chunk in model.stream(prompt):
                                response_text += chunk
                                st.markdown(response_text + "▌")
                            st.markdown(response_text)
                        except AssertionError:
                            # Fallback to non-streaming
                            response_text = model.invoke(prompt)
                            st.markdown(response_text)
                        except Exception as model_error:
                            st.error(f"❌ Model error: {model_error}")
                            response_text = "I'm sorry, I encountered an error while generating the response. Please try again."
                            st.markdown(response_text)
                
                # Cache and store response
                st.session_state.query_cache[query_hash] = response_text
                st.session_state.chat_history.append({"question": query, "answer": response_text})
                
            else:
                error_msg = "I couldn't find relevant information about that in the game rules. Could you try rephrasing your question or ask about Monopoly or Ticket to Ride specifically?"
                st.chat_message("assistant").write(error_msg)
                st.session_state.chat_history.append({"question": query, "answer": error_msg})
                
        except Exception as e:
            error_msg = f"❌ An error occurred: {str(e)}"
            st.error(error_msg)
            st.error(f"Error type: {type(e).__name__}")
            # Still add to chat history so user can see what happened
            st.session_state.chat_history.append({"question": query, "answer": error_msg})
