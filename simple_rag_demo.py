# Simple RAG Application - Core Concept
# This shows the basic RAG workflow in just a few steps

import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Setup (One-time)
@st.cache_resource
def setup_rag():
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    llm = OllamaLLM(model="llama3.2:1b")
    return embeddings, llm

# Step 2: Process Document (When uploaded)
def process_document(file_path, embeddings):
    # Load document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Create vector database
    db = FAISS.from_documents(chunks, embeddings)
    return db

# Step 3: Answer Questions (Main workflow)
def answer_question(question, db, llm):
    # Search for relevant content
    relevant_docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Create prompt
    prompt = f"""Based on this context: {context}
    
    Question: {question}
    Answer:"""
    
    # Generate answer
    response = llm.invoke(prompt)
    return response

# That's it! The entire RAG process:
# 1. Upload document → Process → Store in vector DB
# 2. Ask question → Search → Get context → Generate answer

st.title("Simple RAG Demo")
st.write("See? RAG is just: Upload → Search → Answer!")

embeddings, llm = setup_rag()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    # Save and process file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    db = process_document("temp.pdf", embeddings)
    st.success("Document processed!")
    
    question = st.text_input("Ask a question:")
    if question:
        answer = answer_question(question, db, llm)
        st.write(f"**Answer:** {answer}")

st.write("""
## RAG in 3 Simple Steps:
1. **Process**: Split document → Create embeddings → Store in vector DB
2. **Search**: User asks question → Find relevant chunks
3. **Generate**: Send context + question to LLM → Get answer

That's literally it! Everything else is just UI polish and optimization.
""")
