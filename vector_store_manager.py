from typing import Dict, List, Tuple
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function

class VectorStoreManager:
    def __init__(self, core_path: str = "faiss_core_index", uploaded_path: str = "faiss_uploaded_index"):
        self.core_path = core_path
        self.uploaded_path = uploaded_path
        self.embedding_function = get_embedding_function()
        
        # Try to load existing stores or create new ones
        try:
            self.core_store = FAISS.load_local(core_path, self.embedding_function)
        except:
            self.core_store = FAISS.from_texts([""], self.embedding_function)
            
        try:
            self.uploaded_store = FAISS.load_local(uploaded_path, self.embedding_function)
        except:
            self.uploaded_store = FAISS.from_texts([""], self.embedding_function)
    
    def add_documents(self, documents: List[Document], is_core: bool = False) -> None:
        """Add documents to either core or uploaded vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        splits = text_splitter.split_documents(documents)
        
        if is_core:
            if len(self.core_store.docstore._dict) <= 1:  # If only contains empty string
                self.core_store = FAISS.from_documents(splits, self.embedding_function)
            else:
                self.core_store.add_documents(splits)
            self.core_store.save_local(self.core_path)
        else:
            if len(self.uploaded_store.docstore._dict) <= 1:  # If only contains empty string
                self.uploaded_store = FAISS.from_documents(splits, self.embedding_function)
            else:
                self.uploaded_store.add_documents(splits)
            self.uploaded_store.save_local(self.uploaded_path)
    
    def hybrid_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search both stores with priority to uploaded documents"""
        # Search uploaded documents first
        uploaded_results = self.uploaded_store.similarity_search_with_score(query, k=k)
        
        # If we have enough highly relevant results from uploaded docs, return those
        relevant_uploaded = [(doc, score) for doc, score in uploaded_results if score < 0.3]
        if len(relevant_uploaded) >= k//2:
            return uploaded_results[:k]
        
        # Otherwise, search core documents and combine results
        core_results = self.core_store.similarity_search_with_score(query, k=k)
        
        # Combine and sort by relevance score
        all_results = uploaded_results + core_results
        all_results.sort(key=lambda x: x[1])  # Sort by score (lower is better)
        
        return all_results[:k]
    
    def clear_uploaded_documents(self) -> None:
        """Clear all uploaded documents"""
        self.uploaded_store = FAISS.from_texts([""], self.embedding_function)
        self.uploaded_store.save_local(self.uploaded_path)
