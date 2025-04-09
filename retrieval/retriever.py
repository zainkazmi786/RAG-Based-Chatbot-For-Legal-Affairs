from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import json
import numpy as np
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer issues

class HybridRetriever:
    def __init__(self, processed_data_path):
        # Set up paths
        self.data_path = Path(processed_data_path)
        self.vector_store_dir = self.data_path.parent
        self.vector_store_name = self.data_path.stem
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # print(model.encode(["test"])[0][:5]) 
        
        # Load data
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.documents = [d["text"] for d in data]
        self.metadata = [d["metadata"] for d in data]
        
        # Initialize components
        self.vector_db = self._initialize_vector_store()
        self.bm25 = BM25Okapi([doc.split() for doc in self.documents])
    
    def _initialize_vector_store(self):
        if (self.vector_store_dir / f"{self.vector_store_name}.faiss").exists():
            print("Loading existing vector store...")
            return FAISS.load_local(
                folder_path=str(self.vector_store_dir),
                embeddings=self._embed_query,  # Pass our embedding function
                index_name=self.vector_store_name,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vector store...")
            embeddings = self.embedding_model.encode(self.documents)
            vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(self.documents, embeddings)),
                embedding=self._embed_query,
                metadatas=self.metadata
            )
            vector_db.save_local(
                folder_path=str(self.vector_store_dir),
                index_name=self.vector_store_name
            )
            return vector_db
    
    def _embed_query(self, text):
        """Custom embedding function for FAISS queries"""
        return self.embedding_model.encode([text])[0]
    
    def retrieve(self, query, top_k=3):
        # Vector similarity search
        vector_results = self.vector_db.similarity_search(query, k=top_k)
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:top_k]
        bm25_results = [{"text": self.documents[i], "metadata": self.metadata[i]} for i in bm25_indices]
        
        return {
            "vector": vector_results,
            "keyword": bm25_results
        }