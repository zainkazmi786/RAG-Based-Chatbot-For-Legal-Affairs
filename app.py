from flask import Flask, render_template, request, jsonify
from retrieval.retriever import HybridRetriever
from generation.llm_chain import JudgmentGenerator
from langchain.schema import Document
from pathlib import Path
import traceback
import time
import json

app = Flask(__name__)

# Initialize with base data
retriever = HybridRetriever(
    "data/processed/processed_cases.json",
    new_data_path=["data/processed/FamilyCourtsAct1964_processed.json",
                  "data/processed/processed_cpc_laws.json",
                  "data/processed/processed_ordinance_data.json"]
)
generator = JudgmentGenerator("generation/prompts/legal_judgment.txt")

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return responses."""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        # Show typing indicator on frontend by returning immediately
        # Get retrieved data based on user input
        retrieved_data = retriever.retrieve(user_input)
        
        all_docs = []
        all_docs.extend(retrieved_data["vector"])
        
        # Combine vector and keyword results
        for doc_dict in retrieved_data["keyword"]:
            all_docs.append(Document(
                page_content=doc_dict["text"],
                metadata=doc_dict["metadata"]
            ))
        
        # Remove duplicate documents
        seen = set()
        unique_docs = []
        for doc in all_docs:
            identifier = f"{doc.page_content[:50]}-{str(doc.metadata)}"
            if identifier not in seen:
                seen.add(identifier)
                unique_docs.append(doc)
        
        # Process the query with LLM
        response = generator.generate(user_input, unique_docs)
        
        # Return structured response with sources
        return jsonify({
            "response": response,
            "sources_count": len(unique_docs),
            "timestamp": time.time()
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-source', methods=['POST'])
def add_source():
    """Add a new data source to the retriever."""
    try:
        data = request.json
        source_path = data.get('source_path', '').strip()
        
        if not source_path:
            return jsonify({"error": "No source path provided"}), 400
            
        if not Path(source_path).exists():
            return jsonify({"error": f"File not found at {source_path}"}), 404
            
        if retriever._add_new_data(source_path):
            retriever.vector_db = retriever._initialize_vector_store()
            return jsonify({
                "success": True,
                "message": f"Updated with {len(retriever.documents)} total docs",
                "document_count": len(retriever.documents)
            })
        else:
            return jsonify({
                "success": False,
                "message": "No new documents added (duplicates detected)"
            })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/document-count')
def document_count():
    """Return the current document count."""
    return jsonify({
        "count": len(retriever.documents)
    })

if __name__ == "__main__":
    app.run(debug=True)