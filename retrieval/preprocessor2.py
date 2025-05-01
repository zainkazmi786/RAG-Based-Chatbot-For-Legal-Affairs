import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_cpc_data(input_file, output_file):
    with open(input_file , encoding="utf-8") as f:
        cpc_data = json.load(f)
    
    processed = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Optimal for legal texts
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""]  # Split by sections first
    )

    for order, content in cpc_data["provisions"]["special_provisions"]["orders"].items():
        if content["family_law_relevance"] != "High":
            continue
            
        # Split long sections into coherent chunks
        chunks = text_splitter.split_text(content["content"])
        
        for i, chunk in enumerate(chunks):
            processed.append({
                "text": f"CPC Order {order} [Part {i+1}]: {chunk}",
                "metadata": {
                    "type": "law",
                    "source": "Civil Procedure Code",
                    "section": f"Order {order}",
                    "chunk_id": i,
                    "is_full_text": len(chunks) == 1,
                    "relevance": "High",
                    "keywords": ["family-law", "minors", "guardianship"]
                }
            })
    
    with open(output_file, 'w') as f:
        json.dump(processed, f, indent=2)
# Usage
process_cpc_data("./data/enhanced_cpc_family_law.json", "./data/processed/processed_cpc_laws.json")