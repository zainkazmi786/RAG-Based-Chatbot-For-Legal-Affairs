from retrieval.retriever import HybridRetriever
from generation.llm_chain import JudgmentGenerator
import json
from langchain.schema import Document  
from pathlib import Path
import traceback

def chat_interface():
    try:
        # Initialize with base data
        retriever = HybridRetriever(
            "data/processed/processed_cases.json",
            new_data_path=["data/processed/FamilyCourtsAct1964_processed.json" , "data/processed/processed_cpc_laws.json" ,"data/processed/processed_ordinance_data.json"]  # Optional new data
        )
        generator = JudgmentGenerator("generation/prompts/legal_judgment.txt")
        
        print("Pakistan Family Law Expert System (Type 'quit' to exit)")
        print(f"Knowledge base contains {len(retriever.documents)} documents")
        
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ('quit', 'exit'):
                break

            # Handle special admin commands
            if user_input.startswith("!add "):
                new_data_path = user_input[5:].strip()
                try:
                    if not Path(new_data_path).exists():
                        print(f"Error: File not found at {new_data_path}")
                        continue
                        
                    if retriever._add_new_data(new_data_path):
                        retriever.vector_db = retriever._initialize_vector_store()
                        print(f"Updated with {len(retriever.documents)} total docs")
                    else:
                        print("No new documents added (duplicates detected)")
                except Exception as e:
                    print(f"Addition failed: {str(e)}")
                continue

            # Normal query processing
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
            
            # Generate response with all relevant data
            # response = unique_docs
            response = generator.generate(user_input, unique_docs)
            
            print("\nAssistant:")
            print(response)

    except Exception as e:
        print(f"\nError: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    chat_interface()