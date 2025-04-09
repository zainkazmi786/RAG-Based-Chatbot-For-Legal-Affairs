from retrieval.retriever import HybridRetriever
from generation.llm_chain import JudgmentGenerator
import json
import traceback

def chat_interface():
    # Initialize components
    try:
        retriever = HybridRetriever("data/processed/processed_cases.json")
        generator = JudgmentGenerator("generation/prompts/legal_judgment.txt")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    print("Pakistan Family Law Expert System (Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ('quit', 'exit'):
                break

            # Process all query types
            precedents = retriever.retrieve(user_input)["vector"]
            response = generator.generate(user_input, precedents)
            
            print("\nAssistant:")
            print(response)

        except Exception as e:
            print(f"\nError: {str(e)}\nPlease rephrase your question")

if __name__ == "__main__":
    chat_interface()