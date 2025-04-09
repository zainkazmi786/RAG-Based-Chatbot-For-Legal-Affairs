from retrieval.retriever import HybridRetriever
from generation.llm_chain import JudgmentGenerator
import json

# Initialize components
try:
    retriever = HybridRetriever("data/processed/processed_cases.json")
except UnicodeDecodeError as e:
    print(f"Encoding error: {e}\nPlease ensure the file is UTF-8 encoded")
    exit(1)
generator = JudgmentGenerator("generation/prompts/legal_judgment.txt")

def predict_judgment(scenario):
    # Retrieve relevant cases
    results = retriever.retrieve(scenario["facts"])
    
    # Generate judgment
    return generator.generate(
        facts=scenario["facts"],
        precedents=results["vector"]
    )

# Example usage
scenario = {
    "facts": """Summarize the judgment in "Muhammad Iqbal vs Mst. Naila Bibi 2022 SCMR 1020".

What was the reasoning behind the court’s decision in a child custody case in Lahore High Court, 2023?

Which precedent deals with dower (haq mehr) in Pakistan’s Supreme Court rulings?""",
}

print(predict_judgment(scenario))