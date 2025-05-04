import json
from typing import Dict, List

def preprocess_family_courts_act(input_file: str, output_file: str) -> None:
    """
    Preprocess the Family Courts Act JSON file to create documents with text and metadata
    suitable for a RAG system.
    
    Args:
        input_file: Path to the original JSON file
        output_file: Path to save the preprocessed JSON file
    """
    
    # Load the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Extract the main act data
    act_data = original_data.get("FamilyCourtsAct", {})
    
    # Initialize list to hold processed documents
    documents = []
    
    # Create a document for the basic act information
    documents.append({
        "text": f"Family Courts Act, 1964 (Enacted on {act_data.get('dateEnacted', '')})",
        "metadata": {
            "title": act_data.get("title", ""),
            "short_title": act_data.get("shortTitle", ""),
            "date_enacted": act_data.get("dateEnacted", ""),
            "document_type": "act_overview"
        }
    })
    
    # Process each section
    for section in act_data.get("contents", []):
        section_num = section.get("section", "")
        section_title = section.get("title", "")
        
        # Handle sections with definitions differently
        if "definitions" in section:
            definitions = section["definitions"]
            definition_text = "\n".join([f"{key}: {value}" for key, value in definitions.items()])
            
            documents.append({
                "text": f"Section {section_num}: {section_title}\nDefinitions:\n{definition_text}",
                "metadata": {
                    "section": section_num,
                    "title": section_title,
                    "content_type": "definitions",
                    "act": "Family Courts Act, 1964"
                }
            })
        else:
            # Handle regular sections with details
            details = section.get("details", {})
            details_text = ""
            
            if isinstance(details, str):
                details_text = details
            elif isinstance(details, dict):
                details_text = "\n".join([f"{key}. {value}" if not isinstance(value, dict) else 
                                         f"{key}. {value.get('point', '')}" 
                                         for key, value in details.items()])
            
            documents.append({
                "text": f"Section {section_num}: {section_title}\n{details_text}",
                "metadata": {
                    "section": section_num,
                    "title": section_title,
                    "content_type": "section_details",
                    "act": "Family Courts Act, 1964"
                }
            })
    
    # Process the schedule separately
    schedule = act_data.get("schedule", {})
    schedule_text = "Schedule:\n"
    
    for part, items in schedule.items():
        schedule_text += f"\n{part}:\n"
        if isinstance(items, list):
            schedule_text += "\n".join([f"- {item}" for item in items])
        else:
            schedule_text += str(items)
    
    documents.append({
        "text": schedule_text,
        "metadata": {
            "content_type": "schedule",
            "act": "Family Courts Act, 1964"
        }
    })
    
    # Save the processed documents
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_file = "./data/FamilyCourtsAct1964.json"
    output_file = "./data/processed/FamilyCourtsAct1964_processed.json"
    preprocess_family_courts_act(input_file, output_file)
    print(f"Preprocessing complete. Output saved to {output_file}")