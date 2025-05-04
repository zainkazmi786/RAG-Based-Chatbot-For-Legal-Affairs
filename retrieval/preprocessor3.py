import json
from typing import Dict, List, Any

def preprocess_ordinance_data(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Preprocesses the raw ordinance data to make it suitable for vector database storage.
    
    Args:
        raw_data: The raw JSON data of the ordinance
        
    Returns:
        List of cleaned document chunks suitable for vector database
    """
    processed_documents = []
    
    # Extract metadata
    metadata = raw_data.get("metadata", {})
    act_name = metadata.get("act_name", "Muslim Family Laws Ordinance, 1961")
    
    # Process preamble
    preamble = raw_data.get("structured_data", {}).get("preamble", "")
    if preamble:
        processed_documents.append({
            "text": preamble,
            "metadata": {
                **metadata,
                "section_type": "preamble",
                "section_number": "0",
                "section_title": "Preamble"
            }
        })
    
    # Process definitions
    definitions = raw_data.get("structured_data", {}).get("definitions", {})
    if definitions:
        definitions_text = "\n".join([f"{term}: {definition}" for term, definition in definitions.items()])
        processed_documents.append({
            "text": definitions_text,
            "metadata": {
                **metadata,
                "section_type": "definitions",
                "section_number": "2",
                "section_title": "Definitions"
            }
        })
    
    # Process sections
    sections = raw_data.get("structured_data", {}).get("sections", [])
    for section in sections:
        section_number = section.get("number", "")
        section_title = section.get("title", "")
        section_content = section.get("content", "")
        is_key_section = section.get("is_key_section", False)
        
        # Create base document for the section
        section_doc = {
            "text": f"{section_title}\n{section_content}",
            "metadata": {
                **metadata,
                "section_type": "section",
                "section_number": section_number,
                "section_title": section_title,
                "is_key_section": is_key_section
            }
        }
        processed_documents.append(section_doc)
        
        # Process subsections if they exist
        subsections = section.get("subsections", [])
        for subsection in subsections:
            subsection_number = subsection.get("number", "")
            subsection_content = subsection.get("content", "")
            
            subsection_doc = {
                "text": f"{section_title} - Subsection {subsection_number}\n{subsection_content}",
                "metadata": {
                    **metadata,
                    "section_type": "subsection",
                    "section_number": f"{section_number}.{subsection_number}",
                    "section_title": f"{section_title} - Subsection {subsection_number}",
                    "parent_section": section_number,
                    "is_key_section": is_key_section
                }
            }
            processed_documents.append(subsection_doc)
    
    # Process key provisions
    key_provisions = raw_data.get("structured_data", {}).get("key_provisions", {})
    for provision_type, provisions in key_provisions.items():
        if provisions:
            provision_texts = []
            for provision in provisions:
                section_num = provision.get("number", "")
                content = provision.get("content", "")
                provision_texts.append(f"Section {section_num}: {content}")
            
            provision_doc = {
                "text": f"Key Provisions for {provision_type}:\n" + "\n".join(provision_texts),
                "metadata": {
                    **metadata,
                    "section_type": "key_provisions",
                    "provision_category": provision_type,
                    "is_key_section": True
                }
            }
            processed_documents.append(provision_doc)
    
    return processed_documents

# Example usage
if __name__ == "__main__":
    # Load the raw JSON data
    with open("./data/muslim_family_laws_ordinance_1961.json", "r") as f:
        raw_data = json.load(f)[0]  # Assuming the structure is a list with one item
    
    # Preprocess the data
    processed_data = preprocess_ordinance_data(raw_data)
    
    # Save the processed data
    with open("./data/processed/processed_ordinance_data.json", "w") as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"Processed {len(processed_data)} documents for vector database storage.")