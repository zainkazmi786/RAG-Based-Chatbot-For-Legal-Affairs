import os
import json
import time
from pathlib import Path

def ensure_directory_exists(directory_path):
    """Ensure that a directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def load_json_file(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
        return None

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    # Ensure directory exists
    ensure_directory_exists(os.path.dirname(file_path))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def format_timestamp(timestamp=None):
    """Format timestamp to human-readable format."""
    if timestamp is None:
        timestamp = time.time()
    
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def format_document_for_display(doc):
    """Format a document for display in the UI."""
    # Extract relevant metadata
    metadata = doc.metadata
    source_type = "Unknown"
    
    if "case_number" in metadata:
        source_type = "Case Law"
        source_id = metadata.get("case_number", "N/A")
    elif metadata.get('source') == "Civil Procedure Code":
        source_type = "Civil Procedure Code"
        source_id = metadata.get("section", "N/A")
    elif "act_name" in metadata:
        source_type = metadata.get("act_name", "Statute")
        source_id = f"Section {metadata.get('section_number', 'N/A')}"
    else:
        source_id = "N/A"
    
    return {
        "type": source_type,
        "id": source_id,
        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
        "full_content": doc.page_content,
        "metadata": metadata
    }

def truncate_text(text, max_length=100):
    """Truncate text to a maximum length and add ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."