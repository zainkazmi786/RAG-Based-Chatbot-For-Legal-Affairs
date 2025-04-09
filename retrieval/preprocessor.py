import re
import json
from pathlib import Path

def clean_text(text):
    # Replace non-breaking hyphen (and similar unicode dashes) with normal hyphen
    text = text.replace("\u2011", "-")
    
    # Remove sequences of repeated dashes (common as dividers in legal texts)
    text = re.sub(r'-{2,}', ' ', text)
    
    # Standardize case numbers and similar patterns (e.g., "1968 P Cr")
    # Adjust the regex pattern as needed for your specific formats
    text = re.sub(r'(\d{4})\s*([A-Z]+)\s*(\d+)', r'\1 \2 \3', text)
    
    # Optionally: normalize quotes (uncomment if needed)
    # text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    
    # Remove extra whitespace (newlines, tabs, multiple spaces)
    text = re.sub(r'\s+', ' ', text)
    
    # Trim any leading/trailing whitespace
    return text.strip()

def process_cases(input_path, output_dir):
    """Convert raw JSON to cleaned documents with metadata"""
    with open(input_path) as f:
        cases = json.load(f)
    
    processed = []
    for case in cases:
        processed.append({
            "text": clean_text(case["caseDetails"]),
            "metadata": {
                "case_number": case["caseNumber"],
                "location": case["location"],
                "judge": case["authorJudge"],
                "subject": case["caseSubject"],
                "date": case["dateOfAnnouncement"]
            }
        })
    
    # Save processed data
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "processed_cases.json", 'w', encoding='utf-8') as f:  # Fix here
        json.dump(processed, f, ensure_ascii=False, indent=2)  # indent for readability


if __name__ == "__main__":
    process_cases(
        input_path=Path("./data/cases1.json"),
        output_dir=Path("./data/processed")
    )
