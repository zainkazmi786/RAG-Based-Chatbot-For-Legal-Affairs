import re
import pdfplumber
import json

def extract_ordinance_sections(pdf_path):
    sections = []
    current_section = {}

    # Regex patterns
    # Captures formats like: 1. Title, 1.Title, 1. — Title, etc.
    section_pattern = re.compile(r'^(\d{1,3})\.\s*[—-]?\s*(.*)', re.IGNORECASE)
    # Subsections like: (a), (b), etc., possibly with content after
    subsect_pattern = re.compile(r'^\(([a-z])\)\s+(.*)', re.IGNORECASE)
    # Clause pattern like: (1), (2)
    numbered_clause_pattern = re.compile(r'^\((\d+)\)\s+(.*)')

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = text.split('\n')

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # Match a new section
                section_match = section_pattern.match(line)
                if section_match:
                    # Save previous section
                    if current_section:
                        current_section["text"] = '\n'.join(current_section["text"]).strip()
                        sections.append(current_section)

                    current_section = {
                        "section_number": section_match.group(1),
                        "title": section_match.group(2).strip(),
                        "text": [],
                        "subsections": []
                    }

                # Match a subsection (alphabetical like (a))
                elif subsect_pattern.match(line) and current_section:
                    current_section["subsections"].append(line)

                # Match a numbered clause like (1)
                elif numbered_clause_pattern.match(line) and current_section:
                    current_section["subsections"].append(line)

                # Continue adding text to the current section
                elif current_section:
                    current_section["text"].append(line)

    # Append the last section if exists
    if current_section:
        current_section["text"] = '\n'.join(current_section["text"]).strip()
        sections.append(current_section)

    return sections


# Usage
ordinance_data = extract_ordinance_sections("54-muslim-family-laws-ordinance-1961-viii-of-1961-pdf.pdf")

# Save to JSON
with open("muslim_family_laws.json", "w", encoding='utf-8') as f:
    json.dump({
        "ordinance": "Muslim Family Laws Ordinance 1961",
        "amendment_year": 2015,
        "jurisdiction": "Punjab",
        "sections": ordinance_data
    }, f, indent=2, ensure_ascii=False)
