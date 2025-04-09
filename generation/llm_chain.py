from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  
from langchain_core.output_parsers import StrOutputParser
import os
from config.settings import GROQ_API_KEY
import json

class JudgmentGenerator:
    def __init__(self, prompt_template_path):
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()
        
        self.chain = (
            ChatPromptTemplate.from_template(self.prompt_template)
            | ChatGroq(
                api_key=GROQ_API_KEY,
                model="llama3-70b-8192",  # Top choice for legal work
                temperature=0.2,  # More deterministic for legal answers
                max_tokens=1024  # For longer judgments
            )
            | StrOutputParser()
        )
    
    def generate(self, facts, precedents):
        case_numbers = ", ".join(
            [p.metadata["case_number"] for p in precedents]
        )
        return self.chain.invoke({
            "facts": facts,
            "precedents": "\n".join(
                f"Case {p.metadata['case_number']} ({p.metadata['date']}):\n{p.page_content[:500]}..."
                for p in precedents
            ),
            "case_numbers": case_numbers
        })