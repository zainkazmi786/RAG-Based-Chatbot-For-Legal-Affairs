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
    
    def _determine_query_type(self, query):
        query = query.lower()
        if any(word in query for word in ["what is", "define", "meaning of"]):
            return "term_definition"
        elif any(word in query for word in ["scenario", "what if", "predict"]):
            return "scenario_prediction"
        else:
            return "general_question"

    def generate(self, user_input, precedents):
        query_type = self._determine_query_type(user_input)
        
        case_numbers = ", ".join(
            p.metadata.get("case_number", "N/A") 
            for p in precedents
        ) if precedents else "N/A"

        return self.chain.invoke({
            "input": user_input,
            "precedents": "\n".join(
                f"Case {p.metadata.get('case_number', 'N/A')}: {p.page_content[:300]}..."
                for p in precedents
            ),
            "case_numbers": case_numbers,
            "query_type": query_type
        })