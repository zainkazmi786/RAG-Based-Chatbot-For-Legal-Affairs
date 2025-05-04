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

    def generate(self, user_input, retrieved_docs):
        query_type = self._determine_query_type(user_input)
        
        precedents = []
        legal_provisions = []
        cpc_orders = []
        
        for doc in retrieved_docs:
            metadata = doc.metadata
            if "case_number" in metadata:
                precedents.append(doc)
            elif "source" in metadata and metadata["source"] == "Civil Procedure Code":
                cpc_orders.append(doc)
            else:  # Catch-all for laws/ordinances
                legal_provisions.append(doc)

        def format_law(lp):
            section_num = lp.metadata.get('section_number', 'N/A')
            if lp.metadata.get('parent_section'):
                section_num = f"{lp.metadata['parent_section']}({section_num})"
                
            base = (f"{lp.metadata.get('act_name', 'Law')} "
                f"{lp.metadata.get('section_type', 'Section').title()} {section_num}: "
                f"{lp.metadata.get('section_title', '')}\n"
                f"{lp.page_content}")
                
            if lp.metadata.get('is_key_section', False):
                base += "\n[KEY LEGAL PRINCIPLE]"
            return base + "..."

        legal_provision_texts = "\n\n".join(format_law(lp) for lp in legal_provisions) if legal_provisions else "None"
        
        return self.chain.invoke({
            "user_input": user_input,
            "precedents": "\n".join(f"Case {p.metadata.get('case_number', 'N/A')}: {p.page_content[:1000]}..." 
                        for p in precedents) if precedents else "None",
            "legal_provisions": legal_provision_texts,
            "cpc_orders": "\n".join(f"CPC {co.metadata.get('section', 'Order')}: {co.page_content}..." 
                        for co in cpc_orders) if cpc_orders else "None",
            "query_type": query_type,
            "has_legal_sources": bool(legal_provisions or cpc_orders)
        })
        # return self.chain.invoke({
        #     "input": user_input,
        #     "precedents": precedent_texts,
        #     "legal_provisions": legal_provision_texts,
        #     "cpc_orders": cpc_order_texts,
        #     "case_numbers": case_numbers,
        #     "query_type": query_type,
        #     "has_precedents": bool(precedents),
        #     "has_laws": bool(legal_provisions) or bool(cpc_orders)
        # })