from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from config.settings import GROQ_API_KEY
import json

class JudgmentGenerator:
    def __init__(self, prompt_template_path):
        # Load a unified prompt that lets the LLM classify the query
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()

        self.chain = (
            ChatPromptTemplate.from_template(self.prompt_template)
            | ChatGroq(
                api_key=GROQ_API_KEY,
                model="llama3-70b-8192",  # Top choice for legal work
                temperature=0.2,  # More deterministic for legal answers
                max_tokens=1024   # For longer judgments
            )
            | StrOutputParser()
        )

    def _format_law(self, lp):
        section_num = lp.metadata.get('section_number', 'N/A')
        if lp.metadata.get('parent_section'):
            section_num = f"{lp.metadata['parent_section']}({section_num})"

        base = (
            f"{lp.metadata.get('act_name', 'Law')} "
            f"{lp.metadata.get('section_type', 'Section').title()} {section_num}: "
            f"{lp.metadata.get('section_title', '')}\n"
            f"{lp.page_content}"
        )
        if lp.metadata.get('is_key_section', False):
            base += "\n[KEY LEGAL PRINCIPLE]"
        return base + "..."

    def generate(self, user_input, retrieved_docs):
        # Classify docs into categories
        precedents = []
        legal_provisions = []
        cpc_orders = []

        for doc in retrieved_docs:
            metadata = doc.metadata
            if "case_number" in metadata:
                precedents.append(doc)
            elif metadata.get('source') == "Civil Procedure Code":
                cpc_orders.append(doc)
            else:
                legal_provisions.append(doc)

        # Prepare formatted texts
        precedent_texts = (
            "\n".join(
                f"Case {p.metadata.get('case_number', 'N/A')}: {p.page_content[:1000]}..."
                for p in precedents
            ) if precedents else "None"
        )
        statute_texts = (
            "\n\n".join(self._format_law(lp) for lp in legal_provisions)
            if legal_provisions else "None"
        )
        cpc_texts = (
            "\n".join(
                f"CPC {co.metadata.get('section', 'Order')}: {co.page_content}..."
                for co in cpc_orders
            ) if cpc_orders else "None"
        )

        # Invoke the chain; LLM will determine query type and apply appropriate prompt
        return self.chain.invoke({
            "user_input": user_input,
            "precedents": precedent_texts,
            "legal_provisions": statute_texts,
            "cpc_orders": cpc_texts,
            "has_legal_sources": bool(legal_provisions or cpc_orders)
        })
