import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Load from .env
LEGAL_MODEL = "deepseek-legal"  # Official model name