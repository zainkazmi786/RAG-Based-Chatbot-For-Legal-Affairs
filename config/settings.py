import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Application settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", "5000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model settings
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024

# Retrieval settings
TOP_K_VECTOR = 5
TOP_K_KEYWORD = 3