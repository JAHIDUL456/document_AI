import os 
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT =int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "policy_AI")

EMBED_MODEL= os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

OLLAMA_URL= os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct")

TOP_K= int(os.getenv("TOP_K", 2))