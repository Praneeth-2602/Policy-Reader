# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Settings:
    # Google Gemini API Key
    GEMINI_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # LLM Model names
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    EMBEDDING_MODEL_NAME: str = "text-embedding-004" # This is used for generating embeddings with Google's model

    # Astra DB Settings
    ASTRA_DB_API_ENDPOINT: str = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN: str = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE: str = os.getenv("ASTRA_DB_KEYSPACE", None) # Optional: provide a default or leave as None
    ASTRA_DB_COLLECTION_NAME: str = "policy_documents_vector_collection" # Name of your vector collection in Astra DB

    # NeonDB/PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # Paths (still relevant for local data, but FAISS_INDEX_PATH is no longer needed)
    DATA_DIR: str = "data"
    KNOWLEDGE_BASE_DIR: str = os.path.join(DATA_DIR, "knowledge_base")

    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Query Parsing Schema
    QUERY_SCHEMA = {
        "age": "integer",
        "gender": "string",
        "procedure": "string",
        "location": "string",
        "policy_duration_months": "integer",
        "keywords": "list"
    }

settings = Settings()