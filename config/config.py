import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for UPSIDA Chatbot system"""
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    PROCESSED_DIR = DATA_DIR / "processed"
    # Vector Database Settings
    VECTOR_DB_PATH = str(DATA_DIR / "chromadb")
    COLLECTION_NAME = "upsida_documents"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # LLM Settings
    LLM_MODEL = "microsoft/DialoGPT-medium" # Can be replaced with any HuggingFace model
    MAX_TOKENS = 512
    TEMPERATURE = 0.1
    TOP_K_RESULTS = 5
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    OCR_LANGUAGE = "eng+hin" # English + Hindi for Tesseract
    # Language Support
    SUPPORTED_LANGUAGES = ["english", "hindi"]
    DEFAULT_LANGUAGE = "english"
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = True
    # Security
    ALLOWED_FILE_TYPES = [".pdf", ".docx", ".doc", ".txt"]
    MAX_FILE_SIZE = 50 * 1024 * 1024 #50MB
    #Performance
    RESPONSE_TIMEOUT = 30 # seconds
    MAX_CONCURRENT_QUERIES = 100

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.DOCUMENTS_DIR, cls.PROCESSED_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)