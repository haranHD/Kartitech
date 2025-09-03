import os
from pathlib import Path
from typing import Dict, Any

class SpringBootConfig:
    """Configuration class for the Spring Boot RAG system"""

    # Project Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    PROCESSED_DIR = DATA_DIR / "processed_springboot"

    # Vector Database Settings
    VECTOR_DB_PATH = str(DATA_DIR / "chromadb_springboot")
    COLLECTION_NAME = "springboot_documents"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM Settings
    LLM_MODEL = "microsoft/DialoGPT-small"
    MAX_TOKENS = 512
    TEMPERATURE = 0.1
    TOP_K_RESULTS = 5

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    OCR_LANGUAGE = "eng"  # Spring Boot docs are likely English

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.DOCUMENTS_DIR, cls.PROCESSED_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
