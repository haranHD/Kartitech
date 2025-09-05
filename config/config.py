from pathlib import Path

class Config:
    """
    Unified configuration for the entire Document AI Chatbot system.
    This single file manages settings for all chatbots to ensure consistency and prevent errors.
    """
    # --- Universal Project Paths ---
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    PROCESSED_DIR = DATA_DIR / "processed" 
    VECTOR_DB_PATH = str(DATA_DIR / "chromadb") 

    # --- Models ---
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "google/flan-t5-base"

    # --- RAG Pipeline Settings ---
    # CRITICAL FIX: Making chunks smaller to capture specific details like annotations.
    CHUNK_SIZE = 100 
    
    # Overlap to maintain context between the smaller chunks.
    CHUNK_OVERLAP = 20
    
    # Retrieve more chunks to increase the chance of finding specific details.
    TOP_K_RESULTS = 7
    
    SIMILARITY_THRESHOLD = 0.2  
    
    # LLM generation parameters
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.1
    
    # --- Knowledge Base Identifiers ---
    UPSIDA_COLLECTION_NAME = "upsida_documents"
    SPRINGBOOT_COLLECTION_NAME = "springboot_documents"

    # --- Other Settings ---
    OCR_LANGUAGE = "eng+hin"
    
    @classmethod
    def create_directories(cls):
        """A helper method to create the necessary project directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.DOCUMENTS_DIR, cls.PROCESSED_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

