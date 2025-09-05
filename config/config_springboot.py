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
    PROCESSED_DIR = DATA_DIR / "processed" # A single directory for all processed outputs
    VECTOR_DB_PATH = str(DATA_DIR / "chromadb") # A single database for all vector collections

    # --- Models ---
    # Model for converting text into vector embeddings for searching.
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Language Model for generating answers. Flan-T5 is designed specifically for question-answering tasks.
    LLM_MODEL = "google/flan-t5-base"  # CRITICAL FIX: Replaced DialoGPT with a model that can follow instructions.

    # --- RAG Pipeline Settings ---
    # The size of text chunks in words. Smaller chunks are more specific and easier to retrieve.
    CHUNK_SIZE = 200  # CRITICAL FIX: Reduced from 500 for more precise retrieval.
    
    # How many words chunks should overlap to maintain context between them.
    CHUNK_OVERLAP = 25
    
    # The number of relevant chunks to retrieve from the database for a given query.
    TOP_K_RESULTS = 5
    
    # The minimum similarity score for a chunk to be considered relevant.
    SIMILARITY_THRESHOLD = 0.2  # CRITICAL FIX: Lowered to be less strict and find more potential answers.
    
    # LLM generation parameters
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.1
    
    # --- Knowledge Base Identifiers ---
    # Unique names for each chatbot's data collection within the vector database.
    # This allows you to have separate knowledge bases for different documents.
    UPSIDA_COLLECTION_NAME = "upsida_documents"
    SPRINGBOOT_COLLECTION_NAME = "springboot_documents" # This will be used for your spring_boot_tutorial.pdf

    # --- Other Settings ---
    OCR_LANGUAGE = "eng+hin"  # Languages for Optical Character Recognition for scanned documents.
    
    @classmethod
    def create_directories(cls):
        """A helper method to create the necessary project directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.DOCUMENTS_DIR, cls.PROCESSED_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

