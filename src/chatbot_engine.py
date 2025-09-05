from .rag_pipeline import RAGPipeline
from .vector_store import VectorStore
from .data_ingestion import DocumentProcessor
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ChatbotEngine:
    """Orchestrates the chatbot components for a specific knowledge base."""
    def __init__(self, config, collection_name: str):
        self.config = config
        self.collection_name = collection_name
        
        logger.info(f"Initializing engine for collection: '{collection_name}'")
        self.vector_store = VectorStore(config, collection_name)
        self.rag_pipeline = RAGPipeline(config, self.vector_store)
        self.document_processor = DocumentProcessor(config)
        self.conversation_history = {} # Simple in-memory history

    def ingest_documents(self, force_reprocess: bool = False, target_file: str = None) -> Dict[str, Any]:
        """Process and load documents into the vector store."""
        try:
            if force_reprocess:
                self.vector_store.reset_collection()
            
            documents = self.document_processor.process_document_directory(
                self.config.DOCUMENTS_DIR, target_file=target_file
            )
            
            if not documents:
                return {'status': 'no_documents', 'message': 'No valid documents found.'}
                
            if self.vector_store.add_documents(documents):
                return {'status': 'success', 'message': f'Successfully processed {len(documents)} documents.'}
            else:
                return {'status': 'error', 'message': 'Failed to add documents to vector store.'}
        except Exception as e:
            logger.error(f"Error during ingestion for '{self.collection_name}': {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}

    def chat(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """Handle a user's chat query."""
        try:
            response_data = self.rag_pipeline.process_query(user_query)
            
            # Format the final response for the user
            response_text = response_data.get('bilingual_response', {}).get('primary', "Sorry, I couldn't generate a response.")
            sources = response_data.get('sources', [])
            
            return {'response': response_text, 'sources': sources}
        except Exception as e:
            logger.error(f"Error in chat processing: {e}", exc_info=True)
            return {'response': "An internal error occurred.", 'sources': []}
