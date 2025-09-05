import sys
from pathlib import Path
import logging
from typing import Dict

from typing import Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from config.config_springboot import SpringBootConfig
from src.data_ingestion import DocumentProcessor
from src.vector_store import VectorStore
from src.utils import setup_logging

def ingest_documents_springboot(force_reprocess: bool = False) -> Dict[str, Any]:
    """Ingest Spring Boot documents from the documents directory into a dedicated vector store"""
    logger = setup_logging()
    config = SpringBootConfig()
    config.create_directories()

    try:
        vector_store = VectorStore(config)
        document_processor = DocumentProcessor(config)

        if force_reprocess:
            logger.info("Force reprocessing for Spring Boot docs - resetting vector store")
            vector_store.reset_collection()

        stats = vector_store.get_collection_stats()
        if stats.get('total_chunks', 0) > 0 and not force_reprocess:
            logger.info("Spring Boot documents already ingested. Use --reprocess to re-ingest.")
            return {
                'status': 'already_exists',
                'message': 'Spring Boot documents already in vector store',
                'stats': stats
            }

        documents = document_processor.process_document_directory(
            config.DOCUMENTS_DIR,
            target_file_name="spring_boot_tutorial.pdf"
        )
        
        if not documents:
            return {
                'status': 'no_documents',
                'message': 'No valid Spring Boot documents found for processing'
            }

        success = vector_store.add_documents(documents)
        
        if success:
            stats = vector_store.get_collection_stats()
            return {
                'status': 'success',
                'message': f'Successfully processed {len(documents)} Spring Boot documents',
                'documents_processed': len(documents),
                'total_chunks': stats.get('total_chunks', 0),
                'stats': stats
            }
        else:
            return {
                'status': 'error',
                'message': 'Failed to add Spring Boot documents to vector store'
            }
    except Exception as e:
        logger.error(f"Error during Spring Boot document ingestion: {e}")
        return {
            'status': 'error',
            'message': f'Error during ingestion: {str(e)}'
        }

if __name__ == "__main__":
    result = ingest_documents_springboot(force_reprocess=True)
    print(result)
