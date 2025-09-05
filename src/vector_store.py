import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations using ChromaDB."""
    def __init__(self, config, collection_name: str):
        self.config = config
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": f"Embeddings for {collection_name}"}
        )
        logger.info(f"Vector store initialized for collection: '{self.collection_name}'")

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add processed documents to the vector store."""
        try:
            all_chunks, all_metadatas, all_ids = [], [], []
            for doc in documents:
                doc_metadata = doc['metadata']
                for chunk in doc['chunks']:
                    chunk_id = f"{doc['document_id']}_chunk_{chunk['chunk_index']}"
                    chunk_metadata = {'document_id': doc['document_id'], 'file_name': doc_metadata['file_name'], **chunk}
                    
                    all_chunks.append(chunk['text'])
                    all_metadatas.append(chunk_metadata)
                    all_ids.append(chunk_id)
            
            if not all_chunks:
                logger.warning("No chunks to add to the vector store.")
                return False

            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
            
            self.collection.add(embeddings=embeddings.tolist(), documents=all_chunks, metadatas=all_metadatas, ids=all_ids)
            logger.info(f"Successfully added {len(all_chunks)} chunks to '{self.collection_name}'.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for document chunks similar to the query."""
        try:
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            for i, distance in enumerate(results['distances'][0]):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - distance,
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def reset_collection(self):
        """Delete and recreate the collection."""
        logger.warning(f"Resetting vector store collection: '{self.collection_name}'")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info("Vector store reset successfully.")
