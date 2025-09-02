import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations using ChromaDB"""
    def __init__(self, config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        #Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=config.COLLECTION_NAME
            )
            logger.info(f"Loaded existing collection: {config.COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"description": "UPSIDA Document Embeddings"}
            )
            logger.info(f"Created new collection: {config.COLLECTION_NAME}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add processed documents to vector store"""
        try:
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []
            for doc in documents:
                doc_id = doc['document_id']
                doc_metadata = doc['metadata']
                for idx, chunk in enumerate(doc['chunks']):
                    chunk_id = f"{doc_id}_chunk_{idx}"
                    chunk_text = chunk['text']
                    # Prepare metadata for this chunk
                    chunk_metadata = {
                        'document_id': doc_id,
                        'file_name': doc_metadata['file_name'],
                        'chunk_index': idx,
                        'start_index': chunk['start_index'],
                        'end_index': chunk['end_index'],
                        'word_count': chunk['word_count'],
                        'file_type': doc_metadata['file_type']
                    }
                    all_chunks.append(chunk_text)
                    all_metadatas.append(chunk_metadata)
                    all_ids.append(chunk_id)
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings = self.generate_embeddings(all_chunks)
            #Add to ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"Successfully added {len(all_chunks)} chunks to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar document chunks"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i], # Convert distance to similarity
                    'chunk_id': results['ids'][0][i] if 'ids' in results else f"chunk_{i}"
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.config.COLLECTION_NAME,
                'embedding_model': self.config.EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def reset_collection(self) -> bool:
        """Reset the vector store (useful for development)"""
        try:
            self.client.delete_collection(self.config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.config.COLLECTION_NAME,
                metadata={"description": "UPSIDA Document Embeddings"}
            )
            logger.info("Vector store reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            return False