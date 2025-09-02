from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
from .rag_pipeline import RAGPipeline
from .vector_store import VectorStore
from .data_ingestion import DocumentProcessor

logger = logging.getLogger(__name__)

class ChatbotEngine:
    """Main chatbot engine that orchestrates all components"""
    def __init__(self, config):
        self.config = config
        self.conversation_history = {} # In production, use Redis or database
        self.feedback_data = [] # Store user feedback
        # Initialize components
        self.vector_store = VectorStore(config)
        self.rag_pipeline = RAGPipeline(config, self.vector_store)
        self.document_processor = DocumentProcessor(config)
        logger.info("ChatbotEngine initialized successfully")

    def ingest_documents(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """Ingest all documents from the documents directory"""
        try:
            if force_reprocess:
                logger.info("Force reprocessing - resetting vector store")
                self.vector_store.reset_collection()
            # Check if documents already exist
            stats = self.vector_store.get_collection_stats()
            if stats.get('total_chunks', 0) > 0 and not force_reprocess:
                logger.info("Documents already ingested. Use force_reprocess=True to re-ingest")
                return {
                    'status': 'already_exists',
                    'message': 'Documents already in vector store',
                    'stats': stats
                }
            # Process documents
            documents = self.document_processor.process_document_directory(
                self.config.DOCUMENTS_DIR
            )
            if not documents:
                return {
                    'status': 'no_documents',
                    'message': 'No valid documents found for processing'
                }
            # Add to vector store
            success = self.vector_store.add_documents(documents)
            if success:
                stats = self.vector_store.get_collection_stats()
                return {
                    'status': 'success',
                    'message': f'Successfully processed {len(documents)} documents',
                    'documents_processed': len(documents),
                    'total_chunks': stats.get('total_chunks', 0),
                    'stats': stats
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to add documents to vector store'
                }
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            return {
                'status': 'error',
                'message': f'Error during ingestion: {str(e)}'
            }

    def chat(self, user_query: str, session_id: str = "default") -> Dict[str, Any]:
        """Main chat interface for processing user queries"""
        try:
            timestamp = datetime.now().isoformat()
            # Initialize session if new
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = {
                    'messages': [],
                    'created_at': timestamp
                }
            # Add user message to history
            self.conversation_history[session_id]['messages'].append({
                'type': 'user',
                'content': user_query,
                'timestamp': timestamp
            })
            # Process query through RAG pipeline
            rag_response = self.rag_pipeline.process_query(user_query)
            # Format response for user
            formatted_response = self._format_response(rag_response)
            # Add assistant response to history
            self.conversation_history[session_id]['messages'].append({
                'type': 'assistant',
                'content': formatted_response['full_response'],
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'sources_used': len(rag_response.get('sources', [])),
                    'detected_language': rag_response.get('detected_language', 'english'),
                    'context_chunks': rag_response.get('context_used', 0)
                }
            })
            return {
                'response': formatted_response,
                'session_id': session_id,
                'sources': rag_response.get('sources', []),
                'bilingual_response': rag_response.get('bilingual_response', {}),
                'metadata': {
                    'processing_time': 'calculated_in_production',
                    'confidence_score': self._calculate_confidence(rag_response),
                    'detected_language': rag_response.get('detected_language', 'english')
                }
            }
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return {
                'response': {
                    'text': "I apologize, but I encountered an error while processing your query. Please try again or contact UPSIDA support.",
                    'full_response': "I apologize, but I encountered an error while processing your query. Please try again or contact UPSIDA support.",
                    'citations': ""
                },
                'session_id': session_id,
                'sources': [],
                'error': str(e)
            }

    def _format_response(self, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """Format RAG response for user display"""
        response_text = rag_response.get('bilingual_response', {}).get('primary', rag_response.get('response', ""))
        sources = rag_response.get('sources', [])
        bilingual = rag_response.get('bilingual_response', {})

        if sources:
            citations = "\n\n **Sources:**\n"
            for idx, source in enumerate(sources[:3], 1): # Limit to top 3 sources
                citations += f"{idx}. {source['document']} (Relevance: {source['similarity_score']:.2f})\n"
            formatted_response = {
                'text': response_text,
                'citations': citations,
                'full_response': response_text + citations,
                'bilingual': bilingual
            }
        else:
            formatted_response = {
                'text': response_text,
                'citations': "",
                'full_response': response_text,
                'bilingual': bilingual
            }
        return formatted_response

    def _calculate_confidence(self, rag_response: Dict[str, Any]) -> float:
        """Calculate confidence score based on source relevance"""
        sources = rag_response.get('sources', [])
        if not sources:
            return 0.0
        avg_similarity = sum(source['similarity_score'] for source in sources) / len(sources)
        return min(avg_similarity * 1.2, 1.0) # Boost slightly, cap at 1.0

    def submit_feedback(self, session_id: str, message_index: int,
                        feedback: str, rating: int = None) -> Dict[str, Any]:
        """Handle user feedback (thumbs up/down mechanism)"""
        try:
            feedback_entry = {
                'session_id': session_id,
                'message_index': message_index,
                'feedback': feedback, # 'positive', 'negative', or text
                'rating': rating, #1-5 scale if provided
                'timestamp': datetime.now().isoformat()
            }
            # In production, store in database
            self.feedback_data.append(feedback_entry)
            logger.info(f"Feedback received: {feedback} for session {session_id}")
            return {
                'status': 'success',
                'message': 'Thank you for your feedback! It helps us improve our service.'
            }
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            return {
                'status': 'error',
                'message': 'Failed to record feedback'
            }

    def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Retrieve conversation history for a session"""
        return self.conversation_history.get(session_id, {
            'messages': [],
            'created_at': None
        })

    def get_system_status(self) -> Dict[str, Any]:
        """Get system health and statistics"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            return {
                'status': 'operational',
                'vector_store': vector_stats,
                'total_sessions': len(self.conversation_history),
                'total_feedback_entries': len(self.feedback_data),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }