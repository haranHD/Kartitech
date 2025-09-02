import unittest
import sys
from pathlib import Path
import os
import shutil
from datetime import datetime
import pandas as pd

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config.config import Config
from src.chatbot_engine import ChatbotEngine
from src.data_ingestion import DocumentProcessor
from src.vector_store import VectorStore

class TestChatbotEngine(unittest.TestCase):
    """Test cases for the UPSIDA Chatbot system"""

    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.test_data_dir = self.config.PROJECT_ROOT / "test_data"
        self.config.DATA_DIR = self.test_data_dir
        self.config.DOCUMENTS_DIR = self.test_data_dir / "documents"
        self.config.PROCESSED_DIR = self.test_data_dir / "processed"
        self.config.VECTOR_DB_PATH = str(self.test_data_dir / "chromadb")
        self.config.create_directories()
        # Create a simple test document
        test_doc_content = """
UPSIDA Test Policy
Section 1: Test Requirements
1.1 All applications must include valid documentation.
1.2 Processing time is 15 working days.
Section 2: Test Procedures
2.1 Online application submission required.
2.2 Physical verification may be conducted.
"""
        test_file = self.config.DOCUMENTS_DIR / "test_policy.txt"
        with open(test_file, 'w') as f:
            f.write(test_doc_content)
        
        self.chatbot = ChatbotEngine(self.config)

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_document_processing(self):
        """Test document processing functionality"""
        processor = DocumentProcessor(self.config)
        # Test processing directory
        documents = processor.process_document_directory(self.config.DOCUMENTS_DIR)
        self.assertGreater(len(documents), 0, "Should process at least one document")
        #Check document structure
        doc = documents[0]
        required_keys = ['document_id', 'metadata', 'full_text', 'chunks']
        for key in required_keys:
            self.assertIn(key, doc, f"Document should contain {key}")

    def test_chatbot_initialization(self):
        """Test chatbot engine initialization"""
        self.assertIsNotNone(self.chatbot.vector_store)
        self.assertIsNotNone(self.chatbot.rag_pipeline)
        self.assertIsNotNone(self.chatbot.document_processor)

    def test_ingest_documents(self):
        """Test document ingestion process"""
        result = self.chatbot.ingest_documents(force_reprocess=True)
        self.assertEqual(result['status'], 'success')
        self.assertGreater(result['total_chunks'], 0)

    def test_query_processing(self):
        """Test end-to-end query processing"""
        self.chatbot.ingest_documents(force_reprocess=True)
        # Test query
        response = self.chatbot.chat("What is the processing time?", "test_session")
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], dict)
        self.assertIn('text', response['response'])
        self.assertIn("15 working days", response['response']['text'])
        self.assertGreater(len(response['sources']), 0)

    def test_bilingual_support(self):
        """Test bilingual query support"""
        self.chatbot.ingest_documents(force_reprocess=True)
        # Test Hindi query
        hindi_query = "आवेदन की प्रक्रिया क्या है?"
        response = self.chatbot.chat(hindi_query, "test_hindi_session")
        self.assertIn('bilingual_response', response)
        bilingual = response['bilingual_response']
        self.assertIn('hindi', bilingual)
        self.assertIn('english', bilingual)
        # The primary response text should be in the original language
        self.assertEqual(response['metadata']['detected_language'], 'hindi')
        self.assertIn(bilingual['primary'], bilingual['hindi'])

    def test_feedback_mechanism(self):
        """Test feedback submission"""
        feedback_result = self.chatbot.submit_feedback(
            session_id="test_session",
            message_index=0,
            feedback="positive",
            rating=5
        )
        self.assertEqual(feedback_result['status'], 'success')
        self.assertGreater(len(self.chatbot.feedback_data), 0)

    def test_get_system_status(self):
        """Test system status retrieval"""
        self.chatbot.ingest_documents(force_reprocess=True)
        status = self.chatbot.get_system_status()
        self.assertEqual(status['status'], 'operational')
        self.assertIn('total_chunks', status['vector_store'])
        self.assertGreater(status['vector_store']['total_chunks'], 0)

if __name__ == "__main__":
    unittest.main()