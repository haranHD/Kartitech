from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Dict, Any
import logging
from .vector_store import VectorStore
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class RAGPipeline:
    """The core Retrieval-Augmented Generation pipeline."""
    def __init__(self, config, vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.language_detector = LanguageDetector()
        self._initialize_llm()
        self.system_prompt = """Answer the following query based only on the provided context. If the context does not contain the answer, state that you don't have enough information.
Context: {context}
Query: {query}
Answer:"""

    def _initialize_llm(self):
        try:
            logger.info(f"Loading LLM model: {self.config.LLM_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.LLM_MODEL)
            self.text_generator = pipeline(
                "text2text-generation", model=self.model, tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("LLM model initialized successfully.")
        except Exception as e:
            logger.error(f"Fatal error loading LLM model: {e}")
            raise

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks from the vector store."""
        english_query = self.language_detector.translate_to_english(query)
        results = self.vector_store.search_similar(english_query, self.config.TOP_K_RESULTS)
        
        filtered_results = [r for r in results if r['similarity_score'] > self.config.SIMILARITY_THRESHOLD]
        logger.info(f"Retrieved {len(filtered_results)} relevant chunks.")
        return filtered_results

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the LLM with the retrieved context."""
        context = "\n\n".join(chunk['text'] for chunk in context_chunks)
        prompt = self.system_prompt.format(context=context, query=query)
        
        try:
            response = self.text_generator(prompt, max_new_tokens=self.config.MAX_NEW_TOKENS)[0]['generated_text']
            sources = [{'document': c['metadata']['file_name'], 'score': c['similarity_score']} for c in context_chunks]
            return {'response': response.strip(), 'sources': sources}
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return {'response': "Error generating response.", 'sources': []}

    def process_query(self, query: str) -> Dict[str, Any]:
        """The complete RAG process for a user query."""
        logger.info(f"Processing query: '{query[:100]}...'")
        original_lang = self.language_detector.detect_language(query)
        context_chunks = self.retrieve_context(query)
        
        if not context_chunks:
            return {'response': "I don't have information on this topic.", 'sources': []}
            
        result = self.generate_response(query, context_chunks)
        bilingual = self.language_detector.get_bilingual_response(result['response'], original_lang)
        result['bilingual_response'] = bilingual
        return result
