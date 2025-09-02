from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Any, Optional
import logging
from .vector_store import VectorStore
from .language_detector import LanguageDetector

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Core RAG (Retrieval-Augmented Generation) pipeline"""
    def __init__(self, config, vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.language_detector = LanguageDetector()
        # Initialize LLM
        self._initialize_llm()
        # System prompt for the LLM
        self.system_prompt = """You are an AI assistant for UPSIDA (Uttar Pradesh State Industrial Development Authority).
Your role is to provide accurate, helpful information based ONLY on the official UPSIDA documents provided as context.
CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain relevant information, say "I don't have information about this in the available documents"
3. Always cite your sources by mentioning the document name and section
4. Be precise and factual - do not speculate or add information not in the context
5. For policy or procedural questions, quote relevant sections when helpful
6. Keep responses clear and concise
7. If asked about processes, provide step-by-step guidance when available in the context
Context: {context}
Query: {query}
Response:"""

    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            logger.info(f"Loading LLM model: {self.config.LLM_MODEL}")
            # For demonstration, using a smaller model
            # In production, replace with a more powerful model like GPT-3.5/4 or Llama
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-small", # Smaller model for demo
                padding_side='left'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.use_pipeline = False
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            # Fallback to a text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            self.use_pipeline = True

    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the query"""
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        # Translate query to English if needed for better retrieval
        english_query = self.language_detector.translate_to_english(query)
        # Search vector store
        results = self.vector_store.search_similar(english_query, top_k)
        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity_score'] > 0.5 # Adjust threshold as needed
        ]
        logger.info(f"Retrieved {len(filtered_results)} relevant chunks for query")
        return filtered_results

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context"""
        try:
            # Prepare context
            context_text = "\n\n".join([
                f"Source: {chunk['metadata']['file_name']}\n{chunk['text']}"
                for chunk in context_chunks
            ])
            # Create prompt
            prompt = self.system_prompt.format(
                context=context_text[:2000], # Limit context length
                query=query
            )
            # Generate response
            if self.use_pipeline:
                # Using pipeline fallback
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt.split()) + 100,
                    num_return_sequences=1,
                    temperature=self.config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.text_generator.tokenizer.eos_token_id
                )[0]['generated_text']
                # Extract only the new text
                generated_text = response[len(prompt):].strip()
            else:
                # Using direct model
                inputs = self.tokenizer.encode(prompt, return_tensors='pt')
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + self.config.MAX_TOKENS,
                        temperature=self.config.TEMPERATURE,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:],
                    skip_special_tokens=True
                ).strip()
            # Prepare source citations
            sources = []
            for chunk in context_chunks:
                metadata = chunk['metadata']
                sources.append({
                    'document': metadata['file_name'],
                    'chunk_index': metadata['chunk_index'],
                    'similarity_score': chunk['similarity_score']
                })
            return {
                'response': generated_text,
                'sources': sources,
                'context_used': len(context_chunks),
                'query': query
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': "I apologize, but I encountered an error while processing your query. Please try again.",
                'sources': [],
                'context_used': 0,
                'query': query,
                'error': str(e)
            }

    def process_query(self, query: str) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve + generate"""
        logger.info(f"Processing query: {query[:100]}...")
        # Detect original language
        original_language = self.language_detector.detect_language(query)
        # Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        if not context_chunks:
            response_data = {
                'response': "I don't have information about this topic in the available UPSIDA documents. Please contact the relevant UPSIDA department for assistance.",
                'sources': [],
                'context_used': 0,
                'query': query
            }
        else:
            # Generate response
            response_data = self.generate_response(query, context_chunks)
        # Add bilingual support
        bilingual_response = self.language_detector.get_bilingual_response(
            response_data['response'],
            original_language
        )
        response_data.update({
            'bilingual_response': bilingual_response,
            'detected_language': original_language
        })
        return response_data