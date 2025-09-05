import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, Any, List, Optional
from .vector_store import VectorStore
from .language_detector import LanguageDetector
import logging

logger = logging.getLogger(__name__)

class SpringBootRAGPipeline:
    """Core RAG (Retrieval-Augmented Generation) pipeline"""
    def __init__(self, config, vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.language_detector = LanguageDetector()
        # Initialize LLM
        self._initialize_llm()
        # System prompt for the LLM
        self.system_prompt = """You are an AI assistant who is an expert on the Spring Boot framework.
Your role is to provide accurate, helpful information based ONLY on the provided context, which is an excerpt from the Spring Boot tutorial.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context.
2. If the context doesn't contain relevant information, say "I don't have information about this in the available documents"
3. Be precise and factual - do not speculate or add information not in the context.
4. Keep responses clear and concise.

Context: {context}

Query: {query}

Response:"""
        
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            logger.info(f"Loading LLM model: {self.config.LLM_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LLM_MODEL,
                padding_side='left'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.use_pipeline = False
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM model, falling back to pipeline: {e}")
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            self.use_pipeline = True

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the query"""
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        # Translate query to English if needed for better retrieval
        english_query = self.language_detector.translate_to_english(query)
        
        results = self.vector_store.search_similar(english_query, top_k)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results
            if result['similarity_score'] > 0.5  # Adjust threshold as needed
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} relevant chunks for query")
        return filtered_results

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context"""
        try:
            context_text = "\n\n".join([
                f"Source: {chunk['metadata']['file_name']}\n{chunk['text']}"
                for chunk in context_chunks
            ])

            prompt = self.system_prompt.format(
                context=context_text[:2000],
                query=query
            )

            if self.use_pipeline:
                response = self.text_generator(
                    prompt,
                    max_length=len(prompt.split()) + 100,
                    num_return_sequences=1,
                    temperature=self.config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.text_generator.tokenizer.eos_token_id
                )[0]['generated_text']
                generated_text = response[len(prompt):].strip()
            else:
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
        
        context_chunks = self.retrieve_context(query)
        
        if not context_chunks:
            response_data = {
                'response': "I don't have information about this topic in the available documents. Please try a different query.",
                'sources': [],
                'context_used': 0,
                'query': query
            }
        else:
            response_data = self.generate_response(query, context_chunks)
        
        return {
            'response': {'text': response_data['response'], 'full_response': response_data['response'], 'citations': ""},
            'session_id': "temp",
            'sources': response_data['sources'],
            'bilingual_response': {},
            'metadata': {}
        }
