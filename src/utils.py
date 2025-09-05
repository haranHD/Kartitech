import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

def setup_logging(log_file='chatbot.log', log_level: str = "INFO") -> logging.Logger:
    """Sets up a centralized logger for the application."""
    logger = logging.getLogger()
    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w') # Overwrite log on each run
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    return logger

def generate_document_id(file_path: str) -> str:
    """Creates a unique ID for a document based on its file path."""
    return hashlib.md5(file_path.encode('utf-8')).hexdigest()

def clean_text(text: str) -> str:
    """Performs basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Correctly splits text into smaller, overlapping chunks."""
    if not text: return []
    words = text.split()
    if not words: return []

    chunks = []
    current_pos = 0
    chunk_index = 0
    while current_pos < len(words):
        start_index = current_pos
        end_index = current_pos + chunk_size
        chunk_words = words[start_index:end_index]
        
        chunks.append({
            'text': ' '.join(chunk_words),
            'start_word_index': start_index,
            'end_word_index': start_index + len(chunk_words),
            'chunk_index': chunk_index
        })
        
        current_pos += chunk_size - chunk_overlap
        chunk_index += 1
        # Safety break to prevent infinite loops if chunk_size is less than or equal to overlap
        if current_pos <= start_index: break
    return chunks

def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """Extracts metadata from a file."""
    try:
        stat = file_path.stat()
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': stat.st_size,
            'created_date': stat.st_ctime,
            'modified_date': stat.st_mtime,
            'file_type': file_path.suffix.lower()
        }
    except FileNotFoundError:
        logging.error(f"Could not find file to extract metadata: {file_path}")
        return {}

