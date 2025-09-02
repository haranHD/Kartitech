import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('upsida_chatbot.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_document_id(file_path: str) -> str:
    """Generate unique document ID based on file content"""
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but preserve Indian language characters
    text = re.sub(r'[^\w\s\u0900-\u097F.,!?;:()\-]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append({
            'text': chunk_text,
            'start_index': i,
            'end_index': min(i + chunk_size, len(words)),
            'word_count': len(chunk_words)
        })
        if i + chunk_size >= len(words):
            break
    return chunks

def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from document file"""
    stat = file_path.stat()
    return {
        'file_name': file_path.name,
        'file_path': str(file_path),
        'file_size': stat.st_size,
        'created_date': stat.st_ctime,
        'modified_date': stat.st_mtime,
        'file_type': file_path.suffix.lower()
    }