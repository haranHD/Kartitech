import PyPDF2
import docx
import pandas as pd
import pytesseract
from PIL import Image
import io
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

from .utils import generate_document_id, clean_text, chunk_text, extract_metadata

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion, OCR, and text extraction."""
    def __init__(self, config):
        self.config = config
        self.supported_formats = {'.pdf', '.docx', '.doc', '.txt'}
        if convert_from_path is None:
            logger.warning("pdf2image library not found. OCR for PDF files will be disabled.")

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files, with OCR as a fallback."""
        full_text = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        full_text.append(page_text)
            
            # If no text was extracted directly, attempt OCR
            if not full_text and convert_from_path:
                logger.warning(f"No text extracted directly from '{file_path.name}'. Attempting OCR.")
                images = convert_from_path(file_path)
                for image in images:
                    ocr_text = pytesseract.image_to_string(image, lang=self.config.OCR_LANGUAGE)
                    if ocr_text and ocr_text.strip():
                        full_text.append(ocr_text)

        except PyPDF2.errors.PdfReadError as e:
            logger.error(f"Failed to read PDF '{file_path.name}': {e}. The file is likely corrupted.")
            return ""
        except Exception as e:
            logger.error(f"An error occurred while processing PDF '{file_path.name}': {e}")
            return ""

        return clean_text(" ".join(full_text))

    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        try:
            doc = docx.Document(file_path)
            return clean_text("\n".join(p.text for p in doc.paragraphs))
        except Exception as e:
            logger.error(f"Error extracting text from DOCX '{file_path}': {e}")
            return ""

    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT files."""
        try:
            return clean_text(file_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Error reading TXT file '{file_path}': {e}")
            return ""

    def process_single_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process one document, extract text, chunk it, and return structured data."""
        if not file_path.is_file() or file_path.suffix.lower() not in self.supported_formats:
            return None

        logger.info(f"Processing document: '{file_path.name}'")
        text_extractors = {'.pdf': self.extract_text_from_pdf, '.docx': self.extract_text_from_docx, '.txt': self.extract_text_from_txt}
        extractor = text_extractors.get(file_path.suffix.lower())
        
        text = extractor(file_path) if extractor else ""

        if not text:
            logger.warning(f"No text was extracted from '{file_path.name}'.")
            return None

        chunks = chunk_text(text, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
        doc_id = generate_document_id(str(file_path))
        
        processed_data = {
            'document_id': doc_id, 'metadata': extract_metadata(file_path),
            'full_text': text, 'chunks': chunks, 'chunk_count': len(chunks),
            'processed_date': pd.Timestamp.now().isoformat()
        }
        self.save_processed_document(processed_data)
        return processed_data

    def save_processed_document(self, data: Dict[str, Any]):
        """Save the structured data of a processed document to a JSON file."""
        doc_id = data.get('document_id')
        if not doc_id: return
        
        output_path = self.config.PROCESSED_DIR / f"{doc_id}.json"
        logger.info(f"Saving processed data to '{output_path}'.")
        try:
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path): return str(obj)
                    return super().default(obj)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, cls=CustomEncoder, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save processed file for doc_id '{doc_id}': {e}")

    def process_document_directory(self, directory: Path, target_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a specific file or all supported files in a directory."""
        processed_docs = []
        files_to_process = [directory / target_file] if target_file else list(directory.iterdir())
        
        for file_path in files_to_process:
            if doc_data := self.process_single_document(file_path):
                processed_docs.append(doc_data)
        
        logger.info(f"Finished processing. Found and processed {len(processed_docs)} documents.")
        return processed_docs
