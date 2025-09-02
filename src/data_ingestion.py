import PyPDF2
import docx
import pandas as pd
import pytesseract
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from .utils import generate_document_id, clean_text, chunk_text, extract_metadata

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion, OCR, and text extraction"""
    def __init__(self, config):
        self.config = config
        self.supported_formats = {'.pdf', '.docx', '.doc', '.txt'}

    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    # If text extraction fails, try OCR
                    if not page_text.strip():
                        logger.info(f"OCR processing page {page_num + 1} of {file_path.name}")
                        page_text = self._ocr_pdf_page(page)
                    text += f"\n[Page {page_num + 1}]\n{page_text}\n"
            return clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    def _ocr_pdf_page(self, page) -> str:
        """Perform OCR on PDF page"""
        try:
            # This is a simplified OCR approach
            # In production, you'd convert PDF page to image first
            return pytesseract.image_to_string(
                page,
                lang=self.config.OCR_LANGUAGE,
                config='--psm 6'
            )
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                text += "\n"
            return clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            return ""

    def extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return clean_text(file.read())
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return clean_text(file.read())
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            return ""

    def process_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single document and return structured data"""
        if file_path.suffix.lower() not in self.supported_formats:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return None
        logger.info(f"Processing document: {file_path.name}")
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            logger.error(f"Unsupported format: {file_path.suffix}")
            return None
        if not text.strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return None
        # Generate chunks
        chunks = chunk_text(text, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
        # Extract metadata
        metadata = extract_metadata(file_path)
        # Generate document ID
        doc_id = generate_document_id(str(file_path))
        return {
            'document_id': doc_id,
            'metadata': metadata,
            'full_text': text,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'processed_date': pd.Timestamp.now().isoformat()
        }

    def process_document_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """Process all documents in a directory"""
        processed_docs = []
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                doc_data = self.process_document(file_path)
                if doc_data:
                    processed_docs.append(doc_data)
        logger.info(f"Processed {len(processed_docs)} documents from {directory_path}")
        return processed_docs