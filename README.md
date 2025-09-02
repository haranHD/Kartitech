# UPSIDA AI-Powered Public Chatbot (Module B)
A Retrieval-Augmented Generation (RAG) based chatbot system for Uttar Pradesh State Industrial Development Authority (UPSIDA) public queries.

## Project Overview
This system implements Module B from the Technical Specification Document - an AI-powered chatbot that provides accurate, source-verified answers to public queries based on official UPSIDA documents.

### Key Features
* **RAG Pipeline**: Retrieval-Augmented Generation for factual accuracy
* **Bilingual Support**: English + Hindi query processing
* **Source Citations**: Every response includes document references
* **OCR Support**: Process scanned PDF documents
* **Modular Architecture**: Clean separation of concerns
* **Feedback Mechanism**: User satisfaction tracking

## Architecture
Query Input → Language Detection → Vector Search Context Retrieval → LLM Generation → Bilingual Response + Citations

### Core Components
- **Data Ingestion**: PDF/DOCX processing with OCR
- **Vector Store**: ChromaDB for semantic search
- **RAG Pipeline**: Context retrieval + LLM generation
- **Language Support**: Hindi/English NLU
- **Chatbot Engine**: Main orchestration layer

## Quick Start
### 1. Installation
```bash
# Clone the project
git clone <repository-url>
cd upsida_chatbot
# Install dependencies
pip install -r requirements.txt
# Install additional language support
python -m spacy download en_core_web_sm