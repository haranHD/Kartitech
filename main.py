import sys
import os
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from config.config import Config as UPSIDAConfig
from src.chatbot_engine import ChatbotEngine as UPSIDAChatbotEngine
from config.config_springboot import SpringBootConfig
from src.rag_pipeline_springboot import SpringBootRAGPipeline
from src.utils import setup_logging
from src.data_ingestion import DocumentProcessor
from src.vector_store import VectorStore

def create_sample_documents():
    """Create sample UPSIDA documents for demonstration"""
    UPSIDAConfig.create_directories()
    # Sample UPSIDA documents content
    sample_docs = {
        "UPSIDA_Land_Allotment_Policy_2024.txt": """UPSIDA Land Allotment Policy 2024
========================================================================================
Section 1: Eligibility Criteria
1.1 Industrial units seeking land allotment must have a minimum investment of Rs. 25 lakhs.
1.2 Priority will be given to MSME units, export-oriented industries, and IT/ITES companies.
1.3 Applicants must have a valid industrial license and environmental clearance.
Section 2: Application Process
2.1 Applications must be submitted online through the UPSIDA portal.
2.2 Required documents include project report, financial statements, and NOCs.
2.3 Processing time is 30 working days from complete application submission.
Section 3: Land Rates
3.1 Industrial land rates vary by location and connectivity.
3.2 Kanpur Industrial Area: Rs. 1,200 per sq meter
3.3 Lucknow Industrial Area: Rs. 1,500 per sq meter
3.4 Agra Industrial Area: Rs. 1,000 per sq meter
Section 4: Payment Terms
4.1 25% advance payment required with application.
4.2 Remaining 75% payable within 90 days of allotment letter.
4.3 EMI options available for MSME units.
""",
        "UPSIDA_Investment_Incentives_2024.txt": """
UPSIDA Investment Incentive Scheme 2024
====
Section 1: Eligible Industries
1.1 Manufacturing units with investment above Rs. 1 crore
1.2 IT/Software companies with minimum 50 employees
1.3 Food processing and agro-based industries
1.4 Textile and garment manufacturing units
Section 2: Incentive Structure
2.1 Capital Subsidy: Up to 25% of fixed capital investment (max Rs. 2 crores)
2.2 Interest Subsidy: 5% interest subsidy for 5 years on institutional loans
2.3 SGST Reimbursement: 100% for first 5 years, 50% for next 2 years
2.4 Stamp Duty Exemption: 100% exemption on land purchase/lease documents
Section 3: Employment Incentives
3.1 Rs. 25,000 per job created for local employment
3.2 Additional Rs. 50,000 for employing persons with disabilities
3.3 Training cost reimbursement up to Rs. 10,000 per employee
Section 4: Application Process
4.1 Apply within 6 months of commercial production
4.2 Submit audited financial statements and employment records
4.3 Verification by UPSIDA technical team required
""",
        "UPSIDA_Tender_Guidelines_2024.txt": """
UPSIDA Tender Guidelines 2024
===
Section 1: Tender Eligibility
1.1 Contractors must have minimum 3 years experience in similar projects
1.2 Valid contractor license and GST registration mandatory
1.3 Financial turnover criteria: Minimum 2x of estimated tender value
Section 2: Document Requirements
2.1 Technical bid must include detailed project methodology
2.2 Financial bid submitted in sealed envelope or online portal
2.3 Performance bank guarantee: 5% of contract value
2.4 Validity period: 90 days from tender submission
Section 3: Evaluation Criteria
3.1 Technical evaluation (70% weightage): Quality of methodology, team expertise
3.2 Financial evaluation (30% weightage): L1 basis with quality considerations
3.3 Past performance and client references considered
Section 4: Contract Terms
4.1 Payment terms: 80% against work completion, 20% after defect liability period
4.2 Penalty clause: 0.5% per week delay up to maximum 10%
4.3 Force majeure conditions as per standard government contracts
"""
    }
    # Write sample documents
    for filename, content in sample_docs.items():
        file_path = UPSIDAConfig.DOCUMENTS_DIR / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"Created {len(sample_docs)} sample documents in {UPSIDAConfig.DOCUMENTS_DIR}")

def run_demo(rag_model: str):
    """Run interactive demo of the chatbot"""
    logger = setup_logging()
    print("\n" + "=" * 50)

    if rag_model == "springboot":
        config = SpringBootConfig()
        print("Spring Boot RAG Chatbot Demo")
        chatbot = SpringBootChatbotEngine(config)
    else:
        config = UPSIDAConfig()
        print("UPSIDA AI Chatbot Demo")
        chatbot = UPSIDAChatbotEngine(config)
    
    print("=" * 50)

    # Initialize system
    config.create_directories()

    # Ingest documents
    print("Processing documents...")
    ingestion_result = chatbot.ingest_documents()
    print(f"Ingestion Status: {ingestion_result['message']}")

    # Interactive mode
    print("\nInteractive Mode (type 'exit' to quit):")
    session_id = "interactive_session"
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using the AI Chatbot!")
                break
            if not user_input:
                continue
            response = chatbot.chat(user_input, session_id=session_id)
            print(f"\nAI Bot: {response['response']['text']}")
            if response.get('sources'):
                print("[Sources]:")
                for source in response['sources'][:2]:
                    print(f"  - {source['document']} (Confidence: {source['similarity_score']:.2f})")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

class SpringBootChatbotEngine:
    def __init__(self, config):
        self.config = config
        self.conversation_history = {}
        self.feedback_data = []
        self.vector_store = VectorStore(config)
        self.rag_pipeline = SpringBootRAGPipeline(config, self.vector_store)
        self.document_processor = DocumentProcessor(config)

    def ingest_documents(self, force_reprocess: bool = False) -> Dict[str, Any]:
        return self.document_processor.process_document_directory(
            self.config.DOCUMENTS_DIR,
            target_file_name="spring_boot_tutorial.pdf"
        )
    
    def chat(self, user_query: str, session_id: str) -> Dict[str, Any]:
        return self.rag_pipeline.process_query(user_query)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Chatbot")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo with UPSIDA chatbot")
    parser.add_argument("--springboot", action="store_true", help="Run interactive demo with Spring Boot RAG model")
    parser.add_argument("--ingest", action="store_true", help="Process documents for UPSIDA chatbot")
    parser.add_argument("--ingest-springboot", action="store_true", help="Process documents for Spring Boot RAG model")
    
    args = parser.parse_args()

    if args.demo:
        run_demo("upsida")
    elif args.springboot:
        run_demo("springboot")
    elif args.ingest:
        from ingest_upsida import ingest_documents_upsida
        result = ingest_documents_upsida(force_reprocess=True)
        print(f"Ingestion result: {result}")
    elif args.ingest_springboot:
        from ingest_springboot import ingest_documents_springboot
        result = ingest_documents_springboot(force_reprocess=True)
        print(f"Ingestion result: {result}")
    else:
        print("Please use a command-line argument: --demo, --springboot, --ingest, or --ingest-springboot.")
