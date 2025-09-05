import argparse
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from config.config import Config
from src.chatbot_engine import ChatbotEngine
from src.utils import setup_logging

def run_interactive_chatbot(engine: ChatbotEngine):
    """Run an interactive command-line session with the specified chatbot engine."""
    print("\n" + "="*50)
    print(f" INTERACTIVE CHATBOT ({engine.collection_name})")
    print(" Type 'exit' to quit.")
    print("="*50)
    
    session_id = "interactive_session"
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            if not user_input:
                continue

            response = engine.chat(user_input, session_id=session_id)
            print(f"\nAI Bot: {response['response']}")
            
            if response.get('sources'):
                print("\n[Sources]:")
                for source in response['sources']:
                    print(f"  - {source['document']} (Score: {source['score']:.2f})")
        
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


def main():
    """Main function to handle command-line arguments."""
    # Setup
    config = Config()
    config.create_directories()
    logger = setup_logging()

    # Engine Setup
    upsida_engine = ChatbotEngine(config, config.UPSIDA_COLLECTION_NAME)
    springboot_engine = ChatbotEngine(config, config.SPRINGBOOT_COLLECTION_NAME)

    # Command-line parser
    parser = argparse.ArgumentParser(description="Document AI Chatbot")
    parser.add_argument("--ingest-upsida", action="store_true", help="Ingest all UPSIDA documents.")
    parser.add_argument("--ingest-springboot", action="store_true", help="Ingest the Spring Boot tutorial document.")
    parser.add_argument("--chat-upsida", action="store_true", help="Start interactive chat with the UPSIDA bot.")
    parser.add_argument("--chat-springboot", action="store_true", help="Start interactive chat with the Spring Boot bot.")
    
    args = parser.parse_args()

    if args.ingest_upsida:
        logger.info("Starting ingestion for all UPSIDA documents...")
        result = upsida_engine.ingest_documents(force_reprocess=True)
        logger.info(f"Ingestion result: {result}")

    elif args.ingest_springboot:
        logger.info("Starting ingestion for the Spring Boot tutorial...")
        result = springboot_engine.ingest_documents(force_reprocess=True, target_file="spring_boot_tutorial.pdf")
        logger.info(f"Ingestion result: {result}")

    elif args.chat_upsida:
        run_interactive_chatbot(upsida_engine)

    elif args.chat_springboot:
        run_interactive_chatbot(springboot_engine)

    else:
        print("No action specified. Please choose an option:")
        print("  --ingest-upsida      : Process and load all UPSIDA documents.")
        print("  --ingest-springboot  : Process and load the Spring Boot tutorial.")
        print("  --chat-upsida        : Start a chat session about UPSIDA.")
        print("  --chat-springboot    : Start a chat session about Spring Boot.")

if __name__ == "__main__":
    main()
