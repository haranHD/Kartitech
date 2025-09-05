from googletrans import Translator
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Handles language detection and translation."""
    def __init__(self):
        self.translator = Translator()

    def detect_language(self, text: str) -> str:
        """Detect if the text is primarily Hindi or English."""
        try:
            # A simple heuristic: if a significant number of characters are Hindi, assume Hindi.
            if any('\u0900' <= char <= '\u097F' for char in text):
                return "hindi"
            return "english"
        except Exception:
            return "english" # Default fallback

    def translate_to_english(self, text: str) -> str:
        """Translate text to English if it's not already."""
        if self.detect_language(text) == "hindi":
            try:
                return self.translator.translate(text, src='hi', dest='en').text
            except Exception as e:
                logger.warning(f"Translation to English failed: {e}")
        return text

    def get_bilingual_response(self, english_response: str, original_lang: str) -> dict:
        """Provide response in both English and Hindi."""
        if original_lang == "hindi":
            try:
                hindi_response = self.translator.translate(english_response, src='en', dest='hi').text
                return {"english": english_response, "hindi": hindi_response, "primary": hindi_response}
            except Exception as e:
                logger.warning(f"Translation to Hindi failed: {e}")
                return {"english": english_response, "hindi": english_response, "primary": english_response}
        else:
            return {"english": english_response, "hindi": "Translation to Hindi is available upon request.", "primary": english_response}
