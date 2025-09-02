import re
from typing import Tuple, Dict
from googletrans import Translator
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Handles language detection and translation for bilingual support"""
    def __init__(self):
        self.translator = Translator()
        # Hindi Unicode range patterns
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]')
        self.english_pattern = re.compile(r'[a-zA-Z]')

    def detect_language(self, text: str) -> str:
        """Detect primary language of input text"""
        hindi_chars = len(self.hindi_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        if hindi_chars > english_chars:
            return "hindi"
        elif english_chars > 0:
            return "english"
        else:
            # Fallback to Google Translate detection
            try:
                detected = self.translator.detect(text)
                return "hindi" if detected.lang == "hi" else "english"
            except Exception as e:
                logger.warning(f"Google Translate detection failed: {e}")
                return "english" # Default fallback

    def translate_to_english(self, text: str) -> str:
        """Translate Hindi text to English for processing"""
        try:
            if self.detect_language(text) == "hindi":
                result = self.translator.translate(text, src='hi', dest='en')
                return result.text
            return text
        except Exception as e:
            logger.warning(f"Translation to English failed: {e}")
            return text

    def translate_to_hindi(self, text: str) -> str:
        """Translate English response to Hindi"""
        try:
            result = self.translator.translate(text, src='en', dest='hi')
            return result.text
        except Exception as e:
            logging.warning(f"Translation to Hindi failed: {e}")
            return text

    def get_bilingual_response(self, english_response: str,
                                original_language: str) -> Dict[str, str]:
        """Return response in both languages"""
        if original_language == "hindi":
            hindi_response = self.translate_to_hindi(english_response)
            return {
                "english": english_response,
                "hindi": hindi_response,
                "primary": hindi_response
            }
        else:
            return {
                "english": english_response,
                "hindi": self.translate_to_hindi(english_response),
                "primary": english_response
            }