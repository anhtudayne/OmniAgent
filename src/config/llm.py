import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
dotenv_path = PROJECT_ROOT / "src" / "config" / ".env"
load_dotenv(dotenv_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Config:
    GEMINI_API = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = 'gemini-2.5-flash' 


class Gemini:
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=self.config.GEMINI_API)
        self.llm = genai.GenerativeModel(self.config.GEMINI_MODEL)

    def invoke(self, prompt: str) -> str:
        try:
            response = self.llm.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=16384,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f'Gemini ERROR: {str(e)}')
            return ""
