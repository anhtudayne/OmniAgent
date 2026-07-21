"""Global configuration settings for AI Assistant."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Manage configuration and environment variables."""

    # LLM Settings
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # API Server Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Tool Execution Settings
    TOOL_TIMEOUT_SECONDS: float = float(os.getenv("TOOL_TIMEOUT_SECONDS", "30.0"))

    EXA_API_KEY = os.getenv("EXA_API_KEY")
    EXA_SEARCH_ENDPOINT = os.getenv(
        "EXA_SEARCH_ENDPOINT",
        "https://api.exa.ai/search",
    )
    SEARCH_RESULT_LIMIT = int(os.getenv("SEARCH_RESULT_LIMIT", "5"))
    EXA_TIMEOUT_SECONDS = float(os.getenv("EXA_TIMEOUT_SECONDS", "30"))
    MAX_PAGE_CHARS_PER_SOURCE = int(os.getenv("MAX_PAGE_CHARS_PER_SOURCE", "5000"))
    HIGHLIGHTS_PER_SOURCE = int(os.getenv("HIGHLIGHTS_PER_SOURCE", "5"))
    HIGHLIGHT_SENTENCES = int(os.getenv("HIGHLIGHT_SENTENCES", "3"))

    DATALOGIC_ALLOWED_DOMAINS = (
    )



settings = Settings()
