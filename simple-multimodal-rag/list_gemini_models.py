"""List Gemini models available to the configured API key.

Usage:
    python list_gemini_models.py

The script prints all models returned by the Gemini API and highlights the
ones that support text generation and embedding.
"""

from __future__ import annotations

import os
from typing import Iterable

import google.generativeai as genai

import config


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY") or config.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in the environment or in config.py."
        )
    return api_key


def _format_methods(methods: Iterable[str] | None) -> str:
    if not methods:
        return "-"
    return ", ".join(methods)


def main() -> None:
    genai.configure(api_key=_get_api_key())

    models = list(genai.list_models())
    if not models:
        print("No models returned by the Gemini API.")
        return

    print("Available Gemini models:\n")
    for model in models:
        methods = getattr(model, "supported_generation_methods", None)
        methods_text = _format_methods(methods)
        print(f"- {model.name}")
        print(f"  display_name: {getattr(model, 'display_name', '-')}")
        print(f"  description: {getattr(model, 'description', '-')}")
        print(f"  methods: {methods_text}")
        print()

    print("Supported for generateContent:")
    for model in models:
        methods = getattr(model, "supported_generation_methods", None) or []
        if "generateContent" in methods:
            print(f"- {model.name}")

    print("\nSupported for embedContent:")
    for model in models:
        methods = getattr(model, "supported_generation_methods", None) or []
        if "embedContent" in methods:
            print(f"- {model.name}")


if __name__ == "__main__":
    main()