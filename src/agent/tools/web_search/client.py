from typing import Any
from src.config.settings import settings

from requests import post

from src.agent.tools.web_search.schema import WebSource

def search_datalogic_knowledge(question: str) -> list[WebSource]:
    return exa_search(question)


def exa_search(query: str) -> list[WebSource]:
    response = post(
        settings.EXA_SEARCH_ENDPOINT,
        headers={
            "x-api-key": settings.EXA_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "numResults": settings.SEARCH_RESULT_LIMIT,
            "contents": {
                "highlights": {
                    "query": query,
                    "highlightsPerUrl": settings.HIGHLIGHTS_PER_SOURCE,
                    "numSentences": settings.HIGHLIGHT_SENTENCES,
                },
            },
        },
        timeout=settings.EXA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    results = []
    for item in _extract_exa_result_items(payload):
        url = item.get("url")
        if not url:
            continue

        content = _truncate_chars(
            _extract_highlight_text(item),
            settings.MAX_PAGE_CHARS_PER_SOURCE,
        )
        if not content:
            continue

        results.append(
            WebSource(
                title=item.get("title") or url,
                url=url,
                content=content,
            )
        )

        if len(results) >= settings.SEARCH_RESULT_LIMIT:
            break

    return results


def _build_user_prompt(question: str, sources: list[WebSource]) -> str:
    source_blocks = []
    for index, source in enumerate(sources, start=1):
        source_blocks.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Title: {source.title}",
                    f"URL: {source.url}",
                    f"Highlights: {source.content}",
                ]
            )
        )

    return "\n\n".join(
        [
            f"Question: {question}",
            "Allowed source excerpts:",
            *source_blocks,
        ]
    )


def _extract_highlight_text(item: dict[str, Any]) -> str:
    highlights = item.get("highlights")
    if isinstance(highlights, str):
        return highlights.strip()

    if isinstance(highlights, list):
        highlight_values = [
            _extract_highlight_value(value)
            for value in highlights
        ]
        return "\n".join(value for value in highlight_values if value).strip()

    return ""


def _extract_highlight_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        for key in ("text", "content", "highlight"):
            highlight = value.get(key)
            if isinstance(highlight, str) and highlight.strip():
                return highlight.strip()

    return ""


def _truncate_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{truncated}..."


def _extract_exa_result_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []