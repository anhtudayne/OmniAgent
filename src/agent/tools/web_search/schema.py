from dataclasses import dataclass, field


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str
    content: str = ""


@dataclass(frozen=True)
class WebSearchAnswer:
    answer: str
    sources: list[WebSource] = field(default_factory=list)