from collections.abc import Callable

import pytest

from l3s_agent.literature.discovery import normalize_title
from l3s_agent.literature.models import (
    AuthorRecord,
    LiteratureCandidate,
    OpenAccessInfo,
    QueryMatch,
)


@pytest.fixture
def candidate_factory() -> Callable[..., LiteratureCandidate]:
    def make_candidate(**overrides: object) -> LiteratureCandidate:
        title = str(overrides.pop("title", "Weather Effects on Solar Power Generation"))
        values = {
            "openalex_id": "W1",
            "openalex_url": "https://openalex.org/W1",
            "title": title,
            "normalized_title": normalize_title(title),
            "authors": (AuthorRecord("Ada Example", "A1"),),
            "year": 2022,
            "doi": "10.1000/example",
            "work_type": "article",
            "language": "en",
            "source_url": "https://publisher.example/work",
            "open_access": OpenAccessInfo(
                is_oa=True,
                status="green",
                landing_page_url="https://repository.example/work",
                pdf_urls=("https://repository.example/work.pdf",),
            ),
            "cited_by_count": 10,
            "abstract": "Solar generation changes with meteorological conditions and cloud cover.",
            "topics": ("Renewable Energy", "Weather and Climate"),
            "keywords": ("solar irradiance",),
            "query_matches": (QueryMatch("weather solar power generation", 1, 10.0),),
            "is_retracted": False,
        }
        values.update(overrides)
        return LiteratureCandidate(**values)  # type: ignore[arg-type]

    return make_candidate
