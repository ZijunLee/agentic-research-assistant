"""Small OpenAlex Works API client with deterministic provenance capture."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from .discovery import normalize_doi, normalize_title
from .models import (
    AuthorRecord,
    LiteratureCandidate,
    OpenAccessInfo,
    OpenAlexRequestRecord,
    QueryMatch,
)


WORK_FIELDS = (
    "id,doi,title,display_name,publication_year,publication_date,type,language,"
    "authorships,primary_location,best_oa_location,locations,open_access,"
    "cited_by_count,primary_topic,topics,keywords,abstract_inverted_index,"
    "is_retracted,relevance_score,has_content,content_urls"
)


class OpenAlexError(RuntimeError):
    """Credential-safe OpenAlex failure with useful structured context."""

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        status_code: int | None = None,
        transport_error_type: str | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.status_code = status_code
        self.transport_error_type = transport_error_type
        self.attempts = attempts

    def to_record(self) -> dict[str, object]:
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "query": self.query,
            "status_code": self.status_code,
            "transport_error_type": self.transport_error_type,
            "attempts": self.attempts,
        }


def reconstruct_abstract(inverted_index: object) -> str | None:
    if not isinstance(inverted_index, Mapping) or not inverted_index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, Sequence):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                positioned.append((position, word))
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def _author_records(authorships: object) -> tuple[AuthorRecord, ...]:
    if not isinstance(authorships, Sequence):
        return ()
    authors: list[AuthorRecord] = []
    for authorship in authorships:
        if not isinstance(authorship, Mapping):
            continue
        author = authorship.get("author")
        if not isinstance(author, Mapping):
            continue
        name = str(author.get("display_name") or "").strip()
        if not name:
            continue
        authors.append(
            AuthorRecord(
                display_name=name,
                openalex_id=str(author["id"]) if author.get("id") else None,
                orcid=str(author["orcid"]) if author.get("orcid") else None,
            )
        )
    return tuple(authors)


def _labels(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence):
        return ()
    labels: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            label = value.get("display_name")
            if label:
                labels.append(str(label))
    return tuple(dict.fromkeys(labels))


def _oa_info(work: Mapping[str, Any]) -> OpenAccessInfo:
    open_access = work.get("open_access")
    oa = open_access if isinstance(open_access, Mapping) else {}
    best = work.get("best_oa_location")
    primary = work.get("primary_location")
    locations: list[Mapping[str, Any]] = []
    if isinstance(best, Mapping):
        locations.append(best)
    if isinstance(primary, Mapping) and primary.get("is_oa") is True:
        locations.append(primary)
    raw_locations = work.get("locations")
    if isinstance(raw_locations, Sequence):
        locations.extend(
            item
            for item in raw_locations
            if isinstance(item, Mapping) and item.get("is_oa") is True
        )

    pdf_urls: list[str] = []
    landing_page_url: str | None = None
    license_name: str | None = None
    version: str | None = None
    for location in locations:
        if landing_page_url is None and location.get("landing_page_url"):
            landing_page_url = str(location["landing_page_url"])
        if location.get("pdf_url"):
            url = str(location["pdf_url"])
            if url not in pdf_urls:
                pdf_urls.append(url)
        if license_name is None and location.get("license"):
            license_name = str(location["license"])
        if version is None and location.get("version"):
            version = str(location["version"])

    has_content = work.get("has_content")
    has_content_pdf = bool(
        isinstance(has_content, Mapping) and has_content.get("pdf") is True
    )
    content_urls = work.get("content_urls")
    content_pdf_url = None
    if has_content_pdf and isinstance(content_urls, Mapping) and content_urls.get("pdf"):
        content_pdf_url = str(content_urls["pdf"])
    if has_content_pdf and not content_pdf_url:
        work_id = str(work.get("id") or "").rsplit("/", 1)[-1]
        if work_id:
            content_pdf_url = f"https://content.openalex.org/works/{work_id}.pdf"

    return OpenAccessInfo(
        is_oa=bool(oa.get("is_oa")),
        status=str(oa["oa_status"]) if oa.get("oa_status") else None,
        landing_page_url=landing_page_url,
        pdf_urls=tuple(pdf_urls),
        license=license_name,
        version=version,
        has_content_pdf=has_content_pdf,
        content_pdf_url=content_pdf_url,
    )


def parse_work(work: Mapping[str, Any], *, query: str, rank: int) -> LiteratureCandidate:
    openalex_url = str(work.get("id") or "").strip()
    title = str(work.get("title") or work.get("display_name") or "").strip()
    if not openalex_url:
        raise ValueError("OpenAlex work is missing id")
    primary = work.get("primary_location")
    source_url = None
    if isinstance(primary, Mapping) and primary.get("landing_page_url"):
        source_url = str(primary["landing_page_url"])
    primary_topic = work.get("primary_topic")
    topics = list(_labels(work.get("topics")))
    if isinstance(primary_topic, Mapping) and primary_topic.get("display_name"):
        topics.insert(0, str(primary_topic["display_name"]))
    topics = list(dict.fromkeys(topics))
    relevance = work.get("relevance_score")
    relevance_score = float(relevance) if isinstance(relevance, (int, float)) else None
    return LiteratureCandidate(
        openalex_id=openalex_url.rsplit("/", 1)[-1],
        openalex_url=openalex_url,
        title=title,
        normalized_title=normalize_title(title),
        authors=_author_records(work.get("authorships")),
        year=int(work["publication_year"]) if work.get("publication_year") else None,
        doi=normalize_doi(str(work["doi"])) if work.get("doi") else None,
        work_type=str(work["type"]) if work.get("type") else None,
        language=str(work["language"]).lower() if work.get("language") else None,
        source_url=source_url,
        open_access=_oa_info(work),
        cited_by_count=int(work.get("cited_by_count") or 0),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        topics=tuple(topics),
        keywords=_labels(work.get("keywords")),
        query_matches=(QueryMatch(query=query, rank=rank, relevance_score=relevance_score),),
        is_retracted=bool(work.get("is_retracted")),
        raw_metadata={"publication_date": work.get("publication_date")},
    )


class OpenAlexClient:
    """Search OpenAlex while retaining request/response provenance."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        timeout_seconds: float,
        max_attempts: int,
        cache_dir: Path | None = None,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OPENALEX_API_KEY is required")
        self._api_url = api_url
        self._max_attempts = max_attempts
        self._cache_dir = cache_dir
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "l3s-scientific-agent/0.1"},
        )
        self._authorization = f"Bearer {api_key}"
        self._owns_client = client is None
        self._sleeper = sleeper
        self.request_records: list[OpenAlexRequestRecord] = []

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAlexClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(self, *, query: str, page: int, per_page: int) -> list[LiteratureCandidate]:
        params = {
            "search": query,
            "page": page,
            "per_page": per_page,
            "select": WORK_FIELDS,
        }
        response = self._request(params, query=query)
        try:
            payload = response.json()
        except ValueError:
            raise OpenAlexError(
                f"OpenAlex returned invalid JSON for query {query!r}", query=query
            ) from None
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise OpenAlexError(
                f"OpenAlex response schema invalid for query {query!r}", query=query
            )
        results = payload["results"]
        candidates: list[LiteratureCandidate] = []
        parse_failures: list[str] = []
        for rank, work in enumerate(results, start=1 + ((page - 1) * per_page)):
            if not isinstance(work, Mapping):
                parse_failures.append(f"result {rank}: result is not an object")
                continue
            try:
                candidates.append(parse_work(work, query=query, rank=rank))
            except (TypeError, ValueError) as exc:
                parse_failures.append(f"result {rank}: {exc}")
                continue

        content = response.content
        cache_path = self._cache_response(content, len(self.request_records) + 1)
        self.request_records.append(
            OpenAlexRequestRecord(
                query=query,
                page=page,
                per_page=per_page,
                result_count=len(candidates),
                response_sha256=hashlib.sha256(content).hexdigest(),
                cache_path=cache_path,
                parse_failures=tuple(parse_failures),
            )
        )
        return candidates

    def _request(self, params: Mapping[str, object], *, query: str) -> httpx.Response:
        last_transport_error_type: str | None = None
        attempts = 0
        for attempt in range(1, self._max_attempts + 1):
            attempts = attempt
            try:
                response = self._client.get(
                    self._api_url,
                    params=params,
                    headers={"Authorization": self._authorization},
                )
                if response.status_code in {429} or response.status_code >= 500:
                    if attempt < self._max_attempts:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after else float(attempt)
                        self._sleeper(delay)
                        continue
                if response.is_error:
                    raise OpenAlexError(
                        f"OpenAlex request for query {query!r} failed with "
                        f"HTTP {response.status_code}",
                        query=query,
                        status_code=response.status_code,
                    ) from None
                return response
            except OpenAlexError:
                raise
            except httpx.RequestError as exc:
                last_transport_error_type = type(exc).__name__
                if attempt < self._max_attempts:
                    self._sleeper(float(attempt))
                    continue
                break
        raise OpenAlexError(
            f"OpenAlex request for query {query!r} failed after bounded retries",
            query=query,
            transport_error_type=last_transport_error_type,
            attempts=attempts,
        ) from None

    def _cache_response(self, content: bytes, sequence: int) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"response_{sequence:03d}.json"
        parsed = json.loads(content)
        path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
