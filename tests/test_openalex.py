import json
import traceback
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from l3s_agent.literature.openalex import OpenAlexClient, OpenAlexError, parse_work


FIXTURE = Path(__file__).parent / "fixtures" / "openalex_works.json"


def test_openalex_parses_provenance_and_oa_fields(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observed: dict[str, list[str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(parse_qs(request.url.query.decode()))
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        return httpx.Response(200, json=payload)

    client = OpenAlexClient(
        api_key="secret-test-key",
        api_url="https://api.openalex.org/works",
        timeout_seconds=1,
        max_attempts=1,
        cache_dir=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    candidates = client.search(query="weather solar", page=1, per_page=10)

    assert "api_key" not in observed
    assert observed["search"] == ["weather solar"]
    assert "sort" not in observed
    assert "has_content" in observed["select"][0].split(",")
    assert "content_urls" in observed["select"][0].split(",")
    assert candidates[0].doi == "10.1000/solar"
    assert candidates[0].authors[0].display_name == "Ada Example"
    assert candidates[0].open_access.pdf_url == "https://repository.example/solar.pdf"
    assert candidates[0].open_access.pdf_urls == (
        "https://repository.example/solar.pdf",
        "https://alternate.example/solar.pdf",
    )
    assert candidates[0].open_access.has_content_pdf is True
    assert (
        candidates[0].open_access.content_pdf_url
        == "https://content.openalex.org/works/W123.pdf"
    )
    assert candidates[0].abstract == "Solar generation varies with weather"
    assert client.request_records[0].response_sha256
    assert client.request_records[0].cache_path is not None
    cached = client.request_records[0].cache_path.read_text(encoding="utf-8")
    metadata = json.dumps(asdict(client.request_records[0]), default=str)
    assert "secret-test-key" not in cached
    assert "secret-test-key" not in metadata


def test_openalex_retries_429_without_leaking_key() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=payload)

    client = OpenAlexClient(
        api_key="secret-test-key",
        api_url="https://api.openalex.org/works",
        timeout_seconds=1,
        max_attempts=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=delays.append,
    )

    assert len(client.search(query="weather", page=1, per_page=10)) == 1
    assert calls == 2
    assert delays == [0.0]


def test_openalex_http_error_does_not_expose_key() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    client = OpenAlexClient(
        api_key="never-print-this-key",
        api_url="https://api.openalex.org/works",
        timeout_seconds=1,
        max_attempts=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(OpenAlexError) as error:
        client.search(query="weather", page=1, per_page=10)
    exception = error.value
    rendered_traceback = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    error_record = json.dumps(exception.to_record(), sort_keys=True)

    assert exception.status_code == 401
    assert exception.query == "weather"
    assert exception.__cause__ is None
    assert exception.__context__ is None
    assert "never-print-this-key" not in str(exception)
    assert "never-print-this-key" not in repr(exception)
    assert "never-print-this-key" not in rendered_traceback
    assert "never-print-this-key" not in error_record
    assert "never-print-this-key" not in repr(client.request_records)


def test_network_exception_traceback_does_not_expose_authorization_secret() -> None:
    secret = "network-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("controlled network failure", request=request)

    client = OpenAlexClient(
        api_key=secret,
        api_url="https://api.openalex.org/works",
        timeout_seconds=1,
        max_attempts=1,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(OpenAlexError) as error:
        client.search(query="wind climate", page=1, per_page=10)
    exception = error.value
    rendered_traceback = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    assert exception.__cause__ is None
    assert exception.__context__ is None
    assert secret not in str(exception)
    assert secret not in repr(exception)
    assert secret not in rendered_traceback
    assert secret not in json.dumps(exception.to_record(), sort_keys=True)


def test_repeated_read_timeout_retains_only_sanitized_diagnostics(
    tmp_path: Path,
) -> None:
    secret = "timeout-secret-key"
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(
            f"timeout carrying {secret}",
            request=request,
        )

    client = OpenAlexClient(
        api_key=secret,
        api_url="https://api.openalex.org/works",
        timeout_seconds=1,
        max_attempts=3,
        cache_dir=tmp_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=delays.append,
    )

    with pytest.raises(OpenAlexError) as error:
        client.search(query="weather", page=1, per_page=10)

    exception = error.value
    rendered_traceback = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    structured_record = json.dumps(exception.to_record(), sort_keys=True)
    request_metadata = json.dumps(
        [asdict(record) for record in client.request_records], default=str, sort_keys=True
    )

    assert calls == 3
    assert delays == [1.0, 2.0]
    assert exception.transport_error_type == "ReadTimeout"
    assert exception.attempts == 3
    assert exception.__cause__ is None
    assert exception.__context__ is None
    assert secret not in str(exception)
    assert secret not in repr(exception)
    assert secret not in rendered_traceback
    assert secret not in structured_record
    assert secret not in request_metadata
    assert not list(tmp_path.iterdir())


def test_cache_failure_after_http_200_is_not_retried_or_mislabeled(
    tmp_path: Path,
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload)

    cache_path = tmp_path / "not-a-directory"
    cache_path.write_text("occupied", encoding="utf-8")
    client = OpenAlexClient(
        api_key="cache-test-key",
        api_url="https://api.openalex.org/works",
        timeout_seconds=1,
        max_attempts=3,
        cache_dir=cache_path,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(FileExistsError) as error:
        client.search(query="weather", page=1, per_page=10)

    assert calls == 1
    assert "bounded retries" not in str(error.value)
    assert client.request_records == []


def test_openalex_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENALEX_API_KEY"):
        OpenAlexClient(
            api_key="",
            api_url="https://api.openalex.org/works",
            timeout_seconds=1,
            max_attempts=1,
        )


def test_has_content_without_content_urls_uses_canonical_openalex_pdf_url() -> None:
    candidate = parse_work(
        {
            "id": "https://openalex.org/W999",
            "title": "Weather Effects on Wind Power Generation",
            "open_access": {"is_oa": True, "oa_status": "green"},
            "has_content": {"pdf": True},
        },
        query="weather wind power generation",
        rank=1,
    )

    assert candidate.open_access.has_content_pdf is True
    assert candidate.open_access.content_pdf_url == (
        "https://content.openalex.org/works/W999.pdf"
    )
