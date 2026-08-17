from pathlib import Path
from dataclasses import asdict
import json

import httpx

from l3s_agent.literature.download import PDFDownloader
from l3s_agent.literature.models import DownloadStatus


def usable_pdf() -> bytes:
    return b"%PDF-1.7\n" + (b"0" * 1100) + b"\n%%EOF\n"


def test_download_validates_pdf_and_computes_sha256(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, content=usable_pdf(), headers={"content-type": "application/pdf"}
        )
    )
    downloader = PDFDownloader(
        timeout_seconds=1,
        min_pdf_bytes=1024,
        max_pdf_bytes=10_000,
        client=httpx.Client(transport=transport),
    )

    result = downloader.download(
        paper_id="paper_W1",
        pdf_urls=("https://example.org/work.pdf",),
        destination_dir=tmp_path,
    )

    assert result.status is DownloadStatus.SUCCESS
    assert result.sha256 is not None and len(result.sha256) == 64
    assert result.local_path == tmp_path / "paper_W1.pdf"
    assert result.local_path.read_bytes() == usable_pdf()


def test_download_rejects_html_and_tries_next_pdf_url(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bad.pdf":
            return httpx.Response(200, content=b"<html>no</html>", headers={"content-type": "text/html"})
        return httpx.Response(200, content=usable_pdf())

    downloader = PDFDownloader(
        timeout_seconds=1,
        min_pdf_bytes=1024,
        max_pdf_bytes=10_000,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = downloader.download(
        paper_id="paper_W1",
        pdf_urls=("https://example.org/bad.pdf", "https://example.org/good.pdf"),
        destination_dir=tmp_path,
    )

    assert result.status is DownloadStatus.SUCCESS
    assert result.attempted_urls == (
        "https://example.org/bad.pdf",
        "https://example.org/good.pdf",
    )
    assert result.successful_url == "https://example.org/good.pdf"


def test_openalex_content_pdf_has_priority_and_secret_is_not_retained(
    tmp_path: Path,
) -> None:
    secret = "content-secret-key"
    observed_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_urls.append(str(request.url))
        return httpx.Response(200, content=usable_pdf())

    downloader = PDFDownloader(
        timeout_seconds=1,
        min_pdf_bytes=1024,
        max_pdf_bytes=10_000,
        openalex_api_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    content_url = "https://content.openalex.org/works/W1.pdf"
    result = downloader.download(
        paper_id="paper_W1",
        openalex_content_url=content_url,
        pdf_urls=("https://repository.example/best.pdf",),
        destination_dir=tmp_path,
    )

    assert len(observed_urls) == 1
    assert secret in observed_urls[0]
    assert result.status is DownloadStatus.SUCCESS
    assert result.attempted_urls == (content_url,)
    assert result.successful_url == content_url
    assert secret not in repr(result)
    assert secret not in json.dumps(asdict(result), default=str, sort_keys=True)


def test_openalex_then_best_then_alternate_priority_and_sanitized_failure(
    tmp_path: Path,
) -> None:
    secret = "never-persist-content-key"
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.host == "content.openalex.org":
            return httpx.Response(403)
        if request.url.host == "best.example":
            return httpx.Response(
                200, content=b"<html>blocked</html>", headers={"content-type": "text/html"}
            )
        return httpx.Response(200, content=usable_pdf())

    downloader = PDFDownloader(
        timeout_seconds=1,
        min_pdf_bytes=1024,
        max_pdf_bytes=10_000,
        openalex_api_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    content_url = "https://content.openalex.org/works/W1.pdf"
    result = downloader.download(
        paper_id="paper_W1",
        openalex_content_url=content_url,
        pdf_urls=("https://best.example/work.pdf", "https://alternate.example/work.pdf"),
        destination_dir=tmp_path,
    )

    assert paths == ["/works/W1.pdf", "/work.pdf", "/work.pdf"]
    assert result.status is DownloadStatus.SUCCESS
    assert result.attempted_urls == (
        content_url,
        "https://best.example/work.pdf",
        "https://alternate.example/work.pdf",
    )
    assert result.successful_url == "https://alternate.example/work.pdf"
    assert secret not in repr(result)


def test_failed_openalex_content_request_sanitizes_query_parameter_secret(
    tmp_path: Path,
) -> None:
    secret = "failed-content-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    downloader = PDFDownloader(
        timeout_seconds=1,
        min_pdf_bytes=1024,
        max_pdf_bytes=10_000,
        openalex_api_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = downloader.download(
        paper_id="paper_W1",
        openalex_content_url="https://content.openalex.org/works/W1.pdf",
        pdf_urls=(),
        destination_dir=tmp_path,
    )

    serialized = json.dumps(asdict(result), default=str, sort_keys=True)
    assert result.status is DownloadStatus.FAILED
    assert result.failure_reason == (
        "https://content.openalex.org/works/W1.pdf: HTTP 403"
    )
    assert secret not in str(result)
    assert secret not in repr(result)
    assert secret not in serialized


def test_content_request_error_does_not_retain_exception_or_secret(
    tmp_path: Path,
) -> None:
    secret = "transport-content-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timeout with {secret}", request=request)

    downloader = PDFDownloader(
        timeout_seconds=1,
        min_pdf_bytes=1024,
        max_pdf_bytes=10_000,
        openalex_api_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    content_url = "https://content.openalex.org/works/W1.pdf"
    result = downloader.download(
        paper_id="paper_W1",
        openalex_content_url=content_url,
        pdf_urls=(content_url,),
        destination_dir=tmp_path,
    )

    assert result.status is DownloadStatus.FAILED
    assert result.attempted_urls == (content_url,)
    assert result.failure_reason == f"{content_url}: ReadTimeout"
    assert secret not in str(result)
    assert secret not in repr(result)
