"""OA PDF download and byte-level validation without PDF parsing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Sequence

import httpx

from .models import DownloadRecord, DownloadStatus


class PDFDownloader:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        min_pdf_bytes: int,
        max_pdf_bytes: int,
        openalex_api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._min_pdf_bytes = min_pdf_bytes
        self._max_pdf_bytes = max_pdf_bytes
        self._openalex_api_key = openalex_api_key
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "l3s-scientific-agent/0.1"},
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PDFDownloader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def download(
        self,
        *,
        paper_id: str,
        pdf_urls: Sequence[str],
        destination_dir: Path,
        openalex_content_url: str | None = None,
    ) -> DownloadRecord:
        attempted: list[str] = []
        failures: list[str] = []
        sources = (
            ((openalex_content_url, True),) if openalex_content_url else ()
        ) + tuple((url, False) for url in pdf_urls)
        for url, is_openalex_content in sources:
            if not url or url in attempted:
                continue
            attempted.append(url)
            try:
                content = self._fetch(url, authenticated=is_openalex_content)
                self._validate(content)
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{paper_id}.pdf"
                self._atomic_write(destination, content)
                return DownloadRecord(
                    status=DownloadStatus.SUCCESS,
                    attempted_urls=tuple(attempted),
                    successful_url=url,
                    local_path=destination,
                    sha256=hashlib.sha256(content).hexdigest(),
                    content_length=len(content),
                )
            except (httpx.HTTPError, ValueError, OSError) as exc:
                failures.append(f"{url}: {self._safe_failure(exc)}")
        reason = "; ".join(failures) if failures else "no direct OA PDF URL"
        return DownloadRecord(
            status=DownloadStatus.FAILED,
            attempted_urls=tuple(attempted),
            failure_reason=reason,
        )

    def _fetch(self, url: str, *, authenticated: bool = False) -> bytes:
        params = None
        if authenticated and self._openalex_api_key:
            params = {"api_key": self._openalex_api_key}
        with self._client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise ValueError("response is HTML, not PDF")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > self._max_pdf_bytes:
                    raise ValueError("PDF exceeds configured maximum size")
                chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _safe_failure(exc: httpx.HTTPError | ValueError | OSError) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}"
        if isinstance(exc, httpx.RequestError):
            return type(exc).__name__
        return str(exc)

    def _validate(self, content: bytes) -> None:
        if len(content) < self._min_pdf_bytes:
            raise ValueError("PDF is smaller than configured minimum size")
        if b"%PDF-" not in content[:1024]:
            raise ValueError("missing PDF header")
        if b"%%EOF" not in content[-4096:]:
            raise ValueError("missing PDF EOF marker")

    @staticmethod
    def _atomic_write(destination: Path, content: bytes) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
