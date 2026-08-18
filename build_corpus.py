"""Build a source-attributed Hindi RAG corpus from approved government sites.

The crawler deliberately stays small and conservative: it only visits approved
domains, honours robots.txt, applies a per-domain delay, and records the URL and
retrieval time for every document.  Review the generated JSONL before indexing it.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


USER_AGENT = "IndicLLMLinguaCourseProject/0.1 (educational RAG corpus builder)"
HINDI_RE = re.compile(r"[\u0900-\u097F]")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Source:
    name: str
    publisher: str
    start_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    path_prefixes: tuple[str, ...]


SOURCES = {
    "myscheme": Source(
        name="myScheme Hindi",
        publisher="Digital India Corporation, Government of India",
        start_urls=("https://www.myscheme.gov.in/hi/find-scheme",),
        allowed_domains=("www.myscheme.gov.in",),
        path_prefixes=("/hi/",),
    ),
    "india-gov": Source(
        name="National Portal of India",
        publisher="Government of India",
        start_urls=("https://www.india.gov.in/my-government/schemes",),
        allowed_domains=("www.india.gov.in", "services.india.gov.in"),
        path_prefixes=("/",),
    ),
    "pib": Source(
        name="Press Information Bureau Hindi",
        publisher="Press Information Bureau, Government of India",
        start_urls=("https://www.pib.gov.in/indexm.aspx",),
        allowed_domains=("www.pib.gov.in", "pib.gov.in"),
        path_prefixes=("/",),
    ),
    "cooperation": Source(
        name="Ministry of Cooperation schemes",
        publisher="Ministry of Cooperation, Government of India",
        start_urls=("https://www.cooperation.gov.in/schemes-allied-ministriesdept",),
        allowed_domains=("www.cooperation.gov.in", "cooperation.gov.in"),
        path_prefixes=("/",),
    ),
}


class PageParser(HTMLParser):
    """Extract visible text, title, headings and links without extra packages."""

    IGNORED = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._ignored_depth = 0
        self._parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.IGNORED:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a" and self._ignored_depth == 0:
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORED and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title += data
        self._parts.append(data)

    @property
    def text(self) -> str:
        return SPACE_RE.sub(" ", html.unescape(" ".join(self._parts))).strip()


def normalized_url(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/")


def allowed(url: str, source: Source) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.lower() in source.allowed_domains
        and any(parsed.path.startswith(prefix) for prefix in source.path_prefixes)
        and not parsed.path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4"))
    )


def robots_for(url: str, cache: dict[str, RobotFileParser]) -> RobotFileParser:
    parsed = urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in cache:
        parser = RobotFileParser(f"{root}/robots.txt")
        try:
            parser.read()
        except (OSError, URLError):
            # Do not scrape when robots.txt cannot be checked.
            parser.parse(["User-agent: *", "Disallow: /"])
        cache[root] = parser
    return cache[root]


def fetch(url: str, timeout: int) -> tuple[str, str] | None:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "hi,en;q=0.8"})
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: URL is allowlisted above
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "application/xhtml+xml"}:
                return None
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace"), response.geturl()
    except (HTTPError, URLError, TimeoutError, ValueError) as error:
        print(f"Skipping {url}: {error}")
        return None


def crawl(source: Source, limit: int, delay: float, timeout: int) -> Iterable[dict[str, str]]:
    queue = [normalized_url(url) for url in source.start_urls]
    seen: set[str] = set()
    robot_cache: dict[str, RobotFileParser] = {}
    last_request: dict[str, float] = {}

    while queue and len(seen) < limit:
        url = queue.pop(0)
        if url in seen or not allowed(url, source):
            continue
        seen.add(url)
        robot = robots_for(url, robot_cache)
        if not robot.can_fetch(USER_AGENT, url):
            print(f"robots.txt disallows {url}")
            continue

        domain = urlparse(url).netloc
        # A site's robots.txt crawl-delay takes precedence over our requested
        # delay.  It is important not to make a "fast" CLI option circumvent it.
        robots_delay = robot.crawl_delay(USER_AGENT) or robot.crawl_delay("*") or 0
        effective_delay = max(delay, robots_delay)
        wait = effective_delay - (time.monotonic() - last_request.get(domain, 0))
        if wait > 0:
            time.sleep(wait)
        result = fetch(url, timeout)
        last_request[domain] = time.monotonic()
        if not result:
            continue

        page, resolved_url = result
        parser = PageParser()
        parser.feed(page)
        text = parser.text
        for href in parser.links:
            candidate = normalized_url(urljoin(resolved_url, href))
            if candidate not in seen and allowed(candidate, source):
                queue.append(candidate)

        # English navigation pages are common; require enough Hindi text for this corpus.
        if len(text) >= 300 and len(HINDI_RE.findall(text)) >= 30:
            yield {
                "document_id": hashlib.sha256(resolved_url.encode()).hexdigest()[:16],
                "title": SPACE_RE.sub(" ", parser.title).strip() or resolved_url,
                "text": text,
                "source_url": resolved_url,
                "source_name": source.name,
                "publisher": source.publisher,
                "language": "hi",
                "retrieved_at": datetime.now(UTC).isoformat(),
                "license_note": "Retain source attribution; verify reuse terms before redistribution.",
            }


def chunk_text(text: str, words_per_chunk: int, overlap_words: int) -> Iterable[str]:
    words = text.split()
    step = max(1, words_per_chunk - overlap_words)
    for start in range(0, len(words), step):
        chunk = words[start : start + words_per_chunk]
        if len(chunk) >= 30:
            yield " ".join(chunk)
        if start + words_per_chunk >= len(words):
            break


def load_seed_documents(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    """Read reviewed, source-attributed documents maintained in the repository."""
    required = {"document_id", "title", "text", "source_url", "language"}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                document = json.loads(line)
                missing = required - document.keys()
                if missing:
                    raise ValueError(f"{path}:{line_number} is missing fields: {sorted(missing)}")
                yield document


def write_corpus(documents: Iterable[dict[str, str]], output: Path, chunk_output: Path, words: int, overlap: int) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    documents_written = chunks_written = 0
    seen_text: set[str] = set()
    with output.open("w", encoding="utf-8") as docs, chunk_output.open("w", encoding="utf-8") as chunks:
        for document in documents:
            fingerprint = hashlib.sha256(document["text"].encode()).hexdigest()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            docs.write(json.dumps(document, ensure_ascii=False) + "\n")
            documents_written += 1
            for number, text in enumerate(chunk_text(document["text"], words, overlap), start=1):
                record = {
                    **document,
                    "chunk_id": f"{document['document_id']}::chunk-{number:04d}",
                    "text": text,
                }
                chunks.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks_written += 1
    return documents_written, chunks_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--seed-documents", nargs="*", type=Path, default=[], help="Reviewed source-attributed JSONL documents to include.")
    parser.add_argument("--no-crawl", action="store_true", help="Build the corpus from seed documents only.")
    parser.add_argument("--limit-per-source", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.5, help="Minimum seconds between requests to one domain.")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("corpus/processed/documents.jsonl"))
    parser.add_argument("--chunk-output", type=Path, default=Path("corpus/processed/chunks.jsonl"))
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()
    if args.show_sources:
        print(json.dumps({name: asdict(source) for name, source in SOURCES.items()}, indent=2))
        return
    if args.limit_per_source < 1 or args.overlap_words >= args.chunk_words:
        parser.error("limit must be positive and overlap must be smaller than chunk size")

    crawled = (
        document
        for source_name in ([] if args.no_crawl else args.sources)
        for document in crawl(SOURCES[source_name], args.limit_per_source, args.delay, args.timeout)
    )
    documents = (*load_seed_documents(args.seed_documents), *crawled)
    docs, chunks = write_corpus(documents, args.output, args.chunk_output, args.chunk_words, args.overlap_words)
    print(f"Wrote {docs} documents and {chunks} chunks to {args.output.parent}")


if __name__ == "__main__":
    main()
