"""
Crawler registry — maps file extensions to their crawler implementations.

To add a new crawler (e.g. MarkdownCrawler):
  1. Create crawlers/markdown_crawler.py implementing BaseCrawler
  2. Import it here and add to CRAWLERS list
  3. Done — orchestrator picks it up automatically
"""
from pathlib import Path

from app.indexing.crawlers.base import BaseCrawler
from app.indexing.crawlers.python_crawler import PythonCrawler

# Register active crawlers here — order does not matter
CRAWLERS: list[BaseCrawler] = [
    PythonCrawler(),
    # MarkdownCrawler(),  ← uncomment when ready
]

# Build extension lookup map at startup for fast access
_EXTENSION_MAP: dict[str, BaseCrawler] = {
    ext: crawler
    for crawler in CRAWLERS
    for ext in crawler.supported_extensions()
}


def get_crawler(file_path: str) -> BaseCrawler | None:
    """
    Returns the crawler for the given file path based on its extension.
    Returns None if no crawler supports this file type — caller should skip.
    """
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_MAP.get(ext)


def supported_extensions() -> list[str]:
    """Returns all file extensions currently supported across all crawlers."""
    return list(_EXTENSION_MAP.keys())
