"""
Base crawler interface — every file-type crawler must implement this.

Adding a new crawler (e.g. MarkdownCrawler):
  1. Create a new file: crawlers/markdown_crawler.py
  2. Implement BaseCrawler
  3. Register in crawlers/registry.py

No changes needed anywhere else.
"""
from abc import ABC, abstractmethod


class BaseCrawler(ABC):

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Return the file extensions this crawler handles.
        Example: ['.py'] or ['.md', '.mdx']
        """

    @abstractmethod
    def crawl(self, content: str, file_path: str, sha: str) -> dict:
        """
        Parse file content and return structured dict.

        Args:
            content:   Raw file content fetched from GitHub API
            file_path: Relative path within the repo (e.g. 'cart/cart.py')
            sha:       GitHub file SHA — used for incremental change detection

        Returns:
            Structured dict with extracted metadata, code, and relationships.
            Must include 'sha' field for checkpoint comparison on next run.
        """
