"""
Python crawler — handles .py files using Python's built-in AST module.

Wraps python_ast.py from the prototype. The only difference from the
prototype is that content is passed in as a string (fetched from GitHub API)
instead of being read from disk.
"""
import ast
from pathlib import Path

from app.indexing.crawlers.base import BaseCrawler
from app.indexing.crawlers.python_ast import (
    extract_classes,
    extract_functions,
    extract_imports,
    extract_module_level_calls,
    extract_module_level_variables,
)


class PythonCrawler(BaseCrawler):

    def supported_extensions(self) -> list[str]:
        return [".py"]

    def crawl(self, content: str, file_path: str, sha: str) -> dict:
        """
        Parse Python file content using AST and return structured dict.

        Extracts: imports, module-level variables/calls, functions, classes.
        Attaches GitHub SHA for incremental change detection on next run.
        """
        tree = ast.parse(content)

        return {
            "file":             Path(file_path).name,
            "path":             file_path,
            "sha":              sha,
            "imports":          extract_imports(tree),
            "module_variables": extract_module_level_variables(tree),
            "module_calls":     extract_module_level_calls(tree),
            "functions":        extract_functions(tree, content),
            "classes":          extract_classes(tree, content),
        }
