"""
Chunker — converts crawler output into text chunks and upserts to Azure AI Search.

Adapted from chunkers/python_chunker.py in the prototype.
Only change: ChromaDB replaced with Azure AI Search (ai_search.py).

Each chunk stored with:
  doc_key       — opaque base64-encoded document identifier
  metadata      — resource_code, github_org, project, repo, file, type, name, class_name
  content       — searchable text with signature + docstring + code

Filtering by metadata fields (not by parsing an ID):
  filter = "resource_code eq 'x' and repo eq 'y' and type eq 'function'"

Incremental update logic:
  1. Get existing doc_keys for this file from AI Search
  2. Upsert new chunks (track new doc_keys)
  3. Delete stale doc_keys (functions/classes that no longer exist)
"""
from app.services.ai_search import (
    build_doc_key,
    delete_chunks,
    get_doc_keys_for_file,
    upsert_chunk,
)

MAX_CHUNK_TOKENS = 400


def _count_tokens(text: str) -> int:
    """Estimate token count — whitespace split is good enough for chunk size checks."""
    return len(text.split())


def _split_by_lines(text: str, max_tokens: int) -> list[str]:
    """Split long text into chunks by line boundaries within max_tokens."""
    lines = text.split("\n")
    chunks, current, count = [], [], 0

    for line in lines:
        line_tokens = _count_tokens(line)
        if count + line_tokens > max_tokens and current:
            chunks.append("\n".join(current))
            current, count = [line], line_tokens
        else:
            current.append(line)
            count += line_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks


def _build_function_text(func: dict, file_path: str) -> str:
    """Build rich text for a standalone function chunk."""
    params = ", ".join(f"{p['name']} ({p['type']})" for p in func.get("parameters", []))
    calls  = ", ".join(func.get("calls", [])) or "none"

    return f"""Function: {func['name']}
File: {file_path}
Parameters: {params or 'none'}
Returns: {func.get('return_type', 'unknown')}
Calls: {calls}
Docstring: {func.get('docstring', '')}
Code:
{func.get('source_code', '')}"""


def _build_method_text(method: dict, class_name: str, file_path: str) -> str:
    """Build rich text for a class method chunk."""
    params = ", ".join(f"{p['name']} ({p['type']})" for p in method.get("parameters", []))
    calls  = ", ".join(method.get("calls", [])) or "none"

    return f"""Method: {method['name']}
Class: {class_name}
File: {file_path}
Parameters: {params or 'none'}
Returns: {method.get('return_type', 'unknown')}
Calls: {calls}
Docstring: {method.get('docstring', '')}
Code:
{method.get('source_code', '')}"""


def _build_class_text(cls: dict, file_path: str) -> str:
    """Build rich text for a class chunk."""
    method_names = ", ".join(m["name"] for m in cls.get("methods", []))

    return f"""Class: {cls['name']}
File: {file_path}
Methods: {method_names or 'none'}
Docstring: {cls.get('docstring', '')}
Code:
{cls.get('source_code', '')}"""


def _base_metadata(
    resource_code: str, github_org: str, project: str, repo: str,
    file_path: str, sha: str,
) -> dict:
    """Build metadata fields shared across all chunks from one file."""
    return {
        "resource_code": resource_code,
        "github_org":    github_org,
        "project":       project,
        "repo":          repo,
        "file":          file_path,
        "sha":           sha,
    }


async def _upsert_one_chunk(
    doc_key: str, text: str, metadata: dict, new_keys: set,
) -> None:
    """Upsert a single chunk document to AI Search."""
    doc = {"doc_key": doc_key, "content": text, **metadata}
    await upsert_chunk(doc)
    new_keys.add(doc_key)


async def _upsert_chunks_for_text(
    resource_code: str, github_org: str, project: str, repo: str,
    file_path: str, symbol_parts: list[str], text: str, metadata: dict, new_keys: set,
) -> None:
    """
    Upsert one chunk if text fits, otherwise split into multiple chunks.
    Each split part gets its own doc_key with a suffix.
    """
    if _count_tokens(text) <= MAX_CHUNK_TOKENS:
        doc_key = build_doc_key(resource_code, github_org, project, repo, file_path, *symbol_parts)
        await _upsert_one_chunk(doc_key, text, metadata, new_keys)
        return

    parts = _split_by_lines(text, MAX_CHUNK_TOKENS)
    for i, part in enumerate(parts, start=1):
        suffix = f"{i}_of_{len(parts)}"
        doc_key = build_doc_key(resource_code, github_org, project, repo, file_path, *symbol_parts, suffix)
        await _upsert_one_chunk(doc_key, part, metadata, new_keys)


async def chunk_file(
    crawl_result: dict,
    resource_code: str,
    github_org: str,
    project: str,
    repo: str,
) -> dict:
    """
    Process one crawl result and upsert all its chunks to AI Search.
    Deletes stale chunks for functions/classes removed from the file.
    Returns summary with counts of chunks upserted and deleted.
    """
    file_path = crawl_result["path"]
    sha       = crawl_result.get("sha", "")

    base = _base_metadata(resource_code, github_org, project, repo, file_path, sha)

    # get existing doc_keys before upserting — needed for stale detection
    existing_keys = set(await get_doc_keys_for_file(resource_code, repo, file_path))
    new_keys: set = set()

    # chunk standalone functions
    for func in crawl_result.get("functions", []):
        metadata = {**base, "type": "function", "name": func["name"]}
        text     = _build_function_text(func, file_path)
        await _upsert_chunks_for_text(
            resource_code, github_org, project, repo, file_path,
            [func["name"]], text, metadata, new_keys,
        )

    # chunk classes and their methods
    for cls in crawl_result.get("classes", []):
        metadata = {**base, "type": "class", "name": cls["name"]}
        await _upsert_chunks_for_text(
            resource_code, github_org, project, repo, file_path,
            [cls["name"]], _build_class_text(cls, file_path), metadata, new_keys,
        )

        for method in cls.get("methods", []):
            metadata = {**base, "type": "method", "name": method["name"], "class_name": cls["name"]}
            await _upsert_chunks_for_text(
                resource_code, github_org, project, repo, file_path,
                [cls["name"], method["name"]], _build_method_text(method, cls["name"], file_path),
                metadata, new_keys,
            )

    # delete stale chunks — functions/classes removed from source
    stale_keys = list(existing_keys - new_keys)
    await delete_chunks(stale_keys)

    return {
        "file":     file_path,
        "upserted": len(new_keys),
        "deleted":  len(stale_keys),
    }
