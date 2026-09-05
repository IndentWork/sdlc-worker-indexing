"""
Chunker — converts crawler output into text chunks and upserts to Azure AI Search.

Adapted from chunkers/python_chunker.py in the prototype.
Only change: ChromaDB replaced with Azure AI Search (ai_search.py).

Chunk ID follows the naming convention:
    {resource_code}:{github_org}:{project}:{repo}:{file}:{symbol}

Incremental update logic:
  1. Get existing chunk IDs for this file from AI Search
  2. Upsert new chunks
  3. Delete stale chunk IDs (functions/classes that no longer exist)
"""
from app.services.ai_search import upsert_chunk, delete_chunks, get_chunk_ids_for_file

MAX_CHUNK_TOKENS = 400


def _count_tokens(text: str) -> int:
    """Estimate token count — whitespace split is good enough for chunk size checks."""
    return len(text.split())


def _split_by_lines(text: str, max_tokens: int) -> list[str]:
    """
    Split a long text into chunks by line boundaries.
    Each chunk stays within max_tokens.
    """
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


def _build_chunk_id(
    resource_code: str, github_org: str, project: str, repo: str, file_path: str, *parts: str
) -> str:
    """
    Build chunk ID following the naming convention.
    Example: b310545b:sdlc-tenant:ecommerce:cart-service:cart.py:add_to_cart
    """
    return ":".join([resource_code, github_org, project, repo, file_path] + list(parts))


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


async def _upsert_chunks_for_text(
    chunk_id: str, text: str, metadata: dict, new_ids: set
) -> None:
    """
    Upsert one or more chunks for a piece of code.
    Splits into multiple chunks if text exceeds MAX_CHUNK_TOKENS.
    """
    if _count_tokens(text) <= MAX_CHUNK_TOKENS:
        doc = {"chunk_id": chunk_id, "content": text, **metadata}
        await upsert_chunk(doc)
        new_ids.add(chunk_id)
    else:
        parts = _split_by_lines(text, MAX_CHUNK_TOKENS)
        for i, part in enumerate(parts, start=1):
            part_id = f"{chunk_id}:{i}_of_{len(parts)}"
            doc = {"chunk_id": part_id, "content": part, **metadata}
            await upsert_chunk(doc)
            new_ids.add(part_id)


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

    Returns a summary dict with counts of chunks upserted and deleted.
    """
    file_path = crawl_result["path"]
    sha       = crawl_result.get("sha", "")

    base = _base_metadata(resource_code, github_org, project, repo, file_path, sha)

    # get existing chunk IDs before upserting — needed for stale detection
    existing_ids = set(await get_chunk_ids_for_file(resource_code, repo, file_path))
    new_ids: set = set()

    # chunk standalone functions
    for func in crawl_result.get("functions", []):
        chunk_id = _build_chunk_id(resource_code, github_org, project, repo, file_path, func["name"])
        text     = _build_function_text(func, file_path)
        metadata = {**base, "type": "function", "name": func["name"]}
        await _upsert_chunks_for_text(chunk_id, text, metadata, new_ids)

    # chunk classes and their methods
    for cls in crawl_result.get("classes", []):
        class_id = _build_chunk_id(resource_code, github_org, project, repo, file_path, cls["name"])
        await _upsert_chunks_for_text(
            class_id,
            _build_class_text(cls, file_path),
            {**base, "type": "class", "name": cls["name"]},
            new_ids,
        )

        for method in cls.get("methods", []):
            method_id = _build_chunk_id(
                resource_code, github_org, project, repo, file_path, cls["name"], method["name"]
            )
            await _upsert_chunks_for_text(
                method_id,
                _build_method_text(method, cls["name"], file_path),
                {**base, "type": "method", "name": method["name"], "class_name": cls["name"]},
                new_ids,
            )

    # delete stale chunks — functions/classes removed from source
    stale_ids = list(existing_ids - new_ids)
    await delete_chunks(stale_ids)

    return {
        "file":    file_path,
        "upserted": len(new_ids),
        "deleted":  len(stale_ids),
    }
