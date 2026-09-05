"""
Azure AI Search service — upserts and deletes code chunks.

Replaces ChromaDB from the prototype. Same chunking logic, different backend.

Each chunk represents one function, class, or method extracted by the crawler.
The chunk_id follows the naming convention:
    {resource_code}:{github_org}:{project}:{repo}:{file}:{symbol}

Authentication uses DefaultAzureCredential (Managed Identity in Azure).
"""
import os

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField as VectorField,
)

INDEX_NAME = "code-chunks"
VECTOR_DIMENSIONS = 1536  # OpenAI text-embedding-3-small dimensions


def _endpoint() -> str:
    """Derive AI Search endpoint from environment."""
    env = os.environ.get("ENV", "dev")
    scope = os.environ.get("SEARCH_SCOPE", "shared")
    return f"https://srch-sdlc-{scope}-{env}.search.windows.net"


async def ensure_index_exists() -> None:
    """
    Create the code-chunks index if it does not already exist.
    Called once at worker startup — safe to call multiple times (no-op if exists).
    """
    credential = DefaultAzureCredential()

    async with SearchIndexClient(_endpoint(), credential) as client:
        existing = [name async for name in client.list_index_names()]
        if INDEX_NAME in existing:
            return

        index = SearchIndex(
            name=INDEX_NAME,
            fields=[
                SimpleField(name="chunk_id",      type=SearchFieldDataType.String, key=True),
                SimpleField(name="resource_code", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="github_org",    type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="project",       type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="repo",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="file",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="type",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="name",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sha",           type=SearchFieldDataType.String),
                SearchableField(name="content",   type=SearchFieldDataType.String),
                VectorField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=VECTOR_DIMENSIONS,
                    vector_search_profile_name="default-profile",
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
                profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="default-hnsw")],
            ),
        )

        await client.create_index(index)


async def upsert_chunk(chunk: dict) -> None:
    """
    Upsert a single chunk document into AI Search.

    chunk must contain: chunk_id, resource_code, github_org, project,
    repo, file, type, name, sha, content.
    content_vector is optional — added later when OpenAI is wired in.
    """
    credential = DefaultAzureCredential()

    async with SearchClient(_endpoint(), INDEX_NAME, credential) as client:
        # allowUnsafeKeys=True allows chunk_id to contain ':' and '/' characters
        # which are part of our naming convention: {resource_code}:{org}:{project}:...
        await client.upload_documents(documents=[chunk], allowUnsafeKeys=True)


async def delete_chunks(chunk_ids: list[str]) -> None:
    """
    Delete chunks by ID — called when functions/classes are removed from source.
    Keeps the index clean — no stale chunks from deleted code.
    """
    if not chunk_ids:
        return

    credential = DefaultAzureCredential()
    documents = [{"chunk_id": cid} for cid in chunk_ids]

    async with SearchClient(_endpoint(), INDEX_NAME, credential) as client:
        await client.delete_documents(documents=documents)


async def get_chunk_ids_for_file(resource_code: str, repo: str, file_path: str) -> list[str]:
    """
    Return all chunk IDs currently indexed for a given file.
    Used to detect stale chunks when a file is re-crawled.
    """
    credential = DefaultAzureCredential()
    filter_expr = (
        f"resource_code eq '{resource_code}' and "
        f"repo eq '{repo}' and "
        f"file eq '{file_path}'"
    )

    async with SearchClient(_endpoint(), INDEX_NAME, credential) as client:
        results = await client.search(
            search_text="*",
            filter=filter_expr,
            select=["chunk_id"],
            top=1000,
        )
        return [r["chunk_id"] async for r in results]
