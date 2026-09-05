"""
Azure AI Search service — upserts and deletes code chunks.

Replaces ChromaDB from the prototype. Same chunking logic, different backend.

Each chunk represents one function, class, or method extracted by the crawler.
No chunk_id field is stored — filtering is done via metadata fields only:
    resource_code, github_org, project, repo, file, type, name, class_name

Document key (doc_key) is base64 URL-safe encoded from the compound path.
Base64 URL-safe uses only [A-Za-z0-9-_] — all valid AI Search key characters.

Authentication uses DefaultAzureCredential (Managed Identity in Azure).
"""
import base64
import os

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
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


def build_doc_key(
    resource_code: str, github_org: str, project: str, repo: str, file_path: str, *parts: str
) -> str:
    """
    Build the AI Search document key from the compound path.
    URL-safe base64 encoding — uses only [A-Za-z0-9-_] which are all valid.
    The '=' padding is stripped and re-added when decoding.

    Example inputs:
      b310545b, sdlc-tenant, ecommerce, cart-service, cart/main.py, add_to_cart
    Produces something like:
      YjMxMDU0NWI6c2RsYy10ZW5hbnQ6ZWNvbW1lcmNlOmNhcnQtc2VydmljZTpjYXJ0L21haW4ucHk6YWRkX3RvX2NhcnQ
    """
    compound = ":".join([resource_code, github_org, project, repo, file_path] + list(parts))
    return base64.urlsafe_b64encode(compound.encode()).decode().rstrip("=")


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
                # doc_key: opaque base64-encoded AI Search document identifier
                SimpleField(name="doc_key",        type=SearchFieldDataType.String, key=True),

                # metadata fields — all used for filtering
                SimpleField(name="resource_code", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="github_org",    type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="project",       type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="repo",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="file",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="type",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="name",          type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="class_name",    type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sha",           type=SearchFieldDataType.String),

                # searchable text — for BM25 keyword search
                SearchableField(name="content",   type=SearchFieldDataType.String),

                # vector field — for semantic search (populated when OpenAI is wired in)
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
    chunk must contain: doc_key, resource_code, github_org, project,
    repo, file, type, name, sha, content.
    """
    credential = DefaultAzureCredential()

    async with SearchClient(_endpoint(), INDEX_NAME, credential) as client:
        await client.upload_documents(documents=[chunk])


async def delete_chunks(doc_keys: list[str]) -> None:
    """
    Delete chunks by doc_key — called when functions/classes are removed from source.
    Keeps the index clean — no stale chunks from deleted code.
    """
    if not doc_keys:
        return

    credential = DefaultAzureCredential()
    documents = [{"doc_key": k} for k in doc_keys]

    async with SearchClient(_endpoint(), INDEX_NAME, credential) as client:
        await client.delete_documents(documents=documents)


async def get_doc_keys_for_file(resource_code: str, repo: str, file_path: str) -> list[str]:
    """
    Return all doc_keys currently indexed for a given file.
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
            select=["doc_key"],
            top=1000,
        )
        return [r["doc_key"] async for r in results]
