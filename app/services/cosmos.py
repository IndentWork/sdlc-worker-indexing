"""
Cosmos DB service — upserts and deletes graph nodes and edges.

Replaces Neo4j from the prototype. Same graph structure, different backend.

Two containers:
  nodes — File, Function, Class, Method entities
  edges — CALLS, IMPORTS, HAS_METHOD relationships

All documents partitioned by resource_code for fast tenant queries.
Authentication uses DefaultAzureCredential (Managed Identity in Azure).
"""
import os

from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential

DATABASE_NAME  = "sdlc"
NODES_CONTAINER = "nodes"
EDGES_CONTAINER = "edges"


def _endpoint() -> str:
    """Derive Cosmos DB endpoint from environment."""
    env = os.environ.get("ENV", "dev")
    scope = os.environ.get("COSMOS_SCOPE", "shared")
    return f"https://cosmos-sdlc-{scope}-{env}.documents.azure.com:443/"


def _build_node_id(
    resource_code: str,
    github_org: str,
    project: str,
    repo: str,
    file_path: str,
    *parts: str,
) -> str:
    """
    Build a unique stable node ID following the naming convention.

    Format: {resource_code}:{github_org}:{project}:{repo}:{file}:{symbol}
    Example: b310545b:sdlc-tenant:ecommerce:cart-service:cart.py:add_to_cart
    """
    return ":".join([resource_code, github_org, project, repo, file_path] + list(parts))


async def upsert_node(node: dict) -> None:
    """
    Upsert a single graph node (File, Function, Class, or Method).

    node must contain: id, resource_code, type, and type-specific fields.
    Uses upsert so re-indexing the same file is safe — no duplicates.
    """
    credential = DefaultAzureCredential()

    async with CosmosClient(_endpoint(), credential) as client:
        container = client.get_database_client(DATABASE_NAME).get_container_client(NODES_CONTAINER)
        await container.upsert_item(node)


async def upsert_edge(edge: dict) -> None:
    """
    Upsert a single graph edge (CALLS, IMPORTS, or HAS_METHOD relationship).

    edge must contain: id, resource_code, source_id, target_id, relationship.
    """
    credential = DefaultAzureCredential()

    async with CosmosClient(_endpoint(), credential) as client:
        container = client.get_database_client(DATABASE_NAME).get_container_client(EDGES_CONTAINER)
        await container.upsert_item(edge)


async def delete_node(node_id: str, resource_code: str) -> None:
    """
    Delete a node by ID — called when a function/class is removed from source.
    resource_code is the partition key required by Cosmos DB for deletion.
    """
    credential = DefaultAzureCredential()

    async with CosmosClient(_endpoint(), credential) as client:
        container = client.get_database_client(DATABASE_NAME).get_container_client(NODES_CONTAINER)
        await container.delete_item(item=node_id, partition_key=resource_code)


async def get_node_ids_for_file(resource_code: str, repo: str, file_path: str) -> list[str]:
    """
    Return all node IDs currently stored for a given file.
    Used to detect stale nodes when a file is re-crawled.
    """
    credential = DefaultAzureCredential()
    query = (
        "SELECT c.id FROM c "
        "WHERE c.resource_code = @resource_code "
        "AND c.repo = @repo "
        "AND c.file = @file"
    )
    parameters = [
        {"name": "@resource_code", "value": resource_code},
        {"name": "@repo",          "value": repo},
        {"name": "@file",          "value": file_path},
    ]

    async with CosmosClient(_endpoint(), credential) as client:
        container = client.get_database_client(DATABASE_NAME).get_container_client(NODES_CONTAINER)
        items = container.query_items(query=query, parameters=parameters)
        return [item["id"] async for item in items]
