"""
Storage service — reads and writes checkpoint files to Azure Blob Storage.

All blobs live in the single 'sdlc' container following the hierarchy:
  sdlc/{resource_code}/{github_org}/sdlc.yml
  sdlc/{resource_code}/{github_org}/index/{project}/{repo}/{file}.json
  sdlc/{resource_code}/{github_org}/index/{project}/{repo}/_project.json

Authentication uses DefaultAzureCredential (Managed Identity in Azure).
"""
import json
import os

from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

CONTAINER = "sdlc"


def _account_url(tier: str, resource_code: str) -> str:
    """Derive storage account URL from tenant tier and resource_code."""
    env = os.environ.get("ENV", "dev")
    if tier == "shared":
        name = f"stsdlcshared{env}"
    else:
        name = f"stsdlc{resource_code}{env}"
    return f"https://{name}.blob.core.windows.net"


def _index_prefix(resource_code: str, github_org: str, project: str, repo: str) -> str:
    """
    Build the blob path prefix for a repo's index files.
    Example: b310545b/sdlc-tenant/index/ecommerce/cart-service/
    """
    return f"{resource_code}/{github_org}/index/{project}/{repo}"


def _project_index_path(resource_code: str, github_org: str, project: str, repo: str) -> str:
    """
    Path to the _project.json index file for a repo.
    Example: b310545b/sdlc-tenant/index/ecommerce/cart-service/_project.json
    """
    return f"{_index_prefix(resource_code, github_org, project, repo)}/_project.json"


def _file_index_path(
    resource_code: str, github_org: str, project: str, repo: str, file_path: str
) -> str:
    """
    Path to the index JSON for a single source file.
    Example: b310545b/sdlc-tenant/index/ecommerce/cart-service/cart.py.json
    """
    safe_name = file_path.replace("/", "__") + ".json"
    return f"{_index_prefix(resource_code, github_org, project, repo)}/{safe_name}"


async def load_project_index(
    tier: str, resource_code: str, github_org: str, project: str, repo: str,
) -> dict | None:
    """
    Load _project.json for a repo from Storage.
    Returns None if this repo has never been indexed (first run).
    """
    url = _account_url(tier, resource_code)
    blob_path = _project_index_path(resource_code, github_org, project, repo)

    async with BlobServiceClient(url, DefaultAzureCredential()) as client:
        blob = client.get_blob_client(CONTAINER, blob_path)
        try:
            stream = await blob.download_blob()
            content = await stream.readall()
            return json.loads(content)
        except Exception:
            return None


async def save_project_index(
    tier: str, resource_code: str, github_org: str, project: str, repo: str, index: dict,
) -> None:
    """Save updated _project.json for a repo to Storage."""
    url = _account_url(tier, resource_code)
    blob_path = _project_index_path(resource_code, github_org, project, repo)

    async with BlobServiceClient(url, DefaultAzureCredential()) as client:
        blob = client.get_blob_client(CONTAINER, blob_path)
        await blob.upload_blob(
            json.dumps(index, indent=2).encode("utf-8"),
            overwrite=True,
        )


async def load_file_checkpoint(
    tier: str, resource_code: str, github_org: str, project: str, repo: str, file_path: str,
) -> dict | None:
    """Load the index JSON for a single source file. Returns None if never indexed."""
    url = _account_url(tier, resource_code)
    blob_path = _file_index_path(resource_code, github_org, project, repo, file_path)

    async with BlobServiceClient(url, DefaultAzureCredential()) as client:
        blob = client.get_blob_client(CONTAINER, blob_path)
        try:
            stream = await blob.download_blob()
            content = await stream.readall()
            return json.loads(content)
        except Exception:
            return None


async def save_file_checkpoint(
    tier: str, resource_code: str, github_org: str, project: str,
    repo: str, file_path: str, crawl_result: dict,
) -> None:
    """Save crawl output for a single source file to Storage."""
    url = _account_url(tier, resource_code)
    blob_path = _file_index_path(resource_code, github_org, project, repo, file_path)

    async with BlobServiceClient(url, DefaultAzureCredential()) as client:
        blob = client.get_blob_client(CONTAINER, blob_path)
        await blob.upload_blob(
            json.dumps(crawl_result, indent=2).encode("utf-8"),
            overwrite=True,
        )
