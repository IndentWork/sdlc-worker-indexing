"""
Storage service — reads and writes checkpoint files to Azure Blob Storage.

Checkpoint path pattern:
  checkpoints/{resource_code}/{github_org}/{project}/{repo}/{file}.json

One JSON file per Python file crawled.
One _project.json per repo as the index (list of files + SHAs).

This mirrors the prototype's output/ folder structure but in Azure Blob Storage.
Authentication uses DefaultAzureCredential (Managed Identity in Azure).
"""
import json
import os

from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient


def _account_url(tier: str, resource_code: str) -> str:
    """Derive storage account URL from tenant tier and resource_code."""
    env = os.environ.get("ENV", "dev")
    if tier == "shared":
        name = f"stsdlcshared{env}"
    else:
        name = f"stsdlc{resource_code}{env}"
    return f"https://{name}.blob.core.windows.net"


def _checkpoint_prefix(resource_code: str, github_org: str, project: str, repo: str) -> str:
    """
    Build the blob path prefix for a repo's checkpoints.
    All files for this repo live under this prefix.

    Example: checkpoints/b310545b/sdlc-tenant/ecommerce/cart-service/
    """
    return f"checkpoints/{resource_code}/{github_org}/{project}/{repo}"


def _project_index_path(resource_code: str, github_org: str, project: str, repo: str) -> str:
    """
    Path to the _project.json index file for a repo.

    Example: checkpoints/b310545b/sdlc-tenant/ecommerce/cart-service/_project.json
    """
    prefix = _checkpoint_prefix(resource_code, github_org, project, repo)
    return f"{prefix}/_project.json"


def _file_checkpoint_path(
    resource_code: str, github_org: str, project: str, repo: str, file_path: str
) -> str:
    """
    Path to the checkpoint JSON for a single source file.

    Example: checkpoints/b310545b/sdlc-tenant/ecommerce/cart-service/cart.py.json
    """
    prefix = _checkpoint_prefix(resource_code, github_org, project, repo)
    # Replace path separators with __ to keep it as a flat blob name
    safe_name = file_path.replace("/", "__") + ".json"
    return f"{prefix}/{safe_name}"


async def load_project_index(
    tier: str,
    resource_code: str,
    github_org: str,
    project: str,
    repo: str,
) -> dict | None:
    """
    Load _project.json for a repo from Storage.
    Returns None if this repo has never been indexed (first run).
    """
    url = _account_url(tier, resource_code)
    blob_path = _project_index_path(resource_code, github_org, project, repo)
    credential = DefaultAzureCredential()

    async with BlobServiceClient(url, credential) as client:
        blob = client.get_blob_client("checkpoints", blob_path)
        try:
            stream = await blob.download_blob()
            content = await stream.readall()
            return json.loads(content)
        except Exception:
            # Blob does not exist — first time indexing this repo
            return None


async def save_project_index(
    tier: str,
    resource_code: str,
    github_org: str,
    project: str,
    repo: str,
    index: dict,
) -> None:
    """
    Save updated _project.json for a repo to Storage.
    Called after indexing completes to record new SHAs.
    """
    url = _account_url(tier, resource_code)
    blob_path = _project_index_path(resource_code, github_org, project, repo)
    credential = DefaultAzureCredential()

    async with BlobServiceClient(url, credential) as client:
        blob = client.get_blob_client("checkpoints", blob_path)
        await blob.upload_blob(
            json.dumps(index, indent=2).encode("utf-8"),
            overwrite=True,
        )


async def load_file_checkpoint(
    tier: str,
    resource_code: str,
    github_org: str,
    project: str,
    repo: str,
    file_path: str,
) -> dict | None:
    """
    Load the checkpoint JSON for a single source file.
    Returns None if this file has never been crawled.
    """
    url = _account_url(tier, resource_code)
    blob_path = _file_checkpoint_path(resource_code, github_org, project, repo, file_path)
    credential = DefaultAzureCredential()

    async with BlobServiceClient(url, credential) as client:
        blob = client.get_blob_client("checkpoints", blob_path)
        try:
            stream = await blob.download_blob()
            content = await stream.readall()
            return json.loads(content)
        except Exception:
            return None


async def save_file_checkpoint(
    tier: str,
    resource_code: str,
    github_org: str,
    project: str,
    repo: str,
    file_path: str,
    crawl_result: dict,
) -> None:
    """
    Save crawl output for a single source file to Storage.
    Called after crawl_file() returns a result.
    """
    url = _account_url(tier, resource_code)
    blob_path = _file_checkpoint_path(resource_code, github_org, project, repo, file_path)
    credential = DefaultAzureCredential()

    async with BlobServiceClient(url, credential) as client:
        blob = client.get_blob_client("checkpoints", blob_path)
        await blob.upload_blob(
            json.dumps(crawl_result, indent=2).encode("utf-8"),
            overwrite=True,
        )
