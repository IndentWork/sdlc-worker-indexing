"""
SDLC Indexing Worker — listens to Service Bus and indexes tenant repos.

Flow:
  1. Receive message (action=upload_sdlc) from Service Bus
  2. Read sdlc.yml from Storage → list of repos
  3. Get GitHub App installation token
  4. For each repo in parallel:
     a. Fetch file tree from GitHub API
     b. Load checkpoint from Storage
     c. Crawl changed files (SHA comparison)
     d. Chunk → Azure AI Search
     e. Load graph → Cosmos DB
     f. Save updated checkpoint

Environment variables required:
  SERVICEBUS_NAMESPACE  — e.g. sb-sdlc-shared-dev.servicebus.windows.net
  AZURE_CLIENT_ID       — Managed Identity client ID
  GITHUB_APP_ID         — GitHub App ID (4826692)
  KEY_VAULT_URL         — e.g. https://kv-sdlc-base-dev.vault.azure.net
  ENV                   — dev or prod
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import yaml

from azure.identity.aio import DefaultAzureCredential
from azure.servicebus.aio import ServiceBusClient

from app.indexing.chunker import chunk_file
from app.indexing.crawlers.registry import get_crawler, supported_extensions
from app.indexing.graph_loader import load_edges, load_nodes
from app.services.ai_search import ensure_index_exists
from app.services.github import get_file_content, get_installation_token, get_org_installation_id, get_repo_file_tree
from app.services.keyvault import get_github_app_private_key
from app.services.storage import (
    load_file_checkpoint,
    load_project_index,
    save_file_checkpoint,
    save_project_index,
)

TOPIC_NAME        = "sdlc-events"
SUBSCRIPTION_NAME = "indexing"
SQL_FILTER        = "action = 'upload_sdlc'"


# ── Logging ───────────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return logging.getLogger("worker.indexing")


log = _setup_logging()


# ── Indexing pipeline ─────────────────────────────────────────────────────────

async def _index_file(
    file: dict,
    token: str,
    github_org: str,
    repo: str,
    checkpoint_shas: dict,
    resource_code: str,
    github_org_: str,
    project: str,
    tier: str,
) -> dict | None:
    """
    Index one file — crawl, chunk, load graph.
    Returns crawl result if file was processed, None if skipped (unchanged SHA).
    """
    file_path = file["path"]
    github_sha = file["sha"]

    # skip if SHA unchanged since last run
    if checkpoint_shas.get(file_path) == github_sha:
        return None

    # only process supported file types
    crawler = get_crawler(file_path)
    if crawler is None:
        return None

    content = await get_file_content(github_org, repo, file_path, token)
    result  = crawler.crawl(content, file_path, github_sha)

    await chunk_file(result, resource_code, github_org_, project, repo)
    await load_nodes(result, resource_code, github_org_, project, repo)
    await load_edges(result, resource_code, github_org_, project, repo)
    await save_file_checkpoint(tier, resource_code, github_org_, project, repo, file_path, result)

    return result


async def _index_repo(
    repo: str,
    github_org: str,
    project: str,
    token: str,
    resource_code: str,
    tier: str,
) -> None:
    """Index all changed files in one repo."""
    log.info(json.dumps({"event": "repo_started", "repo": repo}))

    # load checkpoint to know which file SHAs we already have
    index = await load_project_index(tier, resource_code, github_org, project, repo)
    checkpoint_shas = {f["path"]: f["sha"] for f in (index or {}).get("files", [])}

    # fetch current file tree from GitHub
    all_files = await get_repo_file_tree(github_org, repo, token)

    # index each file (skip unchanged)
    tasks = [
        _index_file(file, token, github_org, repo, checkpoint_shas, resource_code, github_org, project, tier)
        for file in all_files
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = [r for r in results if r and not isinstance(r, Exception)]
    errors    = [r for r in results if isinstance(r, Exception)]

    for err in errors:
        log.error(json.dumps({"event": "file_index_failed", "error": str(err)}))

    # save updated checkpoint with current SHAs for all files
    updated_index = {
        "resource_code": resource_code,
        "github_org":    github_org,
        "project":       project,
        "repo":          repo,
        "crawled_at":    datetime.now(timezone.utc).isoformat(),
        "files":         [{"path": f["path"], "sha": f["sha"]} for f in all_files],
    }
    await save_project_index(tier, resource_code, github_org, project, repo, updated_index)

    log.info(json.dumps({
        "event":     "repo_complete",
        "repo":      repo,
        "processed": len(processed),
        "skipped":   len(all_files) - len(processed) - len(errors),
        "errors":    len(errors),
    }))


async def _run_indexing_pipeline(payload: dict) -> None:
    """
    Full indexing pipeline for one message.
    Reads sdlc.yml, gets GitHub token, indexes all repos in parallel.
    """
    resource_code = payload["resource_code"]
    tier          = payload["tier"]

    # read sdlc.yml from Storage
    from app.services.storage import load_file_checkpoint as _load
    from azure.identity.aio import DefaultAzureCredential as _Cred
    from azure.storage.blob.aio import BlobServiceClient as _BlobClient

    env = os.environ.get("ENV", "dev")
    account = f"stsdlcshared{env}" if tier == "shared" else f"stsdlc{resource_code}{env}"
    url     = f"https://{account}.blob.core.windows.net"

    cred = DefaultAzureCredential()
    async with _BlobClient(url, cred) as blob_client:
        blob = blob_client.get_blob_client("configs", f"{resource_code}/sdlc.yml")
        stream  = await blob.download_blob()
        content = await stream.readall()

    config = yaml.safe_load(content)
    github_org = config.get("github_org", "")
    project    = config.get("project", "")
    repos      = config.get("repos", [])

    log.info(json.dumps({
        "event":      "pipeline_started",
        "github_org": github_org,
        "project":    project,
        "repos":      repos,
    }))

    # get GitHub App installation token — look up installation_id from org name
    app_id          = os.environ["GITHUB_APP_ID"]
    private_key     = await get_github_app_private_key()
    installation_id = await get_org_installation_id(github_org, app_id, private_key)
    token           = await get_installation_token(installation_id, app_id, private_key)

    # index all repos in parallel
    await asyncio.gather(*[
        _index_repo(repo, github_org, project, token, resource_code, tier)
        for repo in repos
    ])

    log.info(json.dumps({"event": "pipeline_complete", "repos": len(repos)}))


# ── Service Bus listener ──────────────────────────────────────────────────────

async def listen() -> None:
    """
    Long-running loop — receives messages from indexing subscription.
    Creates the subscription on first run if it doesn't exist.
    """
    namespace = os.environ["SERVICEBUS_NAMESPACE"]
    credential = DefaultAzureCredential()

    log.info(json.dumps({
        "event":        "worker_started",
        "namespace":    namespace,
        "topic":        TOPIC_NAME,
        "subscription": SUBSCRIPTION_NAME,
        "supports":     supported_extensions(),
    }))

    # ensure AI Search index exists before processing any messages
    await ensure_index_exists()

    async with ServiceBusClient(namespace, credential) as client:
        try:
            await client.create_subscription(
                TOPIC_NAME,
                SUBSCRIPTION_NAME,
                sql_filter=SQL_FILTER,
                exists_ok=True,
            )
        except Exception as exc:
            log.warning(json.dumps({
                "event": "subscription_create_warning",
                "error": str(exc),
            }))

        async with client.get_subscription_receiver(TOPIC_NAME, SUBSCRIPTION_NAME) as receiver:
            async for message in receiver:
                try:
                    payload = json.loads(str(message))
                    log.info(json.dumps({
                        "event":   "message_received",
                        "action":  payload.get("action"),
                        "resource_code": payload.get("resource_code"),
                    }))

                    await _run_indexing_pipeline(payload)
                    await receiver.complete_message(message)

                except Exception as exc:
                    log.error(json.dumps({
                        "event": "message_failed",
                        "error": str(exc),
                        "type":  type(exc).__name__,
                    }))
                    await receiver.abandon_message(message)


if __name__ == "__main__":
    asyncio.run(listen())
