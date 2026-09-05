"""
GitHub service — all GitHub API interactions for the indexing worker.

Functions:
  _build_jwt()              — sign a JWT using the App private key
  get_installation_token()  — exchange JWT for a short-lived installation token
  get_repo_file_tree()      — fetch all file paths and SHAs in a repo
  get_file_content()        — fetch raw content of a single file

Authentication flow:
  private_key (PEM from Key Vault)
      ↓
  _build_jwt(app_id, private_key)   → JWT (valid 10 minutes)
      ↓
  get_installation_token(...)        → installation token (valid 1 hour)
      ↓
  get_repo_file_tree() / get_file_content()
"""
import time
import base64
import os

import httpx
import jwt as pyjwt


GITHUB_API = "https://api.github.com"


def _build_jwt(app_id: str, private_key: str) -> str:
    """
    Build a signed JWT for GitHub App authentication.

    GitHub requires a JWT signed with the App's private key to request
    installation tokens. The JWT is valid for 10 minutes — enough to
    exchange it for an installation token.
    """
    now = int(time.time())
    payload = {
        "iat": now - 60,       # issued at (60s in past to allow clock skew)
        "exp": now + (10 * 60), # expires in 10 minutes
        "iss": app_id,          # issuer = GitHub App ID
    }
    return pyjwt.encode(payload, private_key, algorithm="RS256")


async def get_installation_token(
    installation_id: str,
    app_id: str,
    private_key: str,
) -> str:
    """
    Exchange a GitHub App JWT for a short-lived installation token.

    The installation token is valid for 1 hour and grants access to
    all repos the tenant has approved for this App installation.
    """
    jwt_token = _build_jwt(app_id, private_key)

    url = f"{GITHUB_API}/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept":        "application/vnd.github+json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers)
        response.raise_for_status()
        return response.json()["token"]


async def get_repo_file_tree(org: str, repo: str, token: str) -> list[dict]:
    """
    Fetch the full recursive file tree for a repo.

    Returns a list of dicts with 'path' and 'sha' for every file.
    The SHA is used for incremental change detection — if SHA matches
    the checkpoint, the file has not changed since last index run.

    Only returns blobs (files), not trees (directories).
    """
    url = f"{GITHUB_API}/repos/{org}/{repo}/git/trees/HEAD"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
    }
    params = {"recursive": "1"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        tree = response.json().get("tree", [])

    # return only files (type=blob), not directories (type=tree)
    return [
        {"path": item["path"], "sha": item["sha"]}
        for item in tree
        if item["type"] == "blob"
    ]


async def get_file_content(org: str, repo: str, path: str, token: str) -> str:
    """
    Fetch the raw text content of a single file from GitHub.

    GitHub returns file content as base64-encoded string.
    We decode it and return the raw text so crawlers can parse it.
    """
    url = f"{GITHUB_API}/repos/{org}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        encoded = response.json()["content"]

    # GitHub returns content as base64 with newlines — strip before decoding
    return base64.b64decode(encoded.replace("\n", "")).decode("utf-8")
