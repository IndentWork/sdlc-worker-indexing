"""
Key Vault service — reads secrets from Azure Key Vault at startup.

The GitHub App private key is a multi-line PEM file stored in Key Vault.
It is read once at startup and reused for all GitHub token requests.
Authentication uses DefaultAzureCredential (Managed Identity in Azure).
"""
import os

from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

# Secret names in Key Vault
GITHUB_APP_PRIVATE_KEY_SECRET = "github-app-private-key"


def _vault_url() -> str:
    """Derive Key Vault URL from environment."""
    env = os.environ.get("ENV", "dev")
    return os.environ.get("KEY_VAULT_URL", f"https://kv-sdlc-base-{env}.vault.azure.net")


async def get_github_app_private_key() -> str:
    """
    Read the GitHub App private key PEM from Key Vault.

    Called once at worker startup. The returned PEM string is used to
    sign JWTs for GitHub App authentication.
    """
    credential = DefaultAzureCredential()

    async with SecretClient(_vault_url(), credential) as client:
        secret = await client.get_secret(GITHUB_APP_PRIVATE_KEY_SECRET)
        return secret.value
