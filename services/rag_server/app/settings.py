from typing import Optional
from urllib.parse import quote_plus

import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets",
        secrets_dir_missing="warn",
        case_sensitive=True,
    )

    OPENAI_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    MOONSHOT_API_KEY: SecretStr | None = None
    RAG_SERVER_DB_USER: SecretStr | None = None
    RAG_SERVER_DB_PASSWORD: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (file_secret_settings,)


SETTINGS: Optional[Settings] = None

# Runtime API key store (takes precedence over file-based keys)
_runtime_api_keys: dict[str, str] = {}


def _sanitize_secret(value: str) -> str:
    # Trim whitespace and remove null bytes commonly present in mounted secret files.
    return value.strip().replace("\x00", "")


def init_settings() -> Settings:
    global SETTINGS
    if SETTINGS is None:
        SETTINGS = Settings()
    return SETTINGS


def get_openai_key() -> str:
    s = init_settings()
    if s.OPENAI_API_KEY is None:
        return ""
    return _sanitize_secret(s.OPENAI_API_KEY.get_secret_value())


def get_anthropic_key() -> str:
    s = init_settings()
    if s.ANTHROPIC_API_KEY is None:
        return ""
    return _sanitize_secret(s.ANTHROPIC_API_KEY.get_secret_value())


def get_google_key() -> str | None:
    s = init_settings()
    if s.GOOGLE_API_KEY is None:
        return None
    value = _sanitize_secret(s.GOOGLE_API_KEY.get_secret_value())
    return value or None


def get_deepseek_key() -> str | None:
    s = init_settings()
    if s.DEEPSEEK_API_KEY is None:
        return None
    value = _sanitize_secret(s.DEEPSEEK_API_KEY.get_secret_value())
    return value or None


def get_moonshot_key() -> str | None:
    s = init_settings()
    if s.MOONSHOT_API_KEY is None:
        return None
    value = _sanitize_secret(s.MOONSHOT_API_KEY.get_secret_value())
    return value or None


def set_runtime_api_key(provider: str, api_key: str) -> None:
    """Store an API key in runtime memory (takes precedence over file-based keys)."""
    _runtime_api_keys[provider.lower()] = api_key


def get_api_key_for_provider(provider: str) -> str | None:
    """Get API key for provider. Checks runtime store first, then file-based keys."""
    provider_key = provider.lower()

    # Check runtime store first
    if provider_key in _runtime_api_keys:
        return _runtime_api_keys[provider_key]

    # Fall back to file-based keys
    if provider_key == "openai":
        return get_openai_key()
    if provider_key == "anthropic":
        return get_anthropic_key()
    if provider_key == "google":
        return get_google_key()
    if provider_key == "deepseek":
        return get_deepseek_key()
    if provider_key == "moonshot":
        return get_moonshot_key()
    return None


def has_anthropic_key() -> bool:
    try:
        return bool(get_anthropic_key())
    except Exception:
        return False


def get_postgres_user() -> str:
    s = init_settings()
    if s.RAG_SERVER_DB_USER is None:
        raise RuntimeError("RAG_SERVER_DB_USER secret not mounted at /run/secrets/")
    return _sanitize_secret(s.RAG_SERVER_DB_USER.get_secret_value())


def get_postgres_password() -> str:
    s = init_settings()
    if s.RAG_SERVER_DB_PASSWORD is None:
        raise RuntimeError("RAG_SERVER_DB_PASSWORD secret not mounted at /run/secrets/")
    return _sanitize_secret(s.RAG_SERVER_DB_PASSWORD.get_secret_value())


def get_database_host() -> str:
    return os.getenv("DATABASE_HOST", "postgres")


def get_database_port() -> int:
    return int(os.getenv("DATABASE_PORT", "5432"))


def get_database_name() -> str:
    return os.getenv("DATABASE_NAME", "ragbench")


def get_database_url() -> str:
    user = quote_plus(get_postgres_user())
    password = quote_plus(get_postgres_password())
    host = get_database_host()
    port = get_database_port()
    db = get_database_name()
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


def get_database_params() -> dict[str, str]:
    return {
        "database": get_database_name(),
        "host": get_database_host(),
        "port": str(get_database_port()),
        "user": get_postgres_user(),
        "password": get_postgres_password(),
    }


def get_shared_upload_dir() -> str:
    return os.getenv("SHARED_UPLOAD_DIR", "/tmp/shared")
