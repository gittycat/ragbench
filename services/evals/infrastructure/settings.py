from typing import Optional

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


def get_api_key_for_provider(provider: str) -> str | None:
    provider_key = provider.lower()
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
