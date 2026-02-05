from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets",
        secrets_dir_missing="error",
        case_sensitive=True,
    )

    OPENAI_API_KEY: SecretStr
    ANTHROPIC_API_KEY: SecretStr
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


def init_settings() -> Settings:
    global SETTINGS
    if SETTINGS is None:
        SETTINGS = Settings()
    return SETTINGS


def get_openai_key() -> str:
    s = init_settings()
    return s.OPENAI_API_KEY.get_secret_value()


def get_anthropic_key() -> str:
    s = init_settings()
    return s.ANTHROPIC_API_KEY.get_secret_value()


def get_google_key() -> str | None:
    s = init_settings()
    if s.GOOGLE_API_KEY is None:
        return None
    return s.GOOGLE_API_KEY.get_secret_value()


def get_deepseek_key() -> str | None:
    s = init_settings()
    if s.DEEPSEEK_API_KEY is None:
        return None
    return s.DEEPSEEK_API_KEY.get_secret_value()


def get_moonshot_key() -> str | None:
    s = init_settings()
    if s.MOONSHOT_API_KEY is None:
        return None
    return s.MOONSHOT_API_KEY.get_secret_value()


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


def has_anthropic_key() -> bool:
    try:
        return bool(get_anthropic_key())
    except Exception:
        return False
