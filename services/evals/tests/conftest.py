import pytest
from pydantic import SecretStr


class _TestSettings:
    OPENAI_API_KEY = SecretStr("test-openai-key")
    ANTHROPIC_API_KEY = SecretStr("test-anthropic-key")
    GOOGLE_API_KEY = SecretStr("test-google-key")
    DEEPSEEK_API_KEY = SecretStr("test-deepseek-key")
    MOONSHOT_API_KEY = SecretStr("test-moonshot-key")


@pytest.fixture(scope="session", autouse=True)
def _test_secrets():
    from infrastructure import settings as eval_settings

    eval_settings.SETTINGS = _TestSettings()
    yield
    eval_settings.SETTINGS = None
