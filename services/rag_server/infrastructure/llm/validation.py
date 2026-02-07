"""API key validation for LLM providers."""

import httpx
from typing import Tuple


async def validate_openai_key(api_key: str) -> Tuple[bool, str | None]:
    """Validate OpenAI API key by calling the models endpoint.

    Returns:
        (valid, error_message) tuple
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code == 200:
                return True, None
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"Validation failed: HTTP {response.status_code}"
    except httpx.TimeoutException:
        return False, "Validation timeout - please try again"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def validate_anthropic_key(api_key: str) -> Tuple[bool, str | None]:
    """Validate Anthropic API key by calling the models endpoint.

    Returns:
        (valid, error_message) tuple
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )

            if response.status_code == 200:
                return True, None
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"Validation failed: HTTP {response.status_code}"
    except httpx.TimeoutException:
        return False, "Validation timeout - please try again"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def validate_google_key(api_key: str) -> Tuple[bool, str | None]:
    """Validate Google API key by calling the models endpoint.

    Returns:
        (valid, error_message) tuple
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            )

            if response.status_code == 200:
                return True, None
            elif response.status_code in (400, 403):
                return False, "Invalid API key"
            else:
                return False, f"Validation failed: HTTP {response.status_code}"
    except httpx.TimeoutException:
        return False, "Validation timeout - please try again"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def validate_deepseek_key(api_key: str) -> Tuple[bool, str | None]:
    """Validate DeepSeek API key by calling the models endpoint.

    Returns:
        (valid, error_message) tuple
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.deepseek.com/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code == 200:
                return True, None
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"Validation failed: HTTP {response.status_code}"
    except httpx.TimeoutException:
        return False, "Validation timeout - please try again"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def validate_moonshot_key(api_key: str) -> Tuple[bool, str | None]:
    """Validate Moonshot API key by calling the models endpoint.

    Returns:
        (valid, error_message) tuple
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.moonshot.cn/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )

            if response.status_code == 200:
                return True, None
            elif response.status_code == 401:
                return False, "Invalid API key"
            else:
                return False, f"Validation failed: HTTP {response.status_code}"
    except httpx.TimeoutException:
        return False, "Validation timeout - please try again"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


async def validate_api_key(provider: str, api_key: str) -> Tuple[bool, str | None]:
    """Validate an API key for the given provider.

    Args:
        provider: Provider name (openai, anthropic, google, deepseek, moonshot)
        api_key: API key to validate

    Returns:
        (valid, error_message) tuple
    """
    provider_lower = provider.lower()

    validators = {
        "openai": validate_openai_key,
        "anthropic": validate_anthropic_key,
        "google": validate_google_key,
        "deepseek": validate_deepseek_key,
        "moonshot": validate_moonshot_key,
    }

    validator = validators.get(provider_lower)
    if not validator:
        return False, f"Unknown provider: {provider}"

    return await validator(api_key)
