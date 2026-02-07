"""Tests for API key validation module."""

import pytest
from pathlib import Path
import sys
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.llm.validation import (
    validate_openai_key,
    validate_anthropic_key,
    validate_google_key,
    validate_deepseek_key,
    validate_moonshot_key,
    validate_api_key,
)


# ============================================================================
# OpenAI Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_openai_key_valid():
    """Valid OpenAI API key should return True"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_openai_key("sk-test-valid-key")

        assert valid is True
        assert error is None


@pytest.mark.asyncio
async def test_validate_openai_key_invalid():
    """Invalid OpenAI API key should return False with error message"""
    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_openai_key("sk-test-invalid-key")

        assert valid is False
        assert error == "Invalid API key"


@pytest.mark.asyncio
async def test_validate_openai_key_network_error():
    """Network error should return False with timeout message"""
    import httpx

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )

        valid, error = await validate_openai_key("sk-test-key")

        assert valid is False
        assert "timeout" in error.lower()


# ============================================================================
# Anthropic Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_anthropic_key_valid():
    """Valid Anthropic API key should return True"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_anthropic_key("sk-ant-test-valid")

        assert valid is True
        assert error is None


@pytest.mark.asyncio
async def test_validate_anthropic_key_invalid():
    """Invalid Anthropic API key should return False"""
    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_anthropic_key("sk-ant-invalid")

        assert valid is False
        assert error == "Invalid API key"


# ============================================================================
# Google Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_google_key_valid():
    """Valid Google API key should return True"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_google_key("AIza-test-valid")

        assert valid is True
        assert error is None


@pytest.mark.asyncio
async def test_validate_google_key_invalid():
    """Invalid Google API key should return False"""
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_google_key("AIza-invalid")

        assert valid is False
        assert error == "Invalid API key"


# ============================================================================
# DeepSeek Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_deepseek_key_valid():
    """Valid DeepSeek API key should return True"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_deepseek_key("sk-test-valid")

        assert valid is True
        assert error is None


@pytest.mark.asyncio
async def test_validate_deepseek_key_invalid():
    """Invalid DeepSeek API key should return False"""
    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_deepseek_key("sk-invalid")

        assert valid is False
        assert error == "Invalid API key"


# ============================================================================
# Moonshot Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_moonshot_key_valid():
    """Valid Moonshot API key should return True"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_moonshot_key("sk-test-valid")

        assert valid is True
        assert error is None


@pytest.mark.asyncio
async def test_validate_moonshot_key_invalid():
    """Invalid Moonshot API key should return False"""
    mock_response = MagicMock()
    mock_response.status_code = 401

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_moonshot_key("sk-invalid")

        assert valid is False
        assert error == "Invalid API key"


# ============================================================================
# Generic validate_api_key Tests
# ============================================================================


@pytest.mark.asyncio
async def test_validate_api_key_openai():
    """validate_api_key should route to OpenAI validator"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_api_key("openai", "sk-test")

        assert valid is True


@pytest.mark.asyncio
async def test_validate_api_key_anthropic():
    """validate_api_key should route to Anthropic validator"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_api_key("anthropic", "sk-ant-test")

        assert valid is True


@pytest.mark.asyncio
async def test_validate_api_key_unknown_provider():
    """validate_api_key should return False for unknown provider"""
    valid, error = await validate_api_key("unknown-provider", "test-key")

    assert valid is False
    assert "Unknown provider" in error


@pytest.mark.asyncio
async def test_validate_api_key_case_insensitive():
    """validate_api_key should be case-insensitive"""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        valid, error = await validate_api_key("OpenAI", "sk-test")

        assert valid is True
