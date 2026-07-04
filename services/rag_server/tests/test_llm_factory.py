from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.llm.config import LLMConfig, LLMProvider


def test_create_llm_client_dispatches_vllm():
    """vllm provider should dispatch to OpenAILike with mapped params, keyless by default."""
    from infrastructure.llm.factory import create_llm_client

    config = LLMConfig(
        provider=LLMProvider.VLLM,
        model="Qwen/Qwen2.5-14B-Instruct",
        base_url="http://vllm:8000/v1",
        timeout=120,
    )

    with patch("llama_index.llms.openai_like.OpenAILike") as mock_llm_class:
        mock_instance = MagicMock()
        mock_llm_class.return_value = mock_instance

        client = create_llm_client(config)
        assert client is not None

        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model"] == "Qwen/Qwen2.5-14B-Instruct"
        assert call_kwargs["api_base"] == "http://vllm:8000/v1"
        assert call_kwargs["api_key"] == "none"
        assert call_kwargs["is_chat_model"] is True


def test_create_llm_client_vllm_respects_explicit_api_key():
    """When vLLM is deployed with an api key, it should be passed through unchanged."""
    from infrastructure.llm.factory import create_llm_client

    config = LLMConfig(
        provider=LLMProvider.VLLM,
        model="Qwen/Qwen2.5-14B-Instruct",
        base_url="http://vllm:8000/v1",
        api_key="vllm-secret",
        timeout=120,
    )

    with patch("llama_index.llms.openai_like.OpenAILike") as mock_llm_class:
        mock_instance = MagicMock()
        mock_llm_class.return_value = mock_instance

        create_llm_client(config)

        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["api_key"] == "vllm-secret"
        assert call_kwargs["is_chat_model"] is True
