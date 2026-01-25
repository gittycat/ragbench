"""Config display utilities for CLI tools."""

from infrastructure.config.models_config import get_models_config


def print_config_banner(compact: bool = True) -> None:
    """Print RAG configuration banner.

    Args:
        compact: If True, show minimal config. If False, show full config.
    """
    try:
        config = get_models_config()

        if compact:
            _print_compact_banner(config)
        else:
            _print_full_banner(config)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")


def _print_compact_banner(config) -> None:
    """Print compact config banner."""
    print("=" * 70)
    print("RAG Configuration")
    print("=" * 70)

    # LLM
    llm_info = f"{config.llm.provider}/{config.llm.model}"
    if config.llm.keep_alive:
        llm_info += f" (keep_alive={config.llm.keep_alive})"
    print(f"  LLM (inference):  {llm_info}")

    # Embedding
    embed_info = f"{config.embedding.provider}/{config.embedding.model}"
    print(f"  Embedding:        {embed_info}")

    # Reranker
    if config.reranker.enabled:
        rerank_info = f"{config.reranker.model} (top_n={config.reranker.top_n})"
        print(f"  Reranker:         {rerank_info}")
    else:
        print(f"  Reranker:         disabled")

    # Eval (if configured)
    if hasattr(config, 'eval') and config.eval:
        eval_info = f"{config.eval.provider}/{config.eval.model}"
        print(f"  Eval (judge):     {eval_info}")

    print("=" * 70)


def _print_full_banner(config) -> None:
    """Print full config banner with all settings."""
    print("=" * 70)
    print("RAG Configuration (Full)")
    print("=" * 70)

    # LLM section
    print("\nLLM (Inference):")
    print(f"  Provider:    {config.llm.provider}")
    print(f"  Model:       {config.llm.model}")
    if config.llm.base_url:
        print(f"  Base URL:    {config.llm.base_url}")
    print(f"  Timeout:     {config.llm.timeout}s")
    if config.llm.keep_alive:
        print(f"  Keep Alive:  {config.llm.keep_alive}")
    print(f"  API Key:     {'configured' if config.llm.api_key else 'not set'}")

    # Embedding section
    print("\nEmbedding:")
    print(f"  Provider:    {config.embedding.provider}")
    print(f"  Model:       {config.embedding.model}")
    if config.embedding.base_url:
        print(f"  Base URL:    {config.embedding.base_url}")

    # Reranker section
    print("\nReranker:")
    print(f"  Enabled:     {config.reranker.enabled}")
    if config.reranker.enabled:
        print(f"  Model:       {config.reranker.model}")
        print(f"  Top N:       {config.reranker.top_n}")

    # Retrieval section
    print("\nRetrieval:")
    print(f"  Top K:                      {config.retrieval.top_k}")
    print(f"  Hybrid Search:              {config.retrieval.enable_hybrid_search}")
    if config.retrieval.enable_hybrid_search:
        print(f"  RRF K:                      {config.retrieval.rrf_k}")
    print(f"  Contextual Retrieval:       {config.retrieval.enable_contextual_retrieval}")

    # Eval section (if configured)
    if hasattr(config, 'eval') and config.eval:
        print("\nEvaluation (LLM-as-Judge):")
        print(f"  Provider:         {config.eval.provider}")
        print(f"  Model:            {config.eval.model}")
        print(f"  Citation Scope:   {config.eval.citation_scope}")
        print(f"  Citation Format:  {config.eval.citation_format}")
        print(f"  API Key:          {'configured' if config.eval.api_key else 'not set'}")

    print("\n" + "=" * 70)
