"""
Centralized prompt management for RAG pipeline.

All prompts are loaded from config.yml for easy customization.
"""
from typing import Optional
from infrastructure.config.models_config import get_models_config


def get_system_prompt() -> str:
    """
    System-level instructions for LLM behavior.

    Defines the assistant's personality, response style, and general approach.
    Applied to all LLM interactions including question condensation and answer generation.
    """
    config = get_models_config()
    return config.prompts.system


def get_context_prompt(
    include_citations: bool = False,
    citation_format: str = "numeric",
) -> str:
    """
    Instructions for using retrieved context to answer questions.

    Specifies strict grounding rules to prevent hallucination and ensure
    answers are based only on the provided document context.

    Placeholders (filled by LlamaIndex):
        {context_str}: Retrieved document chunks
        {chat_history}: Previous conversation messages
    """
    config = get_models_config()

    citation_instructions = ""
    if include_citations:
        if citation_format == "numeric":
            citation_instructions = config.prompts.citation_instructions.numeric
        # Add more citation formats here if needed in the future

    return config.prompts.context.format(
        context_str="{context_str}",  # Keep as placeholder for LlamaIndex
        citation_instructions=citation_instructions,
    )


def get_condense_prompt() -> Optional[str]:
    """
    Optional: Custom question condensation prompt.

    Controls how follow-up questions are reformulated into standalone queries.
    Returns None to use LlamaIndex's built-in DEFAULT_CONDENSE_PROMPT.

    The default prompt works well for most cases:
    "Given a conversation (between Human and Assistant) and a follow up message from Human,
    rewrite the message to be a standalone question that captures all relevant context."

    Only customize if you need different reformulation behavior.
    """
    config = get_models_config()
    return config.prompts.condense


def get_contextual_prefix_prompt(
    document_name: str, document_type: str, chunk_preview: str
) -> str:
    """
    Get prompt for generating contextual prefix for a document chunk.

    Used in contextual retrieval (Anthropic method) to generate context
    that is prepended to chunks before embedding.

    Args:
        document_name: Name of the source document
        document_type: File extension or document type
        chunk_preview: Preview of the chunk content (typically first 400 chars)

    Returns:
        Formatted prompt for LLM to generate contextual prefix
    """
    config = get_models_config()
    return config.prompts.contextual_prefix.format(
        document_name=document_name,
        document_type=document_type,
        chunk_preview=chunk_preview,
    )
