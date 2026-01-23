"""
Session Title Generation

AI-powered and simple title generation for chat sessions.
"""

import logging

logger = logging.getLogger(__name__)


def generate_session_title(first_user_message: str, max_length: int = 50) -> str:
    """
    Generate session title from first user message (simple truncation fallback).

    Truncates to max_length and adds "..." if needed.
    """
    if not first_user_message:
        return "New Chat"

    # Clean whitespace
    title = first_user_message.strip()

    # Truncate if needed
    if len(title) > max_length:
        title = title[:max_length].strip() + "..."

    return title


def generate_ai_title(first_user_message: str, max_length: int = 40) -> str:
    """
    Generate a concise chat title using AI from the first user message.

    Uses the configured LLM to create a short, descriptive title.
    Falls back to simple truncation if AI generation fails.

    Args:
        first_user_message: The user's first message in the chat
        max_length: Maximum length of the generated title

    Returns:
        A concise title for the chat session
    """
    if not first_user_message or not first_user_message.strip():
        return "New Chat"

    try:
        from infrastructure.llm.factory import get_llm_client

        llm = get_llm_client()

        prompt = f"""Generate a very short title (3-6 words max) for a chat that starts with this message.
Reply with ONLY the title, no quotes, no explanation, no punctuation at the end.

User message: {first_user_message[:500]}

Title:"""

        response = llm.complete(prompt)
        title = response.text.strip()

        # Clean up: remove quotes, limit length
        title = title.strip('"\'')
        if len(title) > max_length:
            title = title[:max_length].strip() + "..."

        # Fallback if AI returned empty or nonsensical response
        if not title or len(title) < 3:
            return generate_session_title(first_user_message, max_length)

        logger.info(f"[SESSION] AI-generated title: {title}")
        return title

    except Exception as e:
        logger.warning(f"[SESSION] AI title generation failed: {e}, using fallback")
        return generate_session_title(first_user_message, max_length)
