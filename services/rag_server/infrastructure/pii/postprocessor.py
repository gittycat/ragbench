"""LlamaIndex node postprocessor that masks retrieved chunk text before it
reaches the cloud generation LLM, plus the session-scoped TokenMapping cache
shared across query, retrieved context, and chat history.

Ordering requirement: this MUST run after the reranker. The reranker scores
on-host and needs the original text for quality; masking is the last step
before the prompt is assembled.
"""

from typing import Any, List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from .service import TokenMapping, mask_text


class PIIMaskingPostprocessor(BaseNodePostprocessor):
    """Masks PII in retrieved node text. Never mutates docstore nodes — works on copies."""

    token_mapping: Any
    context_id: Optional[str] = None

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        masked_nodes = []
        for nws in nodes:
            result = mask_text(nws.node.get_content(), existing_mapping=self.token_mapping, context_id=self.context_id)
            masked = TextNode(text=result.masked_text, metadata=nws.node.metadata)
            masked_nodes.append(NodeWithScore(node=masked, score=nws.score))
        return masked_nodes


# Session-scoped mapping cache — [[[PERSON_0]]] must refer to the same person
# across turns, since condense_plus_context re-sends chat history to the LLM
# every request. In-memory only; never persist original PII values to disk.
_session_mappings: dict[str, TokenMapping] = {}


def get_session_token_mapping(session_id: str) -> TokenMapping:
    if session_id not in _session_mappings:
        _session_mappings[session_id] = TokenMapping()
    return _session_mappings[session_id]


def clear_session_token_mapping(session_id: str) -> None:
    _session_mappings.pop(session_id, None)
