"""Task processing for async document ingestion via SKIP LOCKED."""

from infrastructure.tasks.worker import process_document_async

__all__ = ["process_document_async"]
