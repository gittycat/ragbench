from pydantic import BaseModel, Field


class SettingsResponse(BaseModel):
    """Current toggleable settings."""

    contextual_retrieval_enabled: bool = Field(
        ..., description="Whether contextual retrieval is enabled for document ingestion"
    )


class SettingsUpdate(BaseModel):
    """Partial settings update. Only provided fields are updated."""

    contextual_retrieval_enabled: bool | None = Field(
        None, description="Enable or disable contextual retrieval"
    )
