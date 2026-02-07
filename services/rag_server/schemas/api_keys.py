from pydantic import BaseModel


class ApiKeyStatus(BaseModel):
    """Status of an API key for a provider."""

    provider: str
    has_key: bool
    masked_key: str | None


class ApiKeySetRequest(BaseModel):
    """Request to set an API key."""

    api_key: str


class ApiKeySetResponse(BaseModel):
    """Response after setting an API key."""

    provider: str
    status: str
    masked_key: str
