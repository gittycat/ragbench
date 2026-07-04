"""PII masking configuration accessor.

The config schema itself lives on ModelsConfig (infrastructure/config/models_config.py),
alongside every other model/retrieval setting. This module just re-exports the
piece the PII service needs, so callers can `from infrastructure.pii.config import
get_pii_config` without reaching into models_config directly.
"""

from infrastructure.config.models_config import PiiConfig, get_models_config


def get_pii_config() -> PiiConfig:
    """Get PII config from the global models config."""
    return get_models_config().pii
