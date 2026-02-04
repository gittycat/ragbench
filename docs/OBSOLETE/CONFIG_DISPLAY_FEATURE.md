# Config Display Feature

## Overview

This feature adds informational banners to CLI tools showing the RAG configuration, including LLM models used for inference, embedding, ranking, and evaluation.

## Components

### 1. Config Display Module

**File:** `services/rag_server/infrastructure/config/display.py`

Provides two functions:
- `print_config_banner(compact=True)`: Displays config banner
  - `compact=True`: Shows minimal config (LLM, embedding, reranker, eval)
  - `compact=False`: Shows full config with all settings

### 2. Justfile Tasks

**Location:** `justfile`

```bash
# Show RAG configuration (compact)
just show-config

# Show full RAG configuration with all settings
just show-config-full
```

These tasks are also automatically run before:
- `just test-eval`
- `just test-eval-full`
- `just bench-run`

### 3. CLI Integration

**File:** `services/rag_server/evals/cli.py`

The config banner is automatically displayed at the start of:
- `python -m evals.cli eval`
- `python -m evals.cli stats`
- `python -m evals.cli datasets`

## Setup

Before using the config display feature, you need to create the config file:

```bash
cp config/models.yml.example config/models.yml
```

Then edit `config/models.yml` to match your setup.

## Usage Examples

### Show Config via Justfile

```bash
# Compact view
$ just show-config
======================================================================
RAG Configuration
======================================================================
  LLM (inference):  ollama/gemma3:4b (keep_alive=10m)
  Embedding:        ollama/nomic-embed-text:latest
  Reranker:         cross-encoder/ms-marco-MiniLM-L-6-v2 (top_n=5)
  Eval (judge):     anthropic/claude-sonnet-4-20250514
======================================================================

# Full view
$ just show-config-full
======================================================================
RAG Configuration (Full)
======================================================================

LLM (Inference):
  Provider:    ollama
  Model:       gemma3:4b
  Base URL:    http://host.docker.internal:11434
  Timeout:     120s
  Keep Alive:  10m
  API Key:     not set

Embedding:
  Provider:    ollama
  Model:       nomic-embed-text:latest
  Base URL:    http://host.docker.internal:11434

Reranker:
  Enabled:     True
  Model:       cross-encoder/ms-marco-MiniLM-L-6-v2
  Top N:       5

Retrieval:
  Top K:                      10
  Hybrid Search:              True
  RRF K:                      60
  Contextual Retrieval:       False

Evaluation (LLM-as-Judge):
  Provider:         anthropic
  Model:            claude-sonnet-4-20250514
  Citation Scope:   retrieved
  Citation Format:  numeric
  API Key:          configured

======================================================================
```

### Auto-Display in CLI Tools

When running evaluation commands, the config banner is automatically displayed:

```bash
$ python -m evals.cli eval --samples 5
======================================================================
RAG Configuration
======================================================================
  LLM (inference):  ollama/gemma3:4b (keep_alive=10m)
  Embedding:        ollama/nomic-embed-text:latest
  Reranker:         cross-encoder/ms-marco-MiniLM-L-6-v2 (top_n=5)
  Eval (judge):     anthropic/claude-sonnet-4-20250514
======================================================================

Datasets: ['ragbench']
Samples per dataset: 5
RAG server: http://localhost:8001
Judge enabled: True
------------------------------------------------------------
...
```

## Benefits

1. **Transparency**: Users immediately see which models are being used
2. **Debugging**: Helps verify correct configuration before long-running evals
3. **Documentation**: Config snapshots in logs for reproducibility
4. **Quick Reference**: No need to open config files to check settings

## Configuration Source

All settings are loaded from `config/models.yml`. See `config/models.yml.example` for the template.

## API Keys

For security, API keys are never displayed in full:
- Shows "configured" if present
- Shows "not set" if missing

## Error Handling

If config loading fails, a warning is displayed instead of crashing:

```
Warning: Failed to load config: models.yml not found in standard locations: [...]
```

## Future Enhancements

Potential additions:
- Add to other CLI tools (e.g., document processing scripts)
- JSON output format for programmatic access
- Environment variable override indicators
- Model version/size information
