# Secrets Directory

This directory contains sensitive API keys and credentials.

## Setup

Create `secrets/.env` from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# For cloud LLM providers (not needed for Ollama)
LLM_API_KEY=sk-...

# For DeepEval evaluations (required)
ANTHROPIC_API_KEY=sk-ant-...
```

**Note:** Ollama configuration (URL, keep-alive) is now in `config.yml` per model, not in secrets.

## Getting API Keys

### Anthropic (Required for Evaluations)
1. Sign up at https://console.anthropic.com/
2. Navigate to API Keys section
3. Create new key
4. Add to `ANTHROPIC_API_KEY` in `.env`

### OpenAI
1. Sign up at https://platform.openai.com/
2. Navigate to API Keys
3. Create new key
4. Add to `LLM_API_KEY` in `.env`
5. Update `config.yml` to use OpenAI provider

## Security Notes

- **Never commit `.env` files to git** - they're in `.gitignore`
- Store production keys securely (use secret managers in production)
- Rotate keys regularly
- Use different keys for development and production
- The `.env.example` files are safe to commit (no secrets)

## Docker Integration

Environment variables from `.env` are loaded by `docker-compose.yml`:
```yaml
environment:
  - LLM_API_KEY=${LLM_API_KEY:-}
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
```

## Troubleshooting

### "API key required" Error
- Verify API key is set in `.env`
- Restart services: `docker compose down && docker compose up -d`
- Check key format matches provider requirements

### Ollama Connection Failed
- Ensure Ollama is running on host: `ollama list`
- Verify `base_url` in `config.yml` for Ollama models
- Check Docker can reach host: `docker compose exec rag-server curl http://host.docker.internal:11434/api/tags`

### Environment Variables Not Loading
- Ensure `.env` file exists in `secrets/` directory
- Check file permissions (should be readable)
- Restart Docker Compose to reload environment
