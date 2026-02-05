# Secrets Directory

Store passwords, api keys and other credentials in the `secrets/` dir.

**Environment Vars NOT USED**
We do not store credentials as environment variables.
Env variables can leak through application logging, metrics tracing, docker inspect, proc introspection and child processes inheritance, core dump to name a few.

Reading from a mounted file is the current best practice as per the OWASP Secrets Management and Cryptographic Storage cheat sheets.


**Rules:**
- One secret per file.
- The file name is the key. The content of the file is the value.
  Eg: /secrets/OPENAI_API_KEY   => content: sjk-aabbcc112233...   (no comments, just one line)
- Add /secrets to both `.gitignore` AND AI deny rules (eg: in .claude/settings.json)
- Use docker compose secrets to mount the files. This avoids exposing the entire /secrets dir content on mount.

## Required Secrets

Create a file for each of these keys in `secrets/`:
- `POSTGRES_SUPERUSER`
- `POSTGRES_SUPERPASSWORD`
- `RAG_SERVER_DB_USER`
- `RAG_SERVER_DB_PASSWORD`
- `OPENAI_API_KEY` (only if using OpenAI)
- `ANTHROPIC_API_KEY` (only if using Anthropic)

## Getting API Keys

### Anthropic (Required for Evaluations)
1. Sign up at https://console.anthropic.com/
2. Navigate to API Keys section
3. Create new key
4. Add `ANTHROPIC_API_KEY` to `secrets/`
5. Update `config.yml` to use OpenAI provider

### OpenAI
1. Sign up at https://platform.openai.com/
2. Navigate to API Keys
3. Create new key
4. Add `OPENAI_API_KEY` to `secrets/`
5. Update `config.yml` to use OpenAI provider

## Security Notes

- **Never commit `.env` files to git** - they're in `.gitignore`
- Store production keys securely (use secret managers in production)
- Rotate keys regularly
- Use different keys for development and production
