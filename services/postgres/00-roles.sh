#!/usr/bin/env bash
set -euo pipefail

read_secret() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing secret file: $path" >&2
    exit 1
  fi
  tr -d '\n' < "$path"
}

sql_escape() {
  local s="$1"
  s="${s//\'/\'\'}"
  printf "%s" "$s"
}

RAG_USER="$(read_secret /run/secrets/RAG_SERVER_DB_USER)"
RAG_PASS="$(read_secret /run/secrets/RAG_SERVER_DB_PASSWORD)"
SUPERUSER="$(read_secret /run/secrets/POSTGRES_SUPERUSER)"
DB_NAME="${POSTGRES_DB:-postgres}"

RAG_USER_SQL="$(sql_escape "$RAG_USER")"
RAG_PASS_SQL="$(sql_escape "$RAG_PASS")"

psql -v ON_ERROR_STOP=1 \
  -U "$SUPERUSER" \
  -d "$DB_NAME" \
  <<SQL
DO \$\$
DECLARE
  rag_user text := '${RAG_USER_SQL}';
  rag_pass text := '${RAG_PASS_SQL}';
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = rag_user) THEN
    EXECUTE format(
      'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT',
      rag_user,
      rag_pass
    );
  ELSE
    EXECUTE format(
      'ALTER ROLE %I WITH LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT',
      rag_user,
      rag_pass
    );
  END IF;

END
\$\$;
SQL
