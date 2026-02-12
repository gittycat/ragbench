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
SUPERUSER="$(read_secret /run/secrets/POSTGRES_SUPERUSER)"
DB_NAME="${POSTGRES_DB:-postgres}"

RAG_USER_SQL="$(sql_escape "$RAG_USER")"

psql -v ON_ERROR_STOP=1 \
  -U "$SUPERUSER" \
  -d "$DB_NAME" \
  <<SQL
DO \$\$
DECLARE
  rag_user text := '${RAG_USER_SQL}';
BEGIN
  EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', rag_user);
  EXECUTE format('GRANT CREATE ON SCHEMA public TO %I', rag_user);
  EXECUTE format(
    'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %I',
    rag_user
  );
  EXECUTE format(
    'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I',
    rag_user
  );
  EXECUTE format(
    'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I',
    rag_user
  );
  EXECUTE format(
    'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO %I',
    rag_user
  );
END
\$\$;
SQL
