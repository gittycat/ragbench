# PostgreSQL with pgvector + pg_search (ParadeDB) + pgmq
# Base: ParadeDB includes pgvector and pg_search
# Added: pgmq for message queue (installed via PGXN)

FROM paradedb/paradedb:latest

# Install pgmq via PGXN (PostgreSQL Extension Network)
# pgmq is pure PL/pgSQL, no compilation needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pgxnclient \
        build-essential \
        postgresql-server-dev-17 && \
    pgxn install pgmq && \
    apt-get purge -y --auto-remove pgxnclient build-essential postgresql-server-dev-17 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
