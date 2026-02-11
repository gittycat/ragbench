# PostgreSQL 17 with pg_textsearch (BM25) + pgmq (message queue)
FROM postgres:17

# Install pg_textsearch v0.5.0 from pre-built Debian package
# Release page: https://github.com/timescale/pg_textsearch/releases/tag/v0.5.0
# Auto-detect architecture and download correct .deb
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    ARCH=$(dpkg --print-architecture) && \
    curl -L -o /tmp/pg_textsearch.deb \
      "https://github.com/timescale/pg_textsearch/releases/download/v0.5.0/pg-textsearch-postgresql-17_0.5.0-1_${ARCH}.deb" && \
    dpkg -i /tmp/pg_textsearch.deb && \
    rm /tmp/pg_textsearch.deb && \
    apt-get purge -y --auto-remove curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pgmq via PGXN (PostgreSQL Extension Network)
# pgmq is pure PL/pgSQL, no compilation needed but PGXN client requires build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pgxnclient \
        build-essential \
        postgresql-server-dev-17 && \
    pgxn install pgmq && \
    apt-get purge -y --auto-remove pgxnclient build-essential postgresql-server-dev-17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
