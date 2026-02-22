# PostgreSQL 17 with pg_textsearch (BM25)
FROM postgres:17

# Install pg_textsearch v0.5.1 from pre-built package (zip containing .deb)
# Release page: https://github.com/timescale/pg_textsearch/releases/tag/v0.5.1
# Auto-detect architecture and download correct package
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates unzip && \
    ARCH=$(dpkg --print-architecture) && \
    curl -L -o /tmp/pg_textsearch.zip \
      "https://github.com/timescale/pg_textsearch/releases/download/v0.5.1/pg-textsearch-v0.5.1-pg17-${ARCH}.zip" && \
    unzip /tmp/pg_textsearch.zip -d /tmp && \
    dpkg -i /tmp/pg-textsearch-postgresql-17_0.5.1-1_${ARCH}.deb && \
    rm /tmp/pg_textsearch.zip /tmp/pg-textsearch-postgresql-17_0.5.1-1_${ARCH}.deb && \
    apt-get purge -y --auto-remove curl unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
