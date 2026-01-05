#!/bin/bash
#
# ChromaDB Backup Script
#
# This script creates a compressed backup of the ChromaDB data directory.
# It can be run manually or scheduled via cron for daily backups.
#
# Usage:
#   ./scripts/backup_chromadb.sh [backup_dir]
#
# If backup_dir is not provided, backups are stored in ./backups/chromadb/
#
# Cron example (daily at 2 AM):
#   0 2 * * * cd /path/to/ragbench && ./scripts/backup_chromadb.sh >> /var/log/chromadb_backup.log 2>&1
#

set -e

# Configuration
BACKUP_DIR="${1:-./backups/chromadb}"
BACKUP_RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="chromadb_backup_${TIMESTAMP}.tar.gz"
CONTAINER_NAME="ragbench-chromadb-1"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting ChromaDB backup..."

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[$(date)] ERROR: ChromaDB container '${CONTAINER_NAME}' is not running"
    echo "[$(date)] Available containers:"
    docker ps --format '{{.Names}}'
    exit 1
fi

# Get document count before backup
DOC_COUNT=$(docker exec "$CONTAINER_NAME" sh -c "find /chroma/chroma -type f -name '*.sqlite3' | wc -l" 2>/dev/null || echo "unknown")
echo "[$(date)] ChromaDB data files found: $DOC_COUNT"

# Create backup
echo "[$(date)] Creating backup: ${BACKUP_DIR}/${BACKUP_FILE}"
docker exec "$CONTAINER_NAME" tar -czf "/tmp/${BACKUP_FILE}" -C /chroma chroma

# Copy backup from container to host
docker cp "${CONTAINER_NAME}:/tmp/${BACKUP_FILE}" "${BACKUP_DIR}/${BACKUP_FILE}"

# Remove temporary backup from container
docker exec "$CONTAINER_NAME" rm "/tmp/${BACKUP_FILE}"

# Verify backup was created
if [ -f "${BACKUP_DIR}/${BACKUP_FILE}" ]; then
    BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_FILE}" | cut -f1)
    echo "[$(date)] Backup completed successfully: ${BACKUP_FILE} (${BACKUP_SIZE})"
else
    echo "[$(date)] ERROR: Backup file not found after creation"
    exit 1
fi

# Clean up old backups (keep last N days)
echo "[$(date)] Cleaning up backups older than ${BACKUP_RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "chromadb_backup_*.tar.gz" -mtime +${BACKUP_RETENTION_DAYS} -delete

# List remaining backups
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "chromadb_backup_*.tar.gz" | wc -l)
echo "[$(date)] Total backups retained: $BACKUP_COUNT"

echo "[$(date)] Backup process completed"
