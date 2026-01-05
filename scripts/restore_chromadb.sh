#!/bin/bash
#
# ChromaDB Restore Script
#
# This script restores ChromaDB data from a backup file.
#
# Usage:
#   ./scripts/restore_chromadb.sh <backup_file>
#
# Example:
#   ./scripts/restore_chromadb.sh ./backups/chromadb/chromadb_backup_20251013_020000.tar.gz
#
# WARNING: This will replace all existing ChromaDB data!
#

set -e

BACKUP_FILE="$1"
CONTAINER_NAME="ragbench-chromadb-1"

if [ -z "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file path is required"
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Available backups:"
    ls -lh ./backups/chromadb/chromadb_backup_*.tar.gz 2>/dev/null || echo "  No backups found in ./backups/chromadb/"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "[$(date)] Starting ChromaDB restore from: $BACKUP_FILE"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[$(date)] ERROR: ChromaDB container '${CONTAINER_NAME}' is not running"
    echo "[$(date)] Please start the services first: docker compose up -d"
    exit 1
fi

# Confirm restore operation
read -p "WARNING: This will replace all existing ChromaDB data. Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "[$(date)] Restore cancelled"
    exit 0
fi

# Stop rag-server and celery-worker to prevent writes during restore
echo "[$(date)] Stopping rag-server and celery-worker..."
docker compose stop rag-server celery-worker

# Copy backup to container
BACKUP_FILENAME=$(basename "$BACKUP_FILE")
echo "[$(date)] Copying backup to container..."
docker cp "$BACKUP_FILE" "${CONTAINER_NAME}:/tmp/${BACKUP_FILENAME}"

# Remove existing data
echo "[$(date)] Removing existing ChromaDB data..."
docker exec "$CONTAINER_NAME" sh -c "rm -rf /chroma/chroma/*"

# Extract backup
echo "[$(date)] Extracting backup..."
docker exec "$CONTAINER_NAME" tar -xzf "/tmp/${BACKUP_FILENAME}" -C /chroma

# Remove temporary backup file from container
docker exec "$CONTAINER_NAME" rm "/tmp/${BACKUP_FILENAME}"

# Restart services
echo "[$(date)] Restarting services..."
docker compose start rag-server celery-worker

# Wait for services to be ready
echo "[$(date)] Waiting for services to initialize..."
sleep 5

# Verify restoration
echo "[$(date)] Verifying restoration..."
RESPONSE=$(curl -s http://localhost:8001/health || echo "failed")
if echo "$RESPONSE" | grep -q "healthy"; then
    echo "[$(date)] Restore completed successfully"
    echo "[$(date)] RAG server is healthy"
else
    echo "[$(date)] WARNING: RAG server health check failed"
    echo "[$(date)] Check logs with: docker compose logs rag-server"
fi
