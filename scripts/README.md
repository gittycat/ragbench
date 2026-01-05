# ChromaDB Backup and Restore Scripts

## Overview

These scripts provide backup and restore functionality for the ChromaDB vector database. This is a defensive measure against persistence reliability issues reported in production environments (2025).

## Scripts

### `backup_chromadb.sh`

Creates a compressed backup of ChromaDB data.

**Usage:**
```bash
# Backup to default location (./backups/chromadb/)
./scripts/backup_chromadb.sh

# Backup to custom location
./scripts/backup_chromadb.sh /path/to/backup/directory
```

**Features:**
- Creates timestamped backups (`chromadb_backup_YYYYMMDD_HHMMSS.tar.gz`)
- Automatically retains last 30 days of backups
- Verifies container is running before backup
- Logs backup size and document count

**Scheduling with Cron:**

For daily backups at 2 AM:
```bash
# Edit crontab
crontab -e

# Add this line (adjust path as needed)
0 2 * * * cd /path/to/ragbench && ./scripts/backup_chromadb.sh >> /var/log/chromadb_backup.log 2>&1
```

### `restore_chromadb.sh`

Restores ChromaDB data from a backup file.

**Usage:**
```bash
# List available backups
ls -lh ./backups/chromadb/

# Restore from specific backup
./scripts/restore_chromadb.sh ./backups/chromadb/chromadb_backup_20251013_020000.tar.gz
```

**Warning:** This replaces ALL existing ChromaDB data. You'll be prompted to confirm.

**Process:**
1. Stops rag-server and celery-worker to prevent writes
2. Removes existing ChromaDB data
3. Extracts backup
4. Restarts services
5. Verifies health check

## Backup Strategy (Phase 1 Implementation)

### Recommended Schedule

- **Daily automated backups** at 2 AM (low-traffic period)
- **30-day retention** (automatically removes older backups)
- **Manual backups** before major changes (model updates, bulk deletions)

### Storage Requirements

Backup size depends on document count and embedding dimensions:
- Typical: 100-500 MB per 1000 documents
- Plan for: 30 days × daily backup size

### Monitoring

Check backup logs regularly:
```bash
# View last backup
ls -lth ./backups/chromadb/ | head -n 2

# Check backup log (if using cron)
tail -f /var/log/chromadb_backup.log
```

## Troubleshooting

### Container Name Not Found

If you see `ERROR: ChromaDB container 'ragbench-chromadb-1' is not running`:

1. Check actual container name:
   ```bash
   docker ps --format '{{.Names}}' | grep chromadb
   ```

2. Update `CONTAINER_NAME` variable in scripts if different

### Permission Denied

Ensure scripts are executable:
```bash
chmod +x scripts/*.sh
```

### Disk Space Issues

Monitor backup directory size:
```bash
du -sh ./backups/chromadb/
```

Reduce retention period if needed (edit `BACKUP_RETENTION_DAYS` in backup script).

## References

- **Phase 1 Implementation**: RAG_ACCURACY_IMPROVEMENT_PLAN_2025.md
- **Issue Context**: ChromaDB HttpClient persistence reliability (2025 production reports)
