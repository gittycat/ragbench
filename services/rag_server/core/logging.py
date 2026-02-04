import logging
import re
import os


class TimestampRemovalFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return True


class URLShortenerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg) if hasattr(record, 'msg') else record.getMessage()

        # Shorten long URLs (keep protocol, host, and last path segment)
        msg = re.sub(
            r'(https?://[^/]+)/([^/]+/){3,}([^/\s"\']+)',
            r'\1/.../\3',
            msg
        )

        # Update the record
        if hasattr(record, 'msg'):
            record.msg = msg
        else:
            record.args = ()

        return True


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


def configure_logging(log_level: str = None):
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    # Remove timestamp from format (no asctime)
    log_format = '%(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        force=True
    )

    # Add URL shortener to all loggers
    url_filter = URLShortenerFilter()
    for logger_name in ['httpx', 'httpcore', 'uvicorn', 'celery']:
        logger = logging.getLogger(logger_name)
        logger.addFilter(url_filter)

    # Add health check filter to uvicorn access logger
    health_filter = HealthCheckFilter()
    logging.getLogger("uvicorn.access").addFilter(health_filter)

    # Configure httpx/httpcore to not show full request details
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Reduce noise from HuggingFace model downloads (filelock and urllib3)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Suppress verbose bm25s library logs
    logging.getLogger("bm25s").setLevel(logging.WARNING)
