"""
Structured Logging Configuration
Following specifications from 2.CRITICAL_INITIAL_SETUP.md Section 3
"""

import logging
import sys


def setup_logging(log_level: str = "INFO"):
    """
    Configure structured logging
    Simple format for development, JSON for production
    """
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("motor").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"✅ Logging configured at level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
