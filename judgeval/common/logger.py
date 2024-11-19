import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Global variables
logger = None
LOGGING_ENABLED = False

# Add these as module-level variables
current_example_id = None
current_timestamp = None

def create_example_handler(timestamp: str, example_idx: int) -> RotatingFileHandler:
    """Creates a file handler for a specific example"""
    log_dir = Path('./logs/examples')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [Example_%(example_id)s][%(timestamp)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a unique file for each example
    file_handler = RotatingFileHandler(
        log_dir / f"{timestamp}_example_{example_idx}.log",
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
        mode='a'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    return file_handler

@contextmanager
def example_logging_context(timestamp: str, example_idx: int):
    """Context manager for example-specific logging"""
    global current_example_id, current_timestamp
    
    # Set the current example info
    current_example_id = example_idx
    current_timestamp = timestamp
    
    handler = create_example_handler(timestamp, example_idx)
    logger.addHandler(handler)
    try:
        yield
    finally:
        # Clear the example info
        current_example_id = None
        current_timestamp = None
        logger.removeHandler(handler)
        handler.close()

@contextmanager
def enable_logging():
    """
    Context manager to temporarily enable logging for a specific block of code.
    """
    global LOGGING_ENABLED
    LOGGING_ENABLED = True
    try:
        logger.info("Logging enabled")
        yield
    finally:
        logger.info("Logging disabled")
        LOGGING_ENABLED = False

def _initialize_logger(
    name: str = "judgeval",
    max_bytes: int = 1024 * 1024,  # 1MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Initialize the global logger instance if it doesn't exist.
    Returns the global logger instance.
    """
    global logger
    if logger is not None:
        return logger
    
    # Create logs directory if it doesn't exist
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a custom formatter that includes example info when available
    class ExampleFormatter(logging.Formatter):
        def format(self, record):
            if current_example_id is not None and current_timestamp is not None:
                record.example_id = current_example_id
                record.timestamp = current_timestamp
                return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [Example_%(example_id)s][%(timestamp)s] %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S').format(record)
            return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S').format(record)
    
    # Use the custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ExampleFormatter())
    console_handler.setLevel(logging.DEBUG)
    
    log_filename = f"{name}.log"
    file_handler = RotatingFileHandler(
        log_dir / log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        mode='a'
    )
    file_handler.setFormatter(ExampleFormatter())
    file_handler.setLevel(logging.DEBUG)

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# Initialize the global logger when module is imported
logger = _initialize_logger()

def log_if_enabled(func):
    """Decorator to check if logging is enabled before executing logging statements"""
    def wrapper(*args, **kwargs):
        if LOGGING_ENABLED:
            return func(*args, **kwargs)
    return wrapper

@log_if_enabled
def debug(msg: str, example_idx: int = None):
    """Log debug message if logging is enabled"""
    logger.debug(msg)

@log_if_enabled
def info(msg: str, example_idx: int = None):
    """Log info message if logging is enabled"""
    logger.info(msg)

@log_if_enabled
def warning(msg: str, example_idx: int = None):
    """Log warning message if logging is enabled"""
    logger.warning(msg)

@log_if_enabled
def error(msg: str, example_idx: int = None):
    """Log error message if logging is enabled"""
    logger.error(msg)