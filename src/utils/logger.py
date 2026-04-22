"""
Logger utility for the bulk modulus MLOps project.

WHY THIS FILE EXISTS:
Every module in our project needs to log messages.
Instead of setting up logging separately in each file,
we create ONE logger here and import it everywhere.

USAGE IN OTHER FILES:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loading data...")
    logger.error("Something failed!")
"""

import logging
from pathlib import Path


def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name      : Always pass __name__
                    Tells us WHICH file the log came from
                    Example: src.data.data_loader

        log_level : How much detail to show
                    DEBUG   = Everything
                    INFO    = Normal operations
                    WARNING = Something unusual
                    ERROR   = Something broke

    Returns:
        Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "project.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logger is working!")
    logger.warning("This is a WARNING")
    logger.error("This is an ERROR")
    print("Check logs/project.log for saved logs!")
