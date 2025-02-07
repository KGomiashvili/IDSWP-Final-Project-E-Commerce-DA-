import logging
from config import LOG_FILE, LOG_LEVEL

def get_logger(name):
    """Returns a logger that logs to both file and console."""
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setLevel(LOG_LEVEL)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    # Define log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
