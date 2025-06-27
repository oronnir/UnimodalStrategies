# =============================================================================
# ECML-PKDD 2025 ‒ “Unimodal Strategies in Density-Based Clustering”
# Authors : Oron Nir, Jay Tenenbaum, Ariel Shamir
# Paper   : https://arxiv.org/abs/######   (pre-print link)
# Code    : https://github.com/oronnir/UnimodalStrategies
# License : MIT (see LICENSE file for full text)
# =============================================================================
import logging
import sys
from typing import Optional

def setup_logger(name: str = 'unimodal_strategies', level: Optional[str] = None) -> logging.Logger:
    """
    Setup and return a logger with appropriate configuration.
    
    Args:
        name: Logger name
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
               If None, will use INFO for normal operation, DEBUG if explicitly requested
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Set default level to INFO, but allow override via environment or parameter
    if level is None:
        import os
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

# Global logger instance
logger = setup_logger()
