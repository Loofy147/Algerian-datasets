import logging
import sys
from .config import settings

def setup_logging():
    """Sets up the logging configuration for the application."""
    log_level = logging.DEBUG if settings.ENV == "development" else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set levels for specific loggers if needed
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

logger = logging.getLogger("algeria_data_platform")
