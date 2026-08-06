from src.logger import get_logger

logger = get_logger(__name__)

logger.info("Project logger is working correctly.")
logger.warning("This is a warning message.")
logger.error("This is a sample error message.")