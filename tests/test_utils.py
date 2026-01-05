# tests/test_util.py
from src.logger import get_logger


def test_logger():
    logger = get_logger("test")
    assert logger.name == "test"
    logger.info("Logger is working.")