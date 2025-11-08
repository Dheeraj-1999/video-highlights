import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # TEMP_DIR = "data/raw"
    # PROCESSED_DIR = "data/processed"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    MAX_VIDEO_LENGTH = 10 * 60

    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    @staticmethod
    def ensure_dirs():
        """Make sure all critical directories exist."""
        os.makedirs(Config.RAW_DIR, exist_ok=True)
        os.makedirs(Config.PROCESSED_DIR, exist_ok=True)