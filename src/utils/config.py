import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TEMP_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    MAX_VIDEO_LENGTH = 10 * 60

    @staticmethod
    def validate():
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")