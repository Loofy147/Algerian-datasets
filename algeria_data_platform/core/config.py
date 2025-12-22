from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Manages application-wide settings and configurations.
    - Loads environment variables from a .env file (if present).
    - Provides default values for critical settings.
    - Centralizes configuration to simplify management.
    """
    APP_NAME: str = "Algeria Data Platform API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "API for accessing high-quality Algerian market data and insights."
    ENV: str = "development"  # Default to development
    DATABASE_URL: str = "sqlite:///./algeria_data_platform.db"
    TEST_DATABASE_URL: str = "sqlite:///:memory:"

    class Config:
        # This tells Pydantic to look for a .env file
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single, importable instance of the settings
settings = Settings()
