from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Telegram settings
    telegram_bot_token: str = Field(..., env='TELEGRAM_BOT_TOKEN')
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", env='OLLAMA_BASE_URL')
    ollama_model: str = Field(default="mistral", env='OLLAMA_MODEL')
    ollama_temperature: float = Field(default=0.7, env='OLLAMA_TEMPERATURE')
    ollama_timeout: int = Field(default=30, env='OLLAMA_TIMEOUT')
    
    # Application settings
    debug: bool = Field(default=False, env='DEBUG')
    log_level: str = Field(default="INFO", env='LOG_LEVEL')
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )


# Create global settings instance
settings = Settings() 