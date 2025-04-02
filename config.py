from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Telegram settings
    telegram_bot_token: str = Field(env='TELEGRAM_BOT_TOKEN')
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", env='OLLAMA_BASE_URL')
    ollama_model: str = Field(default="llama3.1", env='OLLAMA_MODEL')
    ollama_temperature: float = Field(default=0.7, env='OLLAMA_TEMPERATURE')
    ollama_timeout: int = Field(default=30, env='OLLAMA_TIMEOUT')
    ollama_max_tokens: int = Field(default=32768, env='OLLAMA_MAX_TOKENS')
    
    # Retriever Ollama settings
    retriever_ollama_base_url: str = Field(default="http://127.0.0.1:11434", env='RETRIEVER_OLLAMA_BASE_URL')
    retriever_ollama_model: str = Field(default="qwen2.5-coder", env='RETRIEVER_OLLAMA_MODEL')
    retriever_ollama_temperature: float = Field(default=0, env='RETRIEVER_OLLAMA_TEMPERATURE')
    retriever_ollama_timeout: int = Field(default=30, env='RETRIEVER_OLLAMA_TIMEOUT')
    retriever_ollama_max_tokens: int = Field(default=8192, env='RETRIEVER_OLLAMA_MAX_TOKENS')
    
    # Retrieval Grader Ollama settings
    retrieval_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", env='RETRIEVER_OLLAMA_BASE_URL')
    retrieval_grader_ollama_model: str = Field(default="qwen2.5-coder", env='RETRIEVER_OLLAMA_MODEL')
    retrieval_grader_ollama_temperature: float = Field(default=0, env='RETRIEVAL_GRADER_OLLAMA_TEMPERATURE')
    retrieval_grader_ollama_timeout: int = Field(default=30, env='RETRIEVAL_GRADER_OLLAMA_TIMEOUT')
    retrieval_grader_ollama_max_tokens: int = Field(default=8192, env='RETRIEVAL_GRADER_OLLAMA_MAX_TOKENS')
    
    # OpenSearch settings
    opensearch_host: str = Field(env='OPENSEARCH_HOST')
    opensearch_port: int = Field(env='OPENSEARCH_PORT')
    opensearch_username: str = Field(env='OPENSEARCH_USERNAME')
    opensearch_password: str = Field(env='OPENSEARCH_PASSWORD')
    opensearch_index: str = Field(env='OPENSEARCH_INDEX')
    opensearch_use_ssl: bool = Field(env='OPENSEARCH_USE_SSL')
    opensearch_verify_certs: bool = Field(env='OPENSEARCH_VERIFY_CERTS')
    opensearch_query_size: int = Field(env='OPENSEARCH_QUERY_SIZE')
    
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