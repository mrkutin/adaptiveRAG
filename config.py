from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Telegram settings
    telegram_bot_token: str = Field(json_schema_extra={"env": "TELEGRAM_BOT_TOKEN"})
    
    # Answerer Ollama settings
    answerer_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "ANSWERER_OLLAMA_BASE_URL"})
    answerer_ollama_model: str = Field(default="llama3.1", json_schema_extra={"env": "ANSWERER_OLLAMA_MODEL"})
    answerer_ollama_temperature: float = Field(default=0.7, json_schema_extra={"env": "ANSWERER_OLLAMA_TEMPERATURE"})
    answerer_ollama_timeout: int = Field(default=30, json_schema_extra={"env": "ANSWERER_OLLAMA_TIMEOUT"})
    answerer_ollama_max_tokens: int = Field(default=32768, json_schema_extra={"env": "ANSWERER_OLLAMA_MAX_TOKENS"})

    # Retriever Ollama settings
    retriever_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "RETRIEVER_OLLAMA_BASE_URL"})
    retriever_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "RETRIEVER_OLLAMA_MODEL"})
    retriever_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "RETRIEVER_OLLAMA_TEMPERATURE"})
    retriever_ollama_timeout: int = Field(default=30, json_schema_extra={"env": "RETRIEVER_OLLAMA_TIMEOUT"})
    retriever_ollama_max_tokens: int = Field(default=8192, json_schema_extra={"env": "RETRIEVER_OLLAMA_MAX_TOKENS"})
    
    # Retrieval Grader Ollama settings
    retrieval_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "RETRIEVER_OLLAMA_BASE_URL"})
    retrieval_grader_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "RETRIEVER_OLLAMA_MODEL"})
    retrieval_grader_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "RETRIEVAL_GRADER_OLLAMA_TEMPERATURE"})
    retrieval_grader_ollama_timeout: int = Field(default=30, json_schema_extra={"env": "RETRIEVAL_GRADER_OLLAMA_TIMEOUT"})
    retrieval_grader_ollama_max_tokens: int = Field(default=8192, json_schema_extra={"env": "RETRIEVAL_GRADER_OLLAMA_MAX_TOKENS"})
    
    # Question Rewriter Ollama settings
    question_rewriter_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_BASE_URL"})
    question_rewriter_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_MODEL"})
    question_rewriter_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_TEMPERATURE"})
    question_rewriter_ollama_timeout: int = Field(default=30, json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_TIMEOUT"})
    question_rewriter_ollama_max_tokens: int = Field(default=8192, json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_MAX_TOKENS"})
    
    # Hallucination Grader Ollama settings
    hallucination_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_BASE_URL"})
    hallucination_grader_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_MODEL"})
    hallucination_grader_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_TEMPERATURE"})
    hallucination_grader_ollama_timeout: int = Field(default=30, json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_TIMEOUT"})
    hallucination_grader_ollama_max_tokens: int = Field(default=8192, json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_MAX_TOKENS"})
    
    # Answer Grader Ollama settings
    answer_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_BASE_URL"})
    answer_grader_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_MODEL"})
    answer_grader_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_TEMPERATURE"})
    answer_grader_ollama_timeout: int = Field(default=30, json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_TIMEOUT"})
    answer_grader_ollama_max_tokens: int = Field(default=8192, json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_MAX_TOKENS"})

    # OpenSearch settings
    opensearch_host: str = Field(json_schema_extra={"env": "OPENSEARCH_HOST"})
    opensearch_port: int = Field(json_schema_extra={"env": "OPENSEARCH_PORT"})
    opensearch_username: str = Field(json_schema_extra={"env": "OPENSEARCH_USERNAME"})
    opensearch_password: str = Field(json_schema_extra={"env": "OPENSEARCH_PASSWORD"})
    opensearch_index: str = Field(json_schema_extra={"env": "OPENSEARCH_INDEX"})
    opensearch_use_ssl: bool = Field(json_schema_extra={"env": "OPENSEARCH_USE_SSL"})
    opensearch_verify_certs: bool = Field(json_schema_extra={"env": "OPENSEARCH_VERIFY_CERTS"})
    opensearch_query_size: int = Field(default=20, json_schema_extra={"env": "OPENSEARCH_QUERY_SIZE", "default": 20})
    
    # Application settings
    debug: bool = Field(default=False, json_schema_extra={"env": "DEBUG"})
    log_level: str = Field(default="INFO", json_schema_extra={"env": "LOG_LEVEL"})
    
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )


# Create global settings instance
settings = Settings() 