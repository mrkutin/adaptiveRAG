from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field



class Settings(BaseSettings):
    # Telegram settings
    telegram_bot_token: str = Field(json_schema_extra={"env": "TELEGRAM_BOT_TOKEN"})
    
    # Answerer Ollama settings
    answerer_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "ANSWERER_OLLAMA_BASE_URL"})
    #1. deepcoder
    #2. codellama
    answerer_ollama_model: str = Field(default="deepcoder", json_schema_extra={"env": "ANSWERER_OLLAMA_MODEL"})
    answerer_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "ANSWERER_OLLAMA_TEMPERATURE"})
    answerer_ollama_num_ctx: int = Field(default=65536, json_schema_extra={"env": "ANSWERER_OLLAMA_NUM_CTX"})

    # Log Summarizer Ollama settings
    log_summarizer_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "LOG_SUMMARIZER_OLLAMA_BASE_URL"})
    log_summarizer_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "LOG_SUMMARIZER_OLLAMA_MODEL"})
    log_summarizer_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "LOG_SUMMARIZER_OLLAMA_TEMPERATURE"})
    log_summarizer_ollama_num_ctx: int = Field(default=8192, json_schema_extra={"env": "LOG_SUMMARIZER_OLLAMA_NUM_CTX"})

    # Retriever Ollama settings
    retriever_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "RETRIEVER_OLLAMA_BASE_URL"})
    retriever_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "RETRIEVER_OLLAMA_MODEL"})
    retriever_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "RETRIEVER_OLLAMA_TEMPERATURE"})
    retriever_ollama_num_ctx: int = Field(default=8192, json_schema_extra={"env": "RETRIEVER_OLLAMA_NUM_CTX"})
    
    # Retrieval Grader Ollama settings
    retrieval_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "RETRIEVER_OLLAMA_BASE_URL"})
    retrieval_grader_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "RETRIEVER_OLLAMA_MODEL"})
    retrieval_grader_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "RETRIEVAL_GRADER_OLLAMA_TEMPERATURE"})
    retrieval_grader_ollama_num_ctx: int = Field(default=8192, json_schema_extra={"env": "RETRIEVAL_GRADER_OLLAMA_NUM_CTX"})
    
    # Question Rewriter Ollama settings
    question_rewriter_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_BASE_URL"})
    question_rewriter_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_MODEL"})
    question_rewriter_ollama_temperature: float = Field(default=1, json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_TEMPERATURE"})
    question_rewriter_ollama_num_ctx: int = Field(default=8192, json_schema_extra={"env": "QUESTION_REWRITER_OLLAMA_NUM_CTX"})
    
    # # Hallucination Grader Ollama settings
    # hallucination_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_BASE_URL"})
    # hallucination_grader_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_MODEL"})
    # hallucination_grader_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_TEMPERATURE"})
    # hallucination_grader_ollama_num_ctx: int = Field(default=8192, json_schema_extra={"env": "HALLUCINATION_GRADER_OLLAMA_NUM_CTX"})
    
    # # Answer Grader Ollama settings
    # answer_grader_ollama_base_url: str = Field(default="http://127.0.0.1:11434", json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_BASE_URL"})
    # answer_grader_ollama_model: str = Field(default="qwen2.5-coder", json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_MODEL"})
    # answer_grader_ollama_temperature: float = Field(default=0, json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_TEMPERATURE"})
    # answer_grader_ollama_num_ctx: int = Field(default=65536, json_schema_extra={"env": "ANSWER_GRADER_OLLAMA_NUM_CTX"})

    # OpenSearch settings
    opensearch_host: str = Field(json_schema_extra={"env": "OPENSEARCH_HOST"})
    opensearch_port: int = Field(json_schema_extra={"env": "OPENSEARCH_PORT"})
    opensearch_username: str = Field(json_schema_extra={"env": "OPENSEARCH_USERNAME"})
    opensearch_password: str = Field(json_schema_extra={"env": "OPENSEARCH_PASSWORD"})
    opensearch_index: str = Field(json_schema_extra={"env": "OPENSEARCH_INDEX"})
    opensearch_use_ssl: bool = Field(json_schema_extra={"env": "OPENSEARCH_USE_SSL"})
    opensearch_verify_certs: bool = Field(json_schema_extra={"env": "OPENSEARCH_VERIFY_CERTS"})
    opensearch_query_size: int = Field(default=10, json_schema_extra={"env": "OPENSEARCH_QUERY_SIZE"})
    
    # Application settings
    debug: bool = Field(default=False, json_schema_extra={"env": "DEBUG"})
    log_level: str = Field(default="INFO", json_schema_extra={"env": "LOG_LEVEL"})
    
    # CodeBaseRetriever settings
    codebase_path: str = Field(default="./code_base/enterprise-service-bus", json_schema_extra={"env": "CODEBASE_PATH"})
    codebase_file_pattern: str = Field(default="**/*", json_schema_extra={"env": "CODEBASE_FILE_PATTERN"})
    codebase_file_extensions: List[str] = Field(default=[".js"], json_schema_extra={"env": "CODEBASE_FILE_EXTENSIONS"})
    codebase_language: str = Field(default="js", json_schema_extra={"env": "CODEBASE_LANGUAGE"})
    codebase_embedding_model: str = Field(default="unclemusclez/jina-embeddings-v2-base-code", json_schema_extra={"env": "CODEBASE_EMBEDDING_MODEL"})
    codebase_k: int = Field(default=1, json_schema_extra={"env": "CODEBASE_K"})
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )


# Create global settings instance
settings = Settings() 