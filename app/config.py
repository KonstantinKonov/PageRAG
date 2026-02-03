from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # App
    APP_NAME: str = "pagerag-api"
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = "debug_logs/pipeline.log"
    LOG_FILE_OVERWRITE: bool = True
    LOG_FILE_LEVEL: str = "DEBUG"

    # Storage
    UPLOAD_DIR: str = "data/uploads"
    DEBUG_LOG_DIR: str = "debug_logs"

    # Ollama
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_LLM_MODEL: str = "qwen3-8b-abliterated:q4km"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_CONTEXT_LENGTH: int = 8192
    OLLAMA_REASONING: bool = False
    OLLAMA_NUM_PREDICT: int = 1024

    # DB
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/pagerag"

    # Embeddings
    EMBEDDING_DIM: int = 768

    # Retrieval
    DEFAULT_TOP_K: int = 5
    DEFAULT_FETCH_K: int = 100

    # Agentic
    MAX_SUB_QUERIES: int = 3
    DEFAULT_LANGUAGE: str = "ru"

    # Web search fallback
    WEB_SEARCH_PROVIDER: str = "generic"
    WEB_SEARCH_ENDPOINT: str = ""
    WEB_SEARCH_API_KEY: str = ""
    WEB_SEARCH_TIMEOUT: float = 10.0
    WEB_SEARCH_MAX_RESULTS: int = 3


settings = Settings()
