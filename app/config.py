import os


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value


class Settings:
    # App
    APP_NAME = _get_env("APP_NAME", "pagerag-api")
    LOG_LEVEL = _get_env("LOG_LEVEL", "INFO")
    LOG_FILE_PATH = _get_env("LOG_FILE_PATH", "debug_logs/pipeline.log")
    LOG_FILE_OVERWRITE = _get_env("LOG_FILE_OVERWRITE", "true").lower() == "true"
    LOG_FILE_LEVEL = _get_env("LOG_FILE_LEVEL", "DEBUG")

    # Storage
    UPLOAD_DIR = _get_env("UPLOAD_DIR", "data/uploads")
    DEBUG_LOG_DIR = _get_env("DEBUG_LOG_DIR", "debug_logs")

    # Ollama
    OLLAMA_BASE_URL = _get_env("OLLAMA_BASE_URL", "http://ollama:11434")
    OLLAMA_LLM_MODEL = _get_env("OLLAMA_LLM_MODEL", "qwen3-8b-abliterated:q4km")
    OLLAMA_EMBED_MODEL = _get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    OLLAMA_CONTEXT_LENGTH = int(_get_env("OLLAMA_CONTEXT_LENGTH", "8192"))
    OLLAMA_REASONING = _get_env("OLLAMA_REASONING", "false").lower() == "true"

    # DB
    DATABASE_URL = _get_env(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@postgres:5432/pagerag",
    )

    # Embeddings
    EMBEDDING_DIM = int(_get_env("EMBEDDING_DIM", "768"))

    # Retrieval
    DEFAULT_TOP_K = int(_get_env("DEFAULT_TOP_K", "5"))
    DEFAULT_FETCH_K = int(_get_env("DEFAULT_FETCH_K", "100"))

    # Agentic
    MAX_SUB_QUERIES = int(_get_env("MAX_SUB_QUERIES", "3"))
    DEFAULT_LANGUAGE = _get_env("DEFAULT_LANGUAGE", "ru")
    OLLAMA_NUM_PREDICT = int(_get_env("OLLAMA_NUM_PREDICT", "1024"))


settings = Settings()
