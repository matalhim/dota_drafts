from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dota_drafts.utils.settings_utils import (
    load_config_to_env,
    lookup_env,
    parse_bool,
    parse_float,
    parse_int,
    set_default_env,
)

rag_config = "rag_config.yaml"
teams_config = "teams_config.yaml"
pro_matches_config = "pro_matches_config.yaml"
langsmith_config = "langsmith_config.yaml"


class RAGSettings(BaseModel):
    """RAG configuration settings."""

    embedding_provider: str = Field(default="huggingface")
    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    download_images: bool = Field(default=True)
    max_teams: int = Field(default=100)
    http_timeout_s: float = Field(default=20.0)
    max_pro_matches: int = Field(default=10000)
    request_delay_s: float = Field(default=0.2)
    max_retries: int = Field(default=5)


def initialise_environment() -> None:
    load_config_to_env(rag_config, prefix="RAG")
    load_config_to_env(teams_config, prefix="TEAMS")
    load_config_to_env(pro_matches_config, prefix="PRO_MATCHES")
    load_config_to_env(langsmith_config, prefix="LANGSMITH")


def build_settings() -> RAGSettings:
    initialise_environment()
    defaults = RAGSettings()

    data: dict[str, Any] = {}

    # Embedding provider - from RAG config
    provider_raw = lookup_env("RAG_EMBEDDING_PROVIDER")
    if provider_raw:
        data["embedding_provider"] = provider_raw.lower()
    else:
        data["embedding_provider"] = defaults.embedding_provider

    # Embedding model - from RAG config
    model_raw = lookup_env("RAG_EMBEDDING_MODEL_NAME") or lookup_env(
        "DOTA_DRAFTS_EMBED_MODEL"
    )
    if model_raw:
        data["embedding_model_name"] = model_raw
    else:
        set_default_env("DOTA_DRAFTS_EMBED_MODEL", defaults.embedding_model_name)

    # Download images - from RAG config
    download_raw = lookup_env("RAG_DOWNLOAD_IMAGES") or lookup_env(
        "DOTA_DRAFTS_DOWNLOAD_IMAGES"
    )
    data["download_images"] = parse_bool(download_raw, defaults.download_images)

    # Max teams - from RAG config or teams config
    max_teams_raw = (
        lookup_env("RAG_MAX_TEAMS")
        or lookup_env("DOTA_DRAFTS_MAX_TEAMS")
        or lookup_env("TEAMS_MAX_TEAMS")
    )
    if max_teams_raw is not None:
        data["max_teams"] = parse_int(max_teams_raw, defaults.max_teams)
    else:
        set_default_env("DOTA_DRAFTS_MAX_TEAMS", defaults.max_teams)

    # HTTP timeout - from RAG config
    timeout_raw = lookup_env("RAG_HTTP_TIMEOUT_S") or lookup_env(
        "DOTA_DRAFTS_HTTP_TIMEOUT"
    )
    if timeout_raw is not None:
        data["http_timeout_s"] = parse_float(timeout_raw, defaults.http_timeout_s)
    else:
        set_default_env("DOTA_DRAFTS_HTTP_TIMEOUT", defaults.http_timeout_s)

    # Max pro matches - from RAG config or pro_matches config
    max_matches_raw = (
        lookup_env("RAG_MAX_PRO_MATCHES")
        or lookup_env("DOTA_DRAFTS_MAX_PRO_MATCHES")
        or lookup_env("PRO_MATCHES_MAX_MATCHES")
    )
    if max_matches_raw is not None:
        data["max_pro_matches"] = parse_int(max_matches_raw, defaults.max_pro_matches)
    else:
        set_default_env("DOTA_DRAFTS_MAX_PRO_MATCHES", defaults.max_pro_matches)

    # Request delay - from RAG config or pro_matches config
    delay_raw = (
        lookup_env("RAG_REQUEST_DELAY_S")
        or lookup_env("DOTA_DRAFTS_REQUEST_DELAY_S")
        or lookup_env("PRO_MATCHES_REQUEST_DELAY_S")
    )
    if delay_raw is not None:
        data["request_delay_s"] = parse_float(delay_raw, defaults.request_delay_s)
    else:
        set_default_env("DOTA_DRAFTS_REQUEST_DELAY_S", defaults.request_delay_s)

    # Max retries - from RAG config
    retries_raw = lookup_env("RAG_MAX_RETRIES") or lookup_env("DOTA_DRAFTS_MAX_RETRIES")
    if retries_raw is not None:
        data["max_retries"] = parse_int(retries_raw, defaults.max_retries)
    else:
        set_default_env("DOTA_DRAFTS_MAX_RETRIES", defaults.max_retries)

    return RAGSettings(**data)


@lru_cache
def get_rag_settings() -> RAGSettings:
    return build_settings()


def reload_rag_settings() -> RAGSettings:
    get_rag_settings.cache_clear()
    return get_rag_settings()


def get_chroma_dir() -> Path:
    """Get Chroma directory from DDConfig."""
    from dota_drafts.config import DDConfig

    chroma_dir = DDConfig.CHROMA_DIR
    # Make absolute if relative
    if not chroma_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        chroma_dir = project_root / chroma_dir
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def get_data_dir() -> Path:
    """Get data directory from DDConfig."""
    from dota_drafts.config import DDConfig

    data_dir = DDConfig.DATA_DIR
    # Make absolute if relative
    if not data_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / data_dir
    return data_dir


def get_open_dota_data_dir() -> Path:
    """Get OpenDota data directory from DDConfig."""
    from dota_drafts.config import DDConfig

    data_dir = DDConfig.OPEN_DOTA_DATA_DIR
    # Make absolute if relative
    if not data_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / data_dir
    return data_dir


def get_images_dir() -> Path:
    """Get images directory from DDConfig."""
    from dota_drafts.config import DDConfig

    images_dir = DDConfig.IMAGES_DIR
    # Make absolute if relative
    if not images_dir.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        images_dir = project_root / images_dir
    return images_dir
