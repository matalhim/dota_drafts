from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from terminal_app.env import source

from dota_drafts.settings import (
    get_chroma_dir,
    get_data_dir,
    get_images_dir,
    get_open_dota_data_dir,
    get_rag_settings,
    langsmith_config,
)
from dota_drafts.utils.settings_utils import (
    lookup_env,
    parse_bool,
)


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    cache_dir: Path
    images_dir: Path
    chroma_dir: Path


def get_paths() -> Paths:
    root = Path(__file__).resolve().parents[3]
    data_dir = get_data_dir()
    cache_dir = get_open_dota_data_dir()
    images_dir = get_images_dir()
    chroma_dir = get_chroma_dir()
    for p in (data_dir, cache_dir, images_dir, chroma_dir):
        p.mkdir(parents=True, exist_ok=True)
    return Paths(
        project_root=root,
        data_dir=data_dir,
        cache_dir=cache_dir,
        images_dir=images_dir,
        chroma_dir=chroma_dir,
    )


def getSettings():
    return get_rag_settings()


@dataclass(frozen=True)
class Settings:
    embedding_provider: str
    embedding_model_name: str
    download_images: bool
    max_teams: int
    http_timeout_s: float
    max_pro_matches: int
    request_delay_s: float
    max_retries: int

    @classmethod
    def from_rag_settings(cls, rag_settings) -> "Settings":
        return cls(
            embedding_provider=rag_settings.embedding_provider,
            embedding_model_name=rag_settings.embedding_model_name,
            download_images=rag_settings.download_images,
            max_teams=rag_settings.max_teams,
            http_timeout_s=rag_settings.http_timeout_s,
            max_pro_matches=rag_settings.max_pro_matches,
            request_delay_s=rag_settings.request_delay_s,
            max_retries=rag_settings.max_retries,
        )


def load_api_keys() -> None:
    configs_dir = get_paths().project_root / "configs"

    google_key_file = configs_dir / ".google_api_key.env"
    if google_key_file.exists() and not os.getenv("GOOGLE_API_KEY"):
        try:
            with google_key_file.open("r") as f:
                line = f.read().strip()
                if line.startswith("GOOGLE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key:
                        os.environ["GOOGLE_API_KEY"] = key
        except Exception:
            pass

    langchain_key_file = configs_dir / ".langchain_api_key.env"
    if langchain_key_file.exists():
        try:
            with langchain_key_file.open("r") as f:
                line = f.read().strip()
                if line.startswith("LANGCHAIN_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key:
                        os.environ["LANGCHAIN_API_KEY"] = key
                        if not os.getenv("LANGSMITH_API_KEY"):
                            os.environ["LANGSMITH_API_KEY"] = key
                elif line.startswith("LANGSMITH_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key:
                        os.environ["LANGSMITH_API_KEY"] = key
                        if not os.getenv("LANGCHAIN_API_KEY"):
                            os.environ["LANGCHAIN_API_KEY"] = key
        except Exception:
            pass

    tracing_enabled = (
        lookup_env("LANGSMITH_TRACING")
        or lookup_env("LANGSMITH_TRACING_ENABLED")
        or lookup_env("LANGCHAIN_TRACING_V2")
    )
    if tracing_enabled is None:
        has_api_key = bool(
            os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        )
        tracing_enabled = "true" if has_api_key else "false"

    if parse_bool(tracing_enabled, False):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"

        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_api_key and not os.getenv("LANGCHAIN_API_KEY"):
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

        langsmith_endpoint = lookup_env("LANGSMITH_ENDPOINT")
        if langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint

        project_name = (
            lookup_env("LANGSMITH_PROJECT")
            or lookup_env("LANGSMITH_PROJECT_NAME")
            or os.getenv("LANGCHAIN_PROJECT")
        )
        if project_name:
            os.environ["LANGCHAIN_PROJECT"] = project_name
            os.environ["LANGSMITH_PROJECT"] = project_name
        elif not os.getenv("LANGCHAIN_PROJECT"):
            default_project = "dota-drafts"
            os.environ["LANGCHAIN_PROJECT"] = default_project
            os.environ["LANGSMITH_PROJECT"] = default_project

        try:
            langsmith_cfg = source(langsmith_config)
            if langsmith_cfg and "tags" in langsmith_cfg:
                tags = langsmith_cfg["tags"]
                if isinstance(tags, list):
                    os.environ["LANGCHAIN_TAGS"] = ",".join(str(tag) for tag in tags)
                elif isinstance(tags, str):
                    os.environ["LANGCHAIN_TAGS"] = tags
        except Exception:
            tags_env = lookup_env("LANGSMITH_TAGS")
            if tags_env:
                os.environ["LANGCHAIN_TAGS"] = tags_env


paths = get_paths()
settings = Settings.from_rag_settings(getSettings())
load_api_keys()
