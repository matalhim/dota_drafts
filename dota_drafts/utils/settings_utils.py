import os
from typing import Any, cast

from terminal_app.env import source


def set_default_env(name: str, value: Any) -> None:
    if value is None:
        return
    if name not in os.environ:
        os.environ[name] = str(value)


def lookup_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is not None:
        stripped = value.strip()
        if stripped:
            return stripped
        if value == "0":
            return value
    return None


def parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def load_config_to_env(config_name: str, prefix: str | None = None) -> dict[str, Any]:
    try:
        raw_config = source(config_name)
        config = cast(dict[str, Any], raw_config) if raw_config else None
        if config:
            for key, value in config.items():
                env_name = key.upper()
                if prefix:
                    env_name = f"{prefix}_{env_name}"
                set_default_env(env_name, value)
            return dict(config)
        return {}
    except Exception:
        return {}
