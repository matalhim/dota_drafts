from __future__ import annotations

from typing import Any

from terminal_app.env import source


def load_dev_config(config_name: str) -> dict[str, Any]:
    """
    Load config from configs/dev/{config_name}.yaml using terminal_app.
    Returns dict (empty if not found).
    """
    try:
        config = source(config_name)
        return dict(config) if config else {}
    except Exception:
        return {}
