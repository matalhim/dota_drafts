__version__ = "0.0.1"
from .config import DDConfig
from .open_dota.rag import (  # noqa: F401
    build_index,
    fetch_and_cache,  # noqa: F401
    load_index,
    query,
)

__all__ = [
    "DDConfig",
]
