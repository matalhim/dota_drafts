from pathlib import Path

from terminal_app.env import ProjectConfig


class DotaDraftsConfig(ProjectConfig):
    LOGGING_DIR: Path = Path("logging")

    DATA_DIR: Path = Path("data")
    GLOB_CONFIG_DIR: Path = Path("configs")

    OPEN_DOTA_DATA_DIR: Path = Path("data/opendota")
    IMAGES_DIR: Path = Path("data/images")
    INPUT_DIR: Path = Path("data/input")
    CHROMA_DIR: Path = Path("data/chroma")


DDConfig = DotaDraftsConfig()
