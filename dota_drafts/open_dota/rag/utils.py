from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_hero_images_info(hero_images: Dict[str, Path]) -> str:
    if not hero_images:
        return ""
    hero_names_list = list(hero_images.keys())[:10]
    if not hero_names_list:
        return ""
    return (
        f"\n\nIMPORTANT: Below are reference hero images from the database "
        f"for visual comparison. Compare hero icons on the draft screenshot with these "
        f"reference images to accurately determine which heroes were picked by each team. "
        f"Heroes from database: {', '.join(hero_names_list)}"
    )


def try_models(
    model_names: List[str],
    create_func,
    default_error: str = "No working model found",
) -> Optional[Any]:
    for model_name in model_names:
        try:
            return create_func(model_name)
        except Exception:
            continue
    raise ValueError(default_error)


def parse_json_from_response(text: str) -> Dict[str, Any]:
    import json

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def get_empty_keywords() -> Dict[str, Any]:
    return {
        "teams": [],
        "left_team_players": [],
        "right_team_players": [],
        "left_team_heroes": [],
        "right_team_heroes": [],
    }
