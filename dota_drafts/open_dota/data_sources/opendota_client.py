from __future__ import annotations

import time
from typing import Any, List, Optional

import httpx
from pydantic import BaseModel

from dota_drafts.settings import get_rag_settings

OPEN_DOTA_BASE_URL = "https://api.opendota.com/api"


class Hero(BaseModel):
    id: int
    name: str  # internal like "npc_dota_hero_axe"
    localized_name: str
    primary_attr: Optional[str] = None
    attack_type: Optional[str] = None
    roles: Optional[List[str]] = None
    img: Optional[str] = None  # some endpoints provide relative paths
    icon: Optional[str] = None

    @property
    def short_name(self) -> str:
        # convert "npc_dota_hero_axe" -> "axe"
        return self.name.replace("npc_dota_hero_", "")

    @property
    def image_url(self) -> str:
        # Prefer dota2 cdn modern react images (stable)
        return f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{self.short_name}.png"

    def get_draft_image_urls(self) -> List[tuple[str, str]]:
        """
        Get multiple image URLs for the hero that are suitable for draft matching.
        Returns list of (url, variant_name) tuples.
        These images are similar to what appears in draft interface.
        """
        urls = [
            # Modern React format (primary, used in current draft UI)
            (
                f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{self.short_name}.png",
                "react",
            ),
            # Full portrait (modern react)
            (
                f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{self.short_name}_full.png",
                "react_full",
            ),
            # Legacy full portrait (older format, may differ slightly)
            (
                f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{self.short_name}_full.png",
                "legacy_full",
            ),
            # Vertical format (portrait orientation)
            (
                f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{self.short_name}_vert.jpg",
                "vert",
            ),
            # Large format (higher resolution)
            (
                f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/heroes/{self.short_name}_lg.png",
                "lg",
            ),
        ]
        return urls


class Team(BaseModel):
    team_id: int
    name: Optional[str] = None
    tag: Optional[str] = None
    logo_url: Optional[str] = None
    rating: Optional[float] = None
    wins: Optional[int] = None
    losses: Optional[int] = None


class TeamPlayer(BaseModel):
    account_id: Optional[int] = None
    name: Optional[str] = None
    is_current_team_member: Optional[bool] = None
    avatarfull: Optional[str] = None
    country_code: Optional[str] = None


class ProPlayer(BaseModel):
    account_id: Optional[int] = None
    name: Optional[str] = None
    country_code: Optional[str] = None
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    team_tag: Optional[str] = None
    is_pro: Optional[bool] = None
    avatarfull: Optional[str] = None


class OpenDotaClient:
    def __init__(self, timeout_s: float | None = None) -> None:
        rag_settings = get_rag_settings()
        self.timeout_s = (
            timeout_s if timeout_s is not None else rag_settings.http_timeout_s
        )
        self._client = httpx.Client(
            timeout=self.timeout_s, headers={"User-Agent": "dota-drafts/0.0.1"}
        )

    def _get(self, path: str) -> Any:
        rag_settings = get_rag_settings()
        url = f"{OPEN_DOTA_BASE_URL}{path}"
        backoff = rag_settings.request_delay_s
        resp = None
        for _attempt in range(rag_settings.max_retries):
            resp = self._client.get(url)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                # Respect Retry-After if provided; otherwise exponential backoff
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else backoff
                time.sleep(sleep_s)
                backoff = min(backoff * 2, 10.0)
                continue
            resp.raise_for_status()
            return resp.json()
        # last try raise
        if resp is not None:
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Failed to get response after all retries")

    def fetch_pro_players(self) -> List[ProPlayer]:
        raw = self._get("/proPlayers")
        return [ProPlayer(**item) for item in raw]

    def fetch_pro_matches_page(
        self, less_than_match_id: Optional[int] = None
    ) -> List[dict]:
        path = "/proMatches"
        if less_than_match_id:
            path += f"?less_than_match_id={less_than_match_id}"
        return self._get(path)

    def fetch_match_detail(self, match_id: int) -> dict:
        return self._get(f"/matches/{match_id}")

    def fetch_heroes(self) -> List[Hero]:
        # heroStats provides many useful fields and localized_name
        raw = self._get("/heroStats")
        heroes = [Hero(**item) for item in raw]
        return heroes

    def fetch_teams(self, limit: int | None = None) -> List[Team]:
        raw = self._get("/teams")
        teams = [Team(**item) for item in raw]
        if limit and limit > 0:
            teams = teams[:limit]
        return teams

    def fetch_team_players(self, team_id: int) -> List[TeamPlayer]:
        raw = self._get(f"/teams/{team_id}/players")
        players = [TeamPlayer(**item) for item in raw]
        return players
