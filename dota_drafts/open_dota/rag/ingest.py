from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from dota_drafts.open_dota.config_loader import load_dev_config
from dota_drafts.open_dota.data_sources.opendota_client import (
    Hero,
    OpenDotaClient,
    ProPlayer,
    Team,
    TeamPlayer,
)
from dota_drafts.open_dota.rag.config import paths, settings

console = Console()


def saveJson(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def downloadImage(url: str, dest: Path) -> None:
    if not url:
        return
    try:
        resp = httpx.get(url, timeout=settings.http_timeout_s)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    except Exception:
        pass


def fetch_and_cache() -> Tuple[List[Hero], List[Team], dict[int, List[TeamPlayer]]]:
    client = OpenDotaClient()
    console.log("Fetching heroes...")
    heroes = client.fetch_heroes()
    saveJson([h.model_dump() for h in heroes], paths.cache_dir / "heroes.json")

    console.log("Fetching teams and pro players...")
    pro_players_all: List[ProPlayer] = client.fetch_pro_players()
    pro_team_ids = {p.team_id for p in pro_players_all if p.team_id}
    teams_all = client.fetch_teams(limit=None)
    teams = [t for t in teams_all if t.team_id in pro_team_ids]
    teams_cfg = load_dev_config("teams_config.yaml") or {}
    top_n_cfg = int(teams_cfg.get("top_rating_teams", 30))
    teams.sort(key=lambda t: (t.rating or 0), reverse=True)
    teams = teams[:top_n_cfg]
    if settings.max_teams and settings.max_teams > 0:
        teams = teams[: settings.max_teams]
    saveJson([t.model_dump() for t in teams], paths.cache_dir / "teams.json")

    selected_team_ids = {t.team_id for t in teams}
    pro_players = [
        p
        for p in pro_players_all
        if (p.is_pro or p.team_id) and (p.team_id in selected_team_ids)
    ]
    saveJson(
        [p.model_dump() for p in pro_players], paths.cache_dir / "pro_players.json"
    )
    team_players: dict[int, List[TeamPlayer]] = {}
    for t in track(teams, description="Fetching rosters"):
        players = client.fetch_team_players(t.team_id)
        current_players = [p for p in players if p.is_current_team_member]
        team_players[t.team_id] = current_players
    saveJson(
        {str(k): [p.model_dump() for p in v] for k, v in team_players.items()},
        paths.cache_dir / "team_players.json",
    )

    if settings.download_images:
        console.log("Downloading hero images (multiple variants for draft matching)...")
        for h in tqdm(heroes, desc="Hero images"):
            image_urls = h.get_draft_image_urls()
            for url, variant in image_urls:
                if variant == "react":
                    dest = paths.images_dir / "heroes" / f"{h.short_name}.png"
                else:
                    dest = paths.images_dir / "heroes" / f"{h.short_name}_{variant}.png"

                if not dest.exists():
                    downloadImage(url, dest)
        console.log("Downloading team logos...")
        for t in tqdm(teams, desc="Team logos"):
            if t.logo_url:
                fname = f"{t.team_id}.png"
                dest = paths.images_dir / "teams" / fname
                if not dest.exists():
                    downloadImage(t.logo_url, dest)

    return heroes, teams, team_players


def fetch_pro_matches_last_3_months(max_matches: Optional[int] = None) -> None:
    client = OpenDotaClient()
    cutoff_unix = int((dt.datetime.utcnow() - dt.timedelta(days=90)).timestamp())
    max_total = max_matches if max_matches is not None else settings.max_pro_matches

    console.log("Fetching pro matches (summaries) for last 3 months...")
    selected_team_ids: set[int] = set()
    teams_cache = paths.cache_dir / "teams.json"
    if teams_cache.exists():
        try:
            with teams_cache.open("r", encoding="utf-8") as f:
                t_raw = json.load(f)
            selected_team_ids = {int(t["team_id"]) for t in t_raw}
        except Exception:
            selected_team_ids = set()
    if not selected_team_ids:
        teams_cfg = load_dev_config("teams_config.yaml") or {}
        top_n_cfg = int(teams_cfg.get("top_rating_teams", 30))
        teams_all = client.fetch_teams(limit=None)
        teams_all.sort(key=lambda t: (t.rating or 0), reverse=True)
        selected_team_ids = {t.team_id for t in teams_all[:top_n_cfg]}

    summaries: List[dict] = []
    last_id: Optional[int] = None
    while True:
        page = client.fetch_pro_matches_page(less_than_match_id=last_id)
        if not page:
            break
        # Keep within last 90 days and only matches where either team is selected
        page_filtered = [
            m
            for m in page
            if m.get("start_time", 0) >= cutoff_unix
            and (
                m.get("radiant_team_id") in selected_team_ids
                or m.get("dire_team_id") in selected_team_ids
            )
        ]
        summaries.extend(page_filtered)
        last_id = min(m["match_id"] for m in page) if page else None
        if page and page[-1].get("start_time", 0) < cutoff_unix:
            break
        if len(summaries) >= max_total:
            summaries = summaries[:max_total]
            break
        time.sleep(settings.request_delay_s)

    saveJson(summaries, paths.cache_dir / "pro_matches_summaries.json")
    console.log(f"Collected {len(summaries)} pro match summaries.")

    details_dir = paths.cache_dir / "pro_match_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    console.log("Fetching detailed match stats (/matches/{id})...")
    for m in tqdm(summaries, desc="Match details"):
        match_id = m.get("match_id")
        if not match_id:
            continue
        dest = details_dir / f"{match_id}.json"
        if dest.exists():
            continue
        try:
            detail = client.fetch_match_detail(match_id)
            saveJson(detail, dest)
        except Exception:
            pass
        time.sleep(settings.request_delay_s)


def fetch_pro_matches_by_config(config_name: str = "pro_matches_config.yaml") -> None:
    cfg = load_dev_config(config_name) or {}
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    max_matches = cfg.get("max_matches", settings.max_pro_matches)
    delay = cfg.get("request_delay_s", settings.request_delay_s)

    client = OpenDotaClient()

    def to_unix(date_str: Optional[str], default: dt.datetime) -> int:
        if not date_str:
            return int(default.timestamp())
        return int(dt.datetime.fromisoformat(date_str).replace(tzinfo=None).timestamp())

    start_unix = to_unix(start_date, dt.datetime.utcnow() - dt.timedelta(days=90))
    end_unix = to_unix(end_date, dt.datetime.utcnow())

    console.log(
        f"Fetching pro matches summaries between {start_unix} and {end_unix}..."
    )
    summaries: List[dict] = []
    last_id: Optional[int] = None
    while True:
        page = client.fetch_pro_matches_page(less_than_match_id=last_id)
        if not page:
            break
        page_filtered = [
            m for m in page if start_unix <= m.get("start_time", 0) <= end_unix
        ]
        summaries.extend(page_filtered)
        last_id = min(m["match_id"] for m in page) if page else None
        if page and (
            page[-1].get("start_time", 0) < start_unix or len(summaries) >= max_matches
        ):
            summaries = summaries[:max_matches]
            break
        time.sleep(delay)

    selected_team_ids: set[int] = set()
    teams_cache = paths.cache_dir / "teams.json"
    if teams_cache.exists():
        try:
            with teams_cache.open("r", encoding="utf-8") as f:
                t_raw = json.load(f)
            selected_team_ids = {int(t["team_id"]) for t in t_raw}
        except Exception:
            selected_team_ids = set()
    if not selected_team_ids:
        teams_cfg2 = load_dev_config("teams_config.yaml") or {}
        top_n_cfg2 = int(teams_cfg2.get("top_rating_teams", 30))
        teams_all2 = client.fetch_teams(limit=None)
        teams_all2.sort(key=lambda t: (t.rating or 0), reverse=True)
        selected_team_ids = {t.team_id for t in teams_all2[:top_n_cfg2]}

    summaries = [
        m
        for m in summaries
        if (
            m.get("radiant_team_id") in selected_team_ids
            or m.get("dire_team_id") in selected_team_ids
        )
    ]
    saveJson(summaries, paths.cache_dir / "pro_matches_summaries.json")
    console.log(
        f"Collected {len(summaries)} pro match summaries (config-driven, filtered by top teams)."
    )

    details_dir = paths.cache_dir / "pro_match_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    console.log("Fetching detailed match stats (/matches/{id})...")
    for m in tqdm(summaries, desc="Match details"):
        mid = m.get("match_id")
        if not mid:
            continue
        dest = details_dir / f"{mid}.json"
        if dest.exists():
            continue
        try:
            detail = client.fetch_match_detail(mid)
            saveJson(detail, dest)
        except Exception:
            pass
        time.sleep(delay)
