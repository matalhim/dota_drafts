from __future__ import annotations

import typer
from rich import print as rprint

from dota_drafts.open_dota.rag import build_index
from dota_drafts.open_dota.rag.config import load_api_keys
from dota_drafts.open_dota.rag.ingest import (
    fetch_and_cache,
    fetch_pro_matches_by_config,
    fetch_pro_matches_last_3_months,
)

load_api_keys()

app = typer.Typer(add_completion=False)


def runFetchAll() -> None:
    rprint("[bold]fetching heroes/teams/players/images...[/bold]")
    fetch_and_cache()
    rprint("[green]core data fetched.[/green]")
    rprint("[bold]fetching pro matches[/bold]")
    try:
        fetch_pro_matches_by_config("pro_matches_config.yaml")
    except Exception:
        fetch_pro_matches_last_3_months()


def runFetchProMatches() -> None:
    rprint("[bold]fetching pro matches[/bold]")
    try:
        fetch_pro_matches_by_config("pro_matches_config.yaml")
    except Exception:
        fetch_pro_matches_last_3_months()


def runBuildRag() -> None:
    rprint("[bold]building chroma index from cache...[/bold]")
    build_index(refresh=False)


@app.command("fetch_all")
def fetch_all() -> None:
    runFetchAll()


@app.command("build_rag")
def build_rag() -> None:
    runBuildRag()


@app.command("pro_matches")
def pro_matches() -> None:
    runFetchProMatches()


if __name__ == "__main__":
    app()
