from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import chromadb
import numpy as np
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import BaseEmbedding
from rich.console import Console
from PIL import Image

from dota_drafts.open_dota.data_sources.opendota_client import (
    Hero,
    Team,
    TeamPlayer,
    ProPlayer,
)
from .ingest import fetch_and_cache
from .config import paths, settings

console = Console()


def loadCached(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def heroToDoc(hero: Hero) -> Document:
    roles = ", ".join(hero.roles or [])
    text = (
        f"Type: Hero\n"
        f"Name: {hero.localized_name}\n"
        f"Internal: {hero.name}\n"
        f"Primary attribute: {hero.primary_attr}\n"
        f"Attack type: {hero.attack_type}\n"
        f"Roles: {roles}\n"
        f"Summary: {hero.localized_name} is a Dota 2 hero."
    )
    metadata = {
        "type": "hero",
        "hero_name": hero.localized_name,
        "short_name": hero.short_name,
        "image_url": hero.image_url,
        "primary_attr": hero.primary_attr,
        "attack_type": hero.attack_type,
        "roles": roles,
    }
    return Document(text=text, metadata=metadata)


def teamToDoc(
    team: Team, players: List[TeamPlayer], pro_players: List[ProPlayer]
) -> Document:
    roster_names = [p.name for p in players if p.is_current_team_member and p.name]
    pro_roster_names = [
        p.name for p in pro_players if p.team_id == team.team_id and p.name
    ]
    roster_str = ", ".join(pro_roster_names or roster_names)
    text = (
        f"Type: Team\n"
        f"Team Name: {team.name or team.tag}\n"
        f"Tag: {team.tag}\n"
        f"Team ID: {team.team_id}\n"
        f"Rating: {team.rating}\n"
        f"Wins: {team.wins} Losses: {team.losses}\n"
        f"Roster: {roster_str if roster_str else 'Unknown'}\n"
        f"Summary: {team.name or team.tag} is a professional Dota 2 team."
    )
    metadata = {
        "type": "team",
        "team_id": team.team_id,
        "name": team.name,
        "tag": team.tag,
        "logo_url": team.logo_url,
        "rating": team.rating,
        "wins": team.wins,
        "losses": team.losses,
        "roster": roster_str,
    }
    return Document(text=text, metadata=metadata)


def playerToDoc(
    team: Team,
    player_name: str,
    account_id: Optional[int],
    avatar_url: Optional[str],
    country_code: Optional[str],
) -> Optional[Document]:
    if not player_name:
        return None
    text = (
        f"Type: Player\n"
        f"Name: {player_name}\n"
        f"Account ID: {account_id}\n"
        f"Current Team: {team.name or team.tag} (ID: {team.team_id})\n"
        f"Country: {country_code}\n"
        f"Summary: {player_name} is a Dota 2 professional player."
    )
    metadata = {
        "type": "player",
        "team_id": team.team_id,
        "team_name": team.name,
        "team_tag": team.tag,
        "account_id": account_id,
        "name": player_name,
        "avatar_url": avatar_url,
        "country_code": country_code,
    }
    return Document(text=text, metadata=metadata)


def makeDocuments(
    heroes: List[Hero],
    teams: List[Team],
    team_players: dict[int, List[TeamPlayer]],
    pro_players: List[ProPlayer],
) -> List[Document]:
    docs: List[Document] = []
    for h in heroes:
        docs.append(heroToDoc(h))
    for t in teams:
        players = team_players.get(t.team_id, [])
        docs.append(teamToDoc(t, players, pro_players))
        for p in pro_players:
            if p.team_id == t.team_id:
                doc = playerToDoc(
                    t,
                    player_name=p.name or "",
                    account_id=p.account_id,
                    avatar_url=p.avatarfull,
                    country_code=p.country_code,
                )
                if doc:
                    docs.append(doc)
    return docs


def matchToDoc(detail: dict, hero_id_to_name: dict[int, str]) -> Optional[Document]:
    match_id = detail.get("match_id")
    if not match_id:
        return None
    radiant_name = detail.get("radiant_name") or "Radiant"
    dire_name = detail.get("dire_name") or "Dire"
    radiant_team_id = detail.get("radiant_team_id")
    dire_team_id = detail.get("dire_team_id")
    league_name = (
        detail.get("league", {}).get("name") or detail.get("league_name") or ""
    )
    radiant_score = detail.get("radiant_score")
    dire_score = detail.get("dire_score")
    duration = detail.get("duration")
    radiant_win = detail.get("radiant_win")
    start_time_unix = detail.get("start_time")
    start_time_iso = (
        datetime.utcfromtimestamp(start_time_unix).isoformat() + "Z"
        if start_time_unix
        else ""
    )

    radiant_players: List[str] = []
    dire_players: List[str] = []
    for p in detail.get("players", []):
        hero_id = p.get("hero_id")
        hero_name = (
            hero_id_to_name.get(hero_id, f"hero_{hero_id}")
            if hero_id
            else "unknown_hero"
        )
        pname = (
            p.get("name") or p.get("personaname") or str(p.get("account_id", "unknown"))
        )
        slot = p.get("player_slot", 0)
        is_radiant = slot < 128
        entry = f"{pname} ({hero_name})"
        if is_radiant:
            radiant_players.append(entry)
        else:
            dire_players.append(entry)

    radiant_players_str = ", ".join(radiant_players)
    dire_players_str = ", ".join(dire_players)
    winner = radiant_name if radiant_win else dire_name

    text = (
        f"Type: Pro Match\n"
        f"Match ID: {match_id}\n"
        f"League: {league_name}\n"
        f"Start Time (UTC): {start_time_iso}\n"
        f"Duration (s): {duration}\n"
        f"Radiant: {radiant_name} (ID: {radiant_team_id}) Score: {radiant_score}\n"
        f"Dire: {dire_name} (ID: {dire_team_id}) Score: {dire_score}\n"
        f"Winner: {winner}\n"
        f"Radiant lineup: {radiant_players_str}\n"
        f"Dire lineup: {dire_players_str}\n"
        f"Summary: Pro match between {radiant_name} and {dire_name}."
    )
    metadata = {
        "type": "pro_match",
        "match_id": match_id,
        "league_name": league_name,
        "start_time": start_time_iso,
        "duration_s": duration,
        "radiant_team_id": radiant_team_id,
        "radiant_name": radiant_name,
        "dire_team_id": dire_team_id,
        "dire_name": dire_name,
        "radiant_score": radiant_score,
        "dire_score": dire_score,
        "radiant_win": radiant_win,
        "radiant_lineup": radiant_players_str,
        "dire_lineup": dire_players_str,
    }
    return Document(text=text, metadata=metadata)


def getClipEmbedding(image_path: Path) -> Optional[np.ndarray]:
    try:
        from transformers import CLIPModel, CLIPProcessor
        import torch

        if not image_path.exists():
            return None

        model_name = "openai/clip-vit-base-patch32"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not hasattr(getClipEmbedding, "model"):
            getClipEmbedding.model = CLIPModel.from_pretrained(model_name).to(device)
            getClipEmbedding.processor = CLIPProcessor.from_pretrained(model_name)
            getClipEmbedding.device = device

        model = getClipEmbedding.model
        processor = getClipEmbedding.processor
        device = getClipEmbedding.device

        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = model.get_image_features(**inputs)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten()
    except Exception as e:
        console.log(
            f"[yellow]Warning: Failed to get CLIP embedding for {image_path}: {e}[/yellow]"
        )
        return None


def addHeroImagesToChroma(heroes: List[Hero], client: chromadb.Client) -> None:
    try:
        collection_name = "dota_hero_images"
        try:
            collection = client.get_collection(collection_name)
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={
                "description": "Hero images with CLIP embeddings for visual matching"
            },
        )

        console.log("Adding hero images to Chroma with CLIP embeddings...")
        added = 0
        skipped = 0

        for hero in heroes:
            image_variants = [
                ("react", paths.images_dir / "heroes" / f"{hero.short_name}.png"),
                (
                    "react_full",
                    paths.images_dir / "heroes" / f"{hero.short_name}_react_full.png",
                ),
                (
                    "legacy_full",
                    paths.images_dir / "heroes" / f"{hero.short_name}_legacy_full.png",
                ),
                ("vert", paths.images_dir / "heroes" / f"{hero.short_name}_vert.jpg"),
                ("lg", paths.images_dir / "heroes" / f"{hero.short_name}_lg.png"),
            ]

            hero_images_added = 0
            for variant, img_path in image_variants:
                if not img_path.exists():
                    continue

                if variant == "react":
                    filename_hero = img_path.stem.lower()
                    if filename_hero != hero.short_name.lower():
                        console.log(
                            f"[yellow]Warning: Filename {img_path.name} doesn't match hero short_name {hero.short_name}[/yellow]"
                        )

                embedding = getClipEmbedding(img_path)
                if embedding is None:
                    continue

                roles_str = ", ".join(hero.roles) if hero.roles else "Unknown"
                doc_text = (
                    f"Hero image: {hero.localized_name} "
                    f"(short name: {hero.short_name}, "
                    f"variant: {variant}, "
                    f"filename: {img_path.name}, "
                    f"roles: {roles_str}, "
                    f"primary attribute: {hero.primary_attr or 'Unknown'}, "
                    f"attack type: {hero.attack_type or 'Unknown'})"
                )

                image_id = f"hero_image_{hero.short_name}_{variant}"

                collection.add(
                    ids=[image_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[
                        {
                            "type": "hero_image",
                            "hero_name": hero.localized_name,
                            "short_name": hero.short_name,
                            "variant": variant,
                            "filename": img_path.name,
                            "hero_id": str(hero.id),
                            "image_path": str(img_path),
                            "image_url": hero.image_url,
                            "roles": roles_str,
                            "primary_attr": hero.primary_attr or "",
                            "attack_type": hero.attack_type or "",
                        }
                    ],
                    documents=[doc_text],
                )
                hero_images_added += 1
                added += 1

            if hero_images_added == 0:
                skipped += 1
                console.log(
                    f"[yellow]Warning: No images found for hero {hero.localized_name} ({hero.short_name})[/yellow]"
                )

        console.log(
            f"[green]Added {added} hero image variants to Chroma (skipped {skipped} heroes with no images)[/green]"
        )
    except Exception as e:
        console.log(
            f"[yellow]Warning: Failed to add hero images to Chroma: {e}[/yellow]"
        )


def build_index(refresh: bool = True) -> VectorStoreIndex:
    heroes: List[Hero]
    teams: List[Team]
    team_players: dict[int, List[TeamPlayer]]
    pro_players: List[ProPlayer]
    if refresh:
        console.log("Fetching and caching latest OpenDota data...")
        heroes, teams, team_players = fetch_and_cache()
        with (paths.cache_dir / "pro_players.json").open("r", encoding="utf-8") as f:
            pro_players_raw = json.load(f)
        pro_players = [ProPlayer(**x) for x in pro_players_raw]
    else:
        h_raw = loadCached(paths.cache_dir / "heroes.json") or []
        t_raw = loadCached(paths.cache_dir / "teams.json") or []
        tp_raw = loadCached(paths.cache_dir / "team_players.json") or {}
        pp_raw = loadCached(paths.cache_dir / "pro_players.json") or []
        heroes = [Hero(**x) for x in h_raw]
        teams = [Team(**x) for x in t_raw]
        team_players = {int(k): [TeamPlayer(**p) for p in v] for k, v in tp_raw.items()}
        pro_players = [ProPlayer(**x) for x in pp_raw]

    docs = makeDocuments(heroes, teams, team_players, pro_players)

    hero_id_to_name = {h.id: h.localized_name for h in heroes}
    summaries_path = paths.cache_dir / "pro_matches_summaries.json"
    details_dir = paths.cache_dir / "pro_match_details"
    if summaries_path.exists() and details_dir.exists():
        try:
            with summaries_path.open("r", encoding="utf-8") as f:
                summaries = json.load(f)
        except Exception:
            summaries = []
        for s in summaries:
            mid = s.get("match_id")
            if not mid:
                continue
            dpath = details_dir / f"{mid}.json"
            if not dpath.exists():
                continue
            try:
                with dpath.open("r", encoding="utf-8") as df:
                    detail = json.load(df)
                mdoc = matchToDoc(detail, hero_id_to_name)
                if mdoc:
                    docs.append(mdoc)
            except Exception:
                continue

    Settings.embed_model = getEmbeddingModel(settings)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

    client = chromadb.PersistentClient(path=str(paths.chroma_dir))
    chroma_collection_name = "dota_rag"
    vector_store = ChromaVectorStore(
        chroma_collection=client.get_or_create_collection(chroma_collection_name)
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    console.log(f"Building index with {len(docs)} documents...")
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True
    )
    console.log("Index build complete.")

    addHeroImagesToChroma(heroes, client)

    return index


def getEmbeddingModel(settings) -> BaseEmbedding:
    provider = settings.embedding_provider.lower()

    if provider == "huggingface":
        return HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    elif provider == "openai":
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(model=settings.embedding_model_name)
        except ImportError:
            console.log(
                "[yellow]llama-index-embeddings-openai not installed, falling back to HuggingFace[/yellow]"
            )
            return HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    elif provider == "cohere":
        try:
            from llama_index.embeddings.cohere import CohereEmbedding

            return CohereEmbedding(
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                model_name=settings.embedding_model_name,
            )
        except ImportError:
            console.log(
                "[yellow]llama-index-embeddings-cohere not installed, falling back to HuggingFace[/yellow]"
            )
            return HuggingFaceEmbedding(model_name=settings.embedding_model_name)
    else:
        console.log(
            f"[yellow]Unknown embedding provider '{provider}', using HuggingFace[/yellow]"
        )
        return HuggingFaceEmbedding(model_name=settings.embedding_model_name)


def load_index() -> VectorStoreIndex:
    Settings.embed_model = getEmbeddingModel(settings)
    client = chromadb.PersistentClient(path=str(paths.chroma_dir))
    vector_store = ChromaVectorStore(
        chroma_collection=client.get_or_create_collection("dota_rag")
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )


def query(question: str, top_k: int = 5) -> str:
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(question)
    return str(response)
