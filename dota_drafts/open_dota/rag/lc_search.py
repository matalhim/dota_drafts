from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from dota_drafts.open_dota.rag.config import paths
from dota_drafts.open_dota.rag.queries import (
    EXTRACTION_PROMPT,
    HERO_SUGGESTION_PROMPT,
    HERO_BBOX_PROMPT,
    RAG_SYSTEM_PROMPT,
    REFINE_DRAFT_FEEDBACK_PROMPT,
    get_draft_analysis_prompt,
)
from dota_drafts.open_dota.rag.utils import (
    get_empty_keywords,
    image_to_base64,
    parse_json_from_response,
    try_models,
)

CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_DEVICE = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
try:
    from langchain_together import ChatTogether
except Exception:
    ChatTogether = None


@dataclass(frozen=True)
class RAGConfig:
    provider: str
    model: str
    top_k: int = 8


def getLlmFromConfig(cfg: RAGConfig):
    return get_llm(cfg.provider, cfg.model)


def get_llm(provider: str, model: str):
    p = provider.lower()
    if p == "google":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai not installed")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required")
        return ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=0.2)
    if p == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai not installed")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        return ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url, temperature=0.2
        )
    if p == "together" or p == "another":
        if ChatTogether is None:
            raise RuntimeError("langchain-together not installed")
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is required")
        return ChatTogether(model=model, temperature=0.2, api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def get_vectorstore() -> Chroma:
    client = chromadb.PersistentClient(path=str(paths.chroma_dir))
    collection = client.get_or_create_collection("dota_rag")
    return Chroma(
        client=client,
        collection_name="dota_rag",
        embedding_function=None,
    )


def get_hero_images_vectorstore() -> Optional[chromadb.Collection]:
    try:
        client = chromadb.PersistentClient(path=str(paths.chroma_dir))
        collection = client.get_collection("dota_hero_images")
        return collection
    except Exception:
        return None


def build_rag_chain(cfg: RAGConfig, vs: Optional[Chroma] = None) -> Runnable:
    llm = getLlmFromConfig(cfg)
    vs = vs or get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": cfg.top_k})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT + "Context:\n{context}"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        chunks = []
        for d in docs:
            meta = d.metadata or {}
            src = (
                meta.get("name")
                or meta.get("tag")
                or meta.get("short_name")
                or meta.get("team_id")
                or ""
            )
            chunks.append(f"[{src}] {d.page_content}")
        return "\n\n".join(chunks)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(
    question: str,
    provider: str = "google",
    model: str = "gemini-1.5-flash",
    top_k: int = 8,
) -> str:
    cfg = RAGConfig(provider=provider, model=model, top_k=top_k)
    chain = build_rag_chain(cfg)
    return chain.invoke(question)


def extract_keywords_from_draft(
    image_path: Path,
    provider: str = "google",
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    from PIL import Image

    if not image_path.exists():
        return get_empty_keywords()

    img = Image.open(image_path).convert("RGB")

    def create_llm(model_name: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model=model_name, api_key=api_key, temperature=0.1
        )

    model_names = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro",
    ]
    if model not in model_names:
        model_names.insert(0, model)

    try:
        vision_llm = try_models(
            model_names, create_llm, "No working Gemini model found"
        )
        img_b64 = image_to_base64(img)

        message = HumanMessage(
            content=[
                {"type": "text", "text": EXTRACTION_PROMPT},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]
        )
        response = vision_llm.invoke([message])
        return parse_json_from_response(response.content)
    except Exception as e:
        import logging

        logging.debug(f"Error extracting keywords: {e}")
        return get_empty_keywords()


def normalize_hero_name(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def normalize_team_name(name: str) -> str:
    if not name:
        return ""
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def refine_players(keywords: Dict[str, Any]) -> Dict[str, Any]:
    vs = get_vectorstore()
    refined = keywords.copy()

    for side in ["left_team_players", "right_team_players"]:
        players = keywords.get(side, [])
        refined_players_list = []
        for player in players:
            if not player:
                continue
            try:
                docs = vs.similarity_search(
                    f"Dota 2 player {player}", k=1, filter={"type": "player"}
                )
                if docs:
                    meta = docs[0].metadata or {}
                    refined_name = meta.get("name") or player
                    refined_players_list.append(refined_name)
                else:
                    refined_players_list.append(player)
            except Exception:
                refined_players_list.append(player)
        refined[side] = refined_players_list

    return refined


def refine_teams_by_players(keywords: Dict[str, Any]) -> Dict[str, Any]:
    vs = get_vectorstore()
    refined = keywords.copy()

    teams = keywords.get("teams", [])
    if not teams:
        return refined

    left_players = keywords.get("left_team_players", [])
    right_players = keywords.get("right_team_players", [])

    refined_teams = []
    for i, team_name in enumerate(teams):
        if not team_name:
            refined_teams.append(team_name)
            continue

        players_to_check = left_players if i == 0 else right_players
        best_match = team_name

        try:
            for player in players_to_check[:3]:
                if not player:
                    continue
                docs = vs.similarity_search(
                    f"Dota 2 player {player}", k=1, filter={"type": "player"}
                )
                if docs:
                    meta = docs[0].metadata or {}
                    player_team = meta.get("team_name") or meta.get("tag")
                    if player_team:
                        best_match = player_team
                        break
        except Exception:
            pass

        refined_teams.append(best_match)

    refined["teams"] = refined_teams
    return refined


def get_all_heroes_context(top_k: int = 200) -> str:
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(
            "Dota 2 heroes list", k=top_k, filter={"type": "hero"}
        )

        context_parts = []
        for doc in docs:
            meta = doc.metadata or {}
            hero_name = meta.get("hero_name", "")
            if not hero_name:
                lines = doc.page_content.split("\n")
                for line in lines:
                    if line.startswith("Name: "):
                        hero_name = line.replace("Name: ", "").strip()
                        break
            if not hero_name:
                continue
            roles = meta.get("roles", "")
            attr = meta.get("primary_attr", "")
            attack = meta.get("attack_type", "")
            context_parts.append(
                f"Hero {hero_name}. Roles: {roles}. Primary attribute: {attr}. Attack type: {attack}."
            )
        return "\n".join(context_parts)
    except Exception as e:
        import logging

        logging.debug(f"Error getting heroes context: {e}")
        return ""


def suggest_heroes_from_draft(
    image_path: Path,
    rag_context: str,
    provider: str = "google",
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    from PIL import Image

    if not image_path.exists():
        return {"left_team_heroes": [], "right_team_heroes": []}

    img = Image.open(image_path).convert("RGB")

    def create_llm(model_name: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model=model_name, api_key=api_key, temperature=0.1
        )

    model_names = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro",
    ]
    if model not in model_names:
        model_names.insert(0, model)

    try:
        vision_llm = try_models(
            model_names, create_llm, "No working Gemini model found"
        )
        img_b64 = image_to_base64(img)

        prompt = HERO_SUGGESTION_PROMPT.format(rag_context=rag_context)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]
        )
        response = vision_llm.invoke([message])
        return parse_json_from_response(response.content)
    except Exception as e:
        import logging

        logging.debug(f"Error suggesting heroes: {e}")
        return {"left_team_heroes": [], "right_team_heroes": []}


def detect_hero_bboxes(
    image_path: Path,
    provider: str = "google",
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    from PIL import Image

    if not image_path.exists():
        return {"left_team_heroes_bboxes": [], "right_team_heroes_bboxes": []}

    img = Image.open(image_path).convert("RGB")

    def create_llm(model_name: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model=model_name, api_key=api_key, temperature=0.1
        )

    model_names = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro",
    ]
    if model not in model_names:
        model_names.insert(0, model)

    try:
        vision_llm = try_models(
            model_names, create_llm, "No working Gemini model found for bbox detection"
        )
        img_b64 = image_to_base64(img)

        prompt = HERO_BBOX_PROMPT
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]
        )
        response = vision_llm.invoke([message])
        return parse_json_from_response(response.content)
    except Exception as e:
        import logging

        logging.debug(f"Error detecting hero bboxes: {e}")
        return {"left_team_heroes_bboxes": [], "right_team_heroes_bboxes": []}


def ensureClipModel():
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_DEVICE
    if CLIP_MODEL is not None and CLIP_PROCESSOR is not None:
        return
    from transformers import CLIPModel, CLIPProcessor
    import torch

    CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
        CLIP_DEVICE
    )
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def clipEmbeddingFromCrop(crop_img: Image.Image) -> List[float]:
    import torch

    ensureClipModel()
    assert CLIP_MODEL is not None and CLIP_PROCESSOR is not None and CLIP_DEVICE

    img = crop_img.convert("RGB")
    inputs = CLIP_PROCESSOR(images=img, return_tensors="pt")
    inputs = {k: v.to(CLIP_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        emb = CLIP_MODEL.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().tolist()


def defaultSlots(
    width: int, height: int
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    slot_w = int(0.085 * width)
    slot_h = int(0.15 * height)
    gap = int(0.015 * width)
    left_start_x = int(0.18 * width)
    left_y = int(0.78 * height)
    right_start_x = int(0.72 * width) - 4 * (slot_w + gap)
    right_y = int(0.08 * height)

    radiant = [
        (left_start_x + i * (slot_w + gap), left_y, slot_w, slot_h) for i in range(5)
    ]
    dire = [
        (right_start_x + i * (slot_w + gap), right_y, slot_w, slot_h) for i in range(5)
    ]
    return radiant, dire


def matchSlotsWithChroma(
    image: Image.Image, slots: List[Tuple[int, int, int, int]], collection
) -> List[str]:
    heroes: List[str] = []
    for x, y, w, h in slots:
        crop = image.crop((x, y, x + w, y + h))
        try:
            embedding = clipEmbeddingFromCrop(crop)
        except Exception:
            continue
        try:
            result = collection.query(
                query_embeddings=[embedding],
                n_results=1,
                include=["metadatas"],
            )
        except Exception:
            continue
        metadatas = result.get("metadatas") or []
        if not metadatas or not metadatas[0]:
            continue
        meta = metadatas[0][0] or {}
        hero_name = meta.get("hero_name") or meta.get("short_name")
        if hero_name:
            heroes.append(hero_name)
    return heroes


def sanitizeBboxes(raw_list: Any) -> List[Tuple[int, int, int, int]]:
    result: List[Tuple[int, int, int, int]] = []
    if not isinstance(raw_list, list):
        return result
    for item in raw_list:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 4
            and all(isinstance(v, (int, float)) for v in item[:4])
        ):
            x, y, w, h = item[:4]
            result.append((int(x), int(y), int(w), int(h)))
        else:
            result.append((0, 0, 0, 0))
    return result


def detect_heroes_with_clip(
    image_path: Path, provider: str = "google", model: str = "gemini-2.5-flash"
) -> Dict[str, List[str]]:
    if not image_path.exists():
        return {"left_team_heroes": [], "right_team_heroes": []}

    collection = get_hero_images_vectorstore()
    if collection is None:
        return {"left_team_heroes": [], "right_team_heroes": []}

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return {"left_team_heroes": [], "right_team_heroes": []}

    bbox_result = detect_hero_bboxes(image_path, provider=provider, model=model)
    left_boxes = sanitizeBboxes(bbox_result.get("left_team_heroes_bboxes"))
    right_boxes = sanitizeBboxes(bbox_result.get("right_team_heroes_bboxes"))

    if len(left_boxes) == 5 and len(right_boxes) == 5:
        radiant_slots, dire_slots = left_boxes, right_boxes
    else:
        radiant_slots, dire_slots = defaultSlots(img.width, img.height)

    left = matchSlotsWithChroma(img, radiant_slots, collection)
    right = matchSlotsWithChroma(img, dire_slots, collection)

    return {
        "left_team_heroes": left,
        "right_team_heroes": right,
    }


def get_team_match_history(team_name: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(
            f"Dota 2 pro match {team_name}", k=limit, filter={"type": "pro_match"}
        )

        matches = []
        for doc in docs:
            meta = doc.metadata or {}
            radiant_name = meta.get("radiant_name", "")
            dire_name = meta.get("dire_name", "")

            if (
                team_name.lower() not in radiant_name.lower()
                and team_name.lower() not in dire_name.lower()
            ):
                continue

            match_id = meta.get("match_id", "")
            start_time = meta.get("start_time", 0)
            duration = meta.get("duration", 0)
            radiant_win = meta.get("radiant_win", False)

            won = (radiant_win and team_name.lower() in radiant_name.lower()) or (
                not radiant_win and team_name.lower() in dire_name.lower()
            )

            matches.append(
                {
                    "match_id": match_id,
                    "start_time": start_time,
                    "duration_s": duration,
                    "won": won,
                    "radiant_name": radiant_name,
                    "dire_name": dire_name,
                    "radiant_score": meta.get("radiant_score", ""),
                    "dire_score": meta.get("dire_score", ""),
                    "lineup": doc.page_content[:200] if doc.page_content else "",
                }
            )

        return matches
    except Exception as e:
        import logging

        logging.debug(f"Error getting team match history: {e}")
        return []


def get_team_statistics_summary(team_name: str, players: List[str]) -> str:
    vs = get_vectorstore()
    full_data_parts = []
    results_queue = Queue()

    def get_team_info():
        try:
            team_docs = vs.similarity_search(
                f"Dota 2 team {team_name}",
                k=1,
                filter={"type": "team"},
            )
            if team_docs:
                doc = team_docs[0]
                results_queue.put(("team", doc.page_content))
        except Exception as e:
            import logging

            logging.debug(f"Error getting team info: {e}")

    def get_match_history():
        try:
            recent_matches = get_team_match_history(team_name, limit=50)
            if recent_matches:
                match_lines = [
                    f"=== MATCH HISTORY {team_name} (last {len(recent_matches)} matches) ==="
                ]
                for i, match in enumerate(recent_matches, 1):
                    match_id = match.get("match_id", "N/A")
                    start_time = match.get("start_time", "N/A")
                    duration = match.get("duration_s", 0)
                    won = match.get("won", False)
                    radiant_name = match.get("radiant_name", "")
                    dire_name = match.get("dire_name", "")
                    lineup = match.get("lineup", "")
                    radiant_score = match.get("radiant_score", "")
                    dire_score = match.get("dire_score", "")

                    result = "WIN" if won else "LOSS"
                    match_lines.append(
                        f"Match {i}: {result} | ID: {match_id} | {radiant_name} vs {dire_name} | "
                        f"Score: {radiant_score}-{dire_score} | Duration: {duration}s | Lineup: {lineup}"
                    )
                match_lines.append("")

                match_docs = vs.similarity_search(
                    f"Dota 2 pro match {team_name}",
                    k=min(30, len(recent_matches) * 2),
                    filter={"type": "pro_match"},
                )

                if match_docs:
                    match_lines.append(f"=== FULL MATCH DATA {team_name} ===")
                    for doc in match_docs[:20]:
                        meta = doc.metadata or {}
                        radiant_name = meta.get("radiant_name", "")
                        dire_name = meta.get("dire_name", "")
                        if (
                            team_name.lower() in radiant_name.lower()
                            or team_name.lower() in dire_name.lower()
                        ):
                            match_lines.append(doc.page_content)
                            match_lines.append("---")
                    match_lines.append("")

                results_queue.put(("matches", "\n".join(match_lines)))
        except Exception as e:
            import logging

            logging.debug(f"Error getting match history: {e}")

    def get_player_info(player_name: str):
        try:
            player_docs = vs.similarity_search(
                f"Dota 2 player {player_name}",
                k=1,
                filter={"type": "player"},
            )
            if player_docs:
                doc = player_docs[0]
                player_meta = doc.metadata or {}
                player_team = player_meta.get("team_name", "")
                if player_team and team_name.lower() in player_team.lower():
                    results_queue.put(("player", player_name, doc.page_content))
        except Exception as e:
            import logging

            logging.debug(f"Error getting player info for {player_name}: {e}")

    def get_hero_info():
        try:
            hero_docs = vs.similarity_search(
                f"Dota 2 team {team_name} heroes picks",
                k=20,
                filter={"type": "pro_match"},
            )
            if hero_docs:
                hero_lines = [f"=== HEROES IN MATCHES {team_name} ==="]
                heroes_mentioned = set()
                for doc in hero_docs[:15]:
                    content = doc.page_content
                    for line in content.split("\n"):
                        if "hero" in line.lower() or "pick" in line.lower():
                            if line not in heroes_mentioned:
                                heroes_mentioned.add(line)
                                hero_lines.append(line)
                hero_lines.append("")
                results_queue.put(("heroes", "\n".join(hero_lines)))
        except Exception as e:
            import logging

            logging.debug(f"Error getting hero info: {e}")

    try:
        threads = []

        t1 = threading.Thread(target=get_team_info)
        t1.start()
        threads.append(t1)

        t2 = threading.Thread(target=get_match_history)
        t2.start()
        threads.append(t2)

        t3 = threading.Thread(target=get_hero_info)
        t3.start()
        threads.append(t3)

        player_threads = []
        for player_name in players[:5]:
            t = threading.Thread(target=get_player_info, args=(player_name,))
            t.start()
            player_threads.append(t)
        threads.extend(player_threads)

        for t in threads:
            t.join()
        team_info = None
        match_info = None
        hero_info = None
        player_infos = []

        while not results_queue.empty():
            result = results_queue.get()
            if result[0] == "team":
                team_info = result[1]
            elif result[0] == "matches":
                match_info = result[1]
            elif result[0] == "heroes":
                hero_info = result[1]
            elif result[0] == "player":
                player_infos.append((result[1], result[2]))

        if team_info:
            full_data_parts.append(f"=== TEAM INFORMATION {team_name} ===")
            full_data_parts.append(team_info)
            full_data_parts.append("")

        if match_info:
            full_data_parts.append(match_info)

        if players and player_infos:
            full_data_parts.append(f"=== PLAYER INFORMATION {team_name} ===")
            for player_name, player_content in player_infos:
                full_data_parts.append(f"--- Player: {player_name} ---")
                full_data_parts.append(player_content)
                full_data_parts.append("")

        if hero_info:
            full_data_parts.append(hero_info)

    except Exception as e:
        import logging

        logging.debug(f"Error getting team statistics: {e}")
        full_data_parts.append(f"Error retrieving data: {e}")

    return (
        "\n".join(full_data_parts)
        if full_data_parts
        else f"Data for team {team_name} not found in RAG."
    )


def analyze_draft_and_predict_winner(
    image_path: Path,
    left_team: str,
    right_team: str,
    left_heroes: List[str],
    right_heroes: List[str],
    team_summaries: Dict[str, str],
    provider: str = "google",
    model: str = "gemini-2.5-flash",
) -> str:
    cfg = RAGConfig(provider=provider, model=model, top_k=8)
    llm = getLlmFromConfig(cfg)

    prompt_text = get_draft_analysis_prompt(
        left_team=left_team,
        right_team=right_team,
        left_heroes=left_heroes,
        right_heroes=right_heroes,
        team_statistics=team_summaries,
    )

    try:
        response = llm.invoke(prompt_text)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)
    except Exception as e:
        import logging

        logging.debug(f"Error analyzing draft: {e}")
        return f"Error analyzing draft: {e}"


def refine_draft_from_feedback(
    image_path: Path,
    current_teams: List[str],
    current_left_heroes: List[str],
    current_right_heroes: List[str],
    user_feedback: str,
    provider: str = "google",
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    from PIL import Image

    if not image_path.exists():
        return {
            "teams": current_teams,
            "left_team_heroes": current_left_heroes,
            "right_team_heroes": current_right_heroes,
        }

    img = Image.open(image_path).convert("RGB")

    def create_llm(model_name: str):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model=model_name, api_key=api_key, temperature=0.1
        )

    model_names = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro",
    ]
    if model not in model_names:
        model_names.insert(0, model)

    try:
        vision_llm = try_models(
            model_names, create_llm, "No working Gemini model found"
        )
        img_b64 = image_to_base64(img)

        left_team_name = current_teams[0] if current_teams else "Unknown"
        right_team_name = current_teams[1] if len(current_teams) > 1 else "Unknown"
        left_heroes_str = (
            ", ".join(current_left_heroes) if current_left_heroes else "not identified"
        )
        right_heroes_str = (
            ", ".join(current_right_heroes)
            if current_right_heroes
            else "not identified"
        )
        prompt = REFINE_DRAFT_FEEDBACK_PROMPT.format(
            left_team=left_team_name,
            right_team=right_team_name,
            left_heroes=left_heroes_str,
            right_heroes=right_heroes_str,
            user_feedback=user_feedback,
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
            ]
        )
        response = vision_llm.invoke([message])
        return parse_json_from_response(response.content)
    except Exception as e:
        import logging

        logging.debug(f"Error refining draft from feedback: {e}")
        return {
            "teams": current_teams,
            "left_team_heroes": current_left_heroes,
            "right_team_heroes": current_right_heroes,
        }
