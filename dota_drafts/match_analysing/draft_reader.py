from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

import cv2  # type: ignore

from dota_drafts.open_dota.config_loader import load_dev_config
from dota_drafts.open_dota.rag.config import paths


@dataclass
class DraftLayout:
    width: int
    height: int
    radiant_slots: List[Tuple[int, int, int, int]]  # list of (x, y, w, h)
    dire_slots: List[Tuple[int, int, int, int]]


def loadLayout(image: Image.Image) -> DraftLayout:
    cfg = load_dev_config("draft_layout.yaml") or {}
    img_w, img_h = image.size

    if cfg:
        width = int(cfg.get("width", img_w))
        height = int(cfg.get("height", img_h))

        def parse_slots(key: str) -> List[Tuple[int, int, int, int]]:
            slots = []
            for s in cfg.get(key) or []:
                # Allow relative (0..1) or absolute pixels
                x, y, w, h = s
                if 0 < x <= 1 and 0 < y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                    slots.append(
                        (int(x * img_w), int(y * img_h), int(w * img_w), int(h * img_h))
                    )
                else:
                    slots.append((int(x), int(y), int(w), int(h)))
            return slots

        r_slots = parse_slots("radiant_slots")
        d_slots = parse_slots("dire_slots")
        if r_slots and d_slots:
            return DraftLayout(
                width=img_w, height=img_h, radiant_slots=r_slots, dire_slots=d_slots
            )

    # Fallback heuristic: typical 1920x1080 draft-like layout (approximate)
    # Two rows near bottom/top with 5 equal slots each.
    w, h = img_w, img_h
    slot_w = int(0.085 * w)
    slot_h = int(0.15 * h)
    gap = int(0.015 * w)
    # Radiant row near bottom-left
    rx0 = int(0.18 * w)
    ry = int(0.78 * h)
    radiant = [(rx0 + i * (slot_w + gap), ry, slot_w, slot_h) for i in range(5)]
    # Dire row near top-right
    dx0 = int(0.72 * w) - 4 * (slot_w + gap)
    dy = int(0.08 * h)
    dire = [(dx0 + i * (slot_w + gap), dy, slot_w, slot_h) for i in range(5)]
    return DraftLayout(width=w, height=h, radiant_slots=radiant, dire_slots=dire)


def loadHeroes() -> List[Dict]:
    f = paths.cache_dir / "heroes.json"
    if not f.exists():
        raise FileNotFoundError(f"Missing heroes cache: {f}")
    with f.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def loadTeams() -> List[Dict]:
    f = paths.cache_dir / "teams.json"
    if not f.exists():
        return []
    with f.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def buildHeroGallery() -> Dict[str, Path]:
    gallery: Dict[str, Path] = {}
    root = paths.images_dir / "heroes"
    if not root.exists():
        return gallery
    for p in root.glob("*.png"):
        short_name = p.stem  # e.g., 'axe'
        gallery[short_name] = p
    return gallery


class ClipMatcher:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.hero_gallery = buildHeroGallery()
        self.hero_db = loadHeroes()
        self.short_to_name = {
            h["name"].replace("npc_dota_hero_", ""): h.get("localized_name", h["name"])
            for h in self.hero_db
        }
        self.embCachePath = paths.cache_dir / "hero_clip_embeds.pt"
        self.embeds, self.keys = self.prepareGalleryEmbeddings()

    def prepareGalleryEmbeddings(self) -> Tuple[torch.Tensor, List[str]]:
        if self.embCachePath.exists():
            data = torch.load(self.embCachePath, map_location=self.device)
            return data["embeds"].to(self.device), data["keys"]
        imgs: List[Image.Image] = []
        keys: List[str] = []
        for k, p in self.hero_gallery.items():
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(img)
                keys.append(k)
            except Exception:
                continue
        if not imgs:
            return torch.empty(0), []
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        torch.save({"embeds": emb.cpu(), "keys": keys}, self.embCachePath)
        return emb, keys

    def match(self, crop: Image.Image, top_k: int = 3) -> List[Tuple[str, float, str]]:
        if self.embeds.numel() == 0 or not self.keys:
            return []
        img = ImageOps.fit(crop.convert("RGB"), (224, 224))
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            q = self.model.get_image_features(**inputs)
            q = q / q.norm(dim=-1, keepdim=True)
            sim = (q @ self.embeds.T).squeeze(0)
            scores, idx = torch.topk(sim, k=min(top_k, sim.shape[0]))
        out: List[Tuple[str, float, str]] = []
        for s, i in zip(scores.tolist(), idx.tolist()):
            short = self.keys[i]
            out.append((short, float(s), self.short_to_name.get(short, short)))
        return out


def fuzzyBestMatch(query: str, candidates: List[str]) -> Optional[str]:
    if not query or not candidates:
        return None
    query = query.lower()
    best = None
    best_score = 0.0
    for c in candidates:
        s = difflib.SequenceMatcher(None, query, c.lower()).ratio()
        if s > best_score:
            best = c
            best_score = s
    if best_score < 0.5:
        return None
    return best


def extractText(image: Image.Image) -> str:
    if pytesseract is None:
        return ""
    # Basic enhancement for OCR
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)
    arr = cv2.bilateralFilter(arr, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    pil = Image.fromarray(th)
    try:
        return pytesseract.image_to_string(pil)
    except Exception:
        return ""


def identifyTeamsFromText(
    image: Image.Image, teams: List[Dict]
) -> Tuple[Optional[Dict], Optional[Dict]]:
    text = extractText(image)
    names = [t.get("name") or "" for t in teams] + [t.get("tag") or "" for t in teams]
    tokens = [tok for tok in (text or "").split() if tok.strip()]
    candidates: List[str] = []
    for tok in tokens:
        m = fuzzyBestMatch(tok, names)
        if m:
            candidates.append(m)
    # pick two distinct best matching teams by name/tag
    found: List[Dict] = []
    seen = set()
    for cand in candidates:
        for t in teams:
            if cand.lower() in (
                (t.get("name") or "").lower(),
                (t.get("tag") or "").lower(),
            ):
                if t["team_id"] not in seen:
                    found.append(t)
                    seen.add(t["team_id"])
                break
        if len(found) >= 2:
            break
    if len(found) == 2:
        return found[0], found[1]
    return None, None


def analyze_draft_image(
    image_path: str | Path,
    use_ocr: bool = True,
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    layout = loadLayout(img)
    teams = loadTeams()
    matcher = ClipMatcher()

    radiant_picks: List[Dict] = []
    dire_picks: List[Dict] = []

    for x, y, w, h in layout.radiant_slots:
        crop = img.crop((x, y, x + w, y + h))
        hits = matcher.match(crop, top_k=3)
        if hits:
            short, score, name = hits[0]
            radiant_picks.append({"short_name": short, "name": name, "score": score})
        else:
            radiant_picks.append({"short_name": None, "name": None, "score": 0.0})

    for x, y, w, h in layout.dire_slots:
        crop = img.crop((x, y, x + w, y + h))
        hits = matcher.match(crop, top_k=3)
        if hits:
            short, score, name = hits[0]
            dire_picks.append({"short_name": short, "name": name, "score": score})
        else:
            dire_picks.append({"short_name": None, "name": None, "score": 0.0})

    rad_team: Optional[Dict] = None
    dire_team: Optional[Dict] = None
    if use_ocr and teams:
        rt, dt = identifyTeamsFromText(img, teams)
        rad_team, dire_team = rt, dt

    return {
        "radiant_team": rad_team,
        "dire_team": dire_team,
        "radiant_heroes": radiant_picks,
        "dire_heroes": dire_picks,
        "meta": {
            "image_path": str(image_path),
            "layout_used": {
                "width": layout.width,
                "height": layout.height,
                "radiant_slots": layout.radiant_slots,
                "dire_slots": layout.dire_slots,
            },
            "ocr_used": use_ocr and (pytesseract is not None),
        },
    }
