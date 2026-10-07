#!/usr/bin/env python3
"""Refresh topic news JSON files using ONLY the Gemini API.

Looks for API key in env (in order): Gemini, GEMINI_API_KEY, GOOGLE_API_KEY.
Does not call Cursor models or web search.

Default outputs:
  AI.News.json
  Sport.News.json
  News.json
  IndustrialAutomation.News.json
  Tech.News.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jdatetime

DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ("gemini-3.5-flash-lite",)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
REPO_ROOT = Path(__file__).resolve().parents[1]
TEHRAN = ZoneInfo("Asia/Tehran")

COMMON_CATEGORIES = {
    "news",
    "product",
    "regulation",
    "research",
    "topic",
    "sideline",
    "match",
    "transfer",
    "result",
    "politics",
    "economy",
    "society",
    "world",
    "industry",
    "company",
}

FEEDS: list[dict] = [
    {
        "key": "ai",
        "filename": "AI.News.json",
        "topic": "artificial-intelligence",
        "id_prefix": "ai",
        "desk": "artificial intelligence",
        "scope": (
            "Exactly 10 items about AI (models, labs, agents, research, "
            "regulation, industry sidelines)."
        ),
        "system": "You output only strict JSON for an AI news feed. No markdown.",
    },
    {
        "key": "sport",
        "filename": "Sport.News.json",
        "topic": "sports",
        "id_prefix": "sport",
        "desk": "sports (Iran and international)",
        "scope": (
            "Exactly 10 items of fresh sports news and updates covering BOTH "
            "Iranian sports and international sports (football/soccer, other "
            "major sports, transfers, matches, federation news). Mix Iran and "
            "world coverage; do not focus only on one side."
        ),
        "system": "You output only strict JSON for a sports news feed. No markdown.",
    },
    {
        "key": "general",
        "filename": "News.json",
        "topic": "iran-and-world-news",
        "id_prefix": "news",
        "desk": "major Iran and world developments",
        "scope": (
            "Exactly 10 items about important political, economic, social, and "
            "geopolitical developments in Iran and the world. Prefer major "
            "breaking or consequential stories over niche/local fluff."
        ),
        "system": "You output only strict JSON for a general news feed. No markdown.",
    },
    {
        "key": "industrial",
        "filename": "IndustrialAutomation.News.json",
        "topic": "industrial-automation",
        "id_prefix": "ia",
        "desk": "industrial automation with emphasis on Siemens",
        "scope": (
            "Exactly 10 items about industrial automation, OT/ICS, PLC/SCADA/DCS, "
            "factory digitalization, robotics on the plant floor, and related "
            "industry progress. Put stronger focus on Siemens (products, "
            "software, plants, partnerships, TIA Portal, SIMATIC, Industrial Edge, "
            "MindSphere/Insights Hub) while still allowing other major vendors "
            "when relevant."
        ),
        "system": (
            "You output only strict JSON for an industrial automation news feed. "
            "No markdown."
        ),
    },
    {
        "key": "tech",
        "filename": "Tech.News.json",
        "topic": "technology",
        "id_prefix": "tech",
        "desk": "technology",
        "scope": (
            "Exactly 10 items about day-to-day technology news and developments "
            "(consumer tech, platforms, chips, software, startups, big tech "
            "product moves). Prefer broad tech over pure AI research unless the "
            "story is a major tech-industry event."
        ),
        "system": "You output only strict JSON for a technology news feed. No markdown.",
    },
]


def resolve_api_key() -> str:
    for name in ("Gemini", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def tehran_jalali_now() -> str:
    now = datetime.now(TEHRAN)
    return jdatetime.datetime.fromgregorian(datetime=now).strftime("%Y/%m/%d %H:%M")


def feed_path(feed: dict) -> Path:
    return REPO_ROOT / feed["filename"]


def load_previous_titles(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    titles = []
    for item in data.get("items") or []:
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return titles


def build_prompt(feed: dict, collected_at: str, previous_titles: list[str]) -> str:
    avoid = "\n".join(f"- {t}" for t in previous_titles[:10]) or "- (none)"
    prefix = feed["id_prefix"]
    return f"""You are a news desk for {feed["desk"]}.

Return ONLY a valid JSON object (no markdown fences, no commentary) with exactly this shape:
{{
  "items": [
    {{
      "id": "{prefix}-YYYYMMDD-NN",
      "title": "Persian title",
      "summary": "Persian summary, about 280-350 characters (roughly 3-4 sentences)",
      "category": "news|product|regulation|research|topic|sideline|match|transfer|result|politics|economy|society|world|industry|company",
      "published_at": "YYYY/MM/DD HH:MM",
      "source": "outlet or org name",
      "source_url": "https://...",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

Rules:
- {feed["scope"]}
- Prefer fresh, real recent stories with real canonical source_url values.
- title and summary MUST be Persian.
- Each summary MUST be substantially longer than one short line: target ~280-350 Persian characters (about 3-4 sentences), roughly double a single-sentence blurb. Include what happened, why it matters, and one concrete detail (product/model/org/number) when available. Do not pad with filler.
- published_at MUST be Jalali calendar + Asia/Tehran as YYYY/MM/DD HH:MM.
- published_at must not be after collected_at={collected_at}.
- id must be unique like {prefix}-20261007-01 .. {prefix}-20261007-10 using today's Gregorian date in the id prefix.
- Avoid repeating these previous titles when possible:
{avoid}
"""


def gemini_url(model: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )


def call_gemini(prompt: str, system: str, model: str) -> str:
    api_key = resolve_api_key()
    if not api_key:
        raise SystemExit(
            "Gemini API key missing. Add a Cloud Agent secret named "
            "`Gemini` or `GEMINI_API_KEY`, then start a NEW agent."
        )

    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        gemini_url(model),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "News-Json-GeminiRefresh/2.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API network error: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"Gemini API timeout: {exc}") from exc

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        content = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {payload!r}") from exc

    if not content.strip():
        raise RuntimeError(f"Empty Gemini content: {payload!r}")
    return content.strip()


def call_gemini_with_fallback(prompt: str, system: str) -> tuple[str, str]:
    models = [GEMINI_MODEL]
    for model in FALLBACK_MODELS:
        if model not in models:
            models.append(model)

    last_error: Exception | None = None
    for model in models:
        for attempt in range(1, 4):
            print(
                f"Calling Gemini model {model} (attempt {attempt}) ...",
                file=sys.stderr,
            )
            try:
                return call_gemini(prompt, system, model), model
            except RuntimeError as exc:
                last_error = exc
                msg = str(exc)
                print(msg, file=sys.stderr)
                # 404 on model: skip to next model immediately
                if "HTTP 404" in msg:
                    break
                time.sleep(min(8 * attempt, 24))
    raise SystemExit(str(last_error) if last_error else "Gemini call failed")


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("root must be object")
    return data


def normalize_items(raw_items: list, collected_at: str, id_prefix: str) -> list[dict]:
    if not isinstance(raw_items, list) or len(raw_items) != 10:
        raise ValueError(
            f"expected exactly 10 items, got "
            f"{len(raw_items) if isinstance(raw_items, list) else type(raw_items)}"
        )

    gregorian = datetime.now(TEHRAN).strftime("%Y%m%d")
    out: list[dict] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"item {idx} is not an object")

        default_id = f"{id_prefix}-{gregorian}-{idx:02d}"
        item_id = str(item.get("id") or default_id).strip()
        if item_id in seen_ids:
            item_id = default_id
        seen_ids.add(item_id)

        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        category = str(item.get("category") or "news").strip().lower()
        if category not in COMMON_CATEGORIES:
            category = "news"
        published_at = str(item.get("published_at") or collected_at).strip()
        source = str(item.get("source") or "Unknown").strip()
        source_url = str(item.get("source_url") or "").strip()
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        tags = [str(t).strip() for t in tags if str(t).strip()][:8]

        if not title or not summary:
            raise ValueError(f"item {idx} missing title/summary")
        if not source_url.startswith("http"):
            raise ValueError(f"item {idx} missing http(s) source_url")

        if published_at > collected_at:
            published_at = collected_at

        out.append(
            {
                "id": item_id,
                "title": title,
                "summary": summary,
                "category": category,
                "published_at": published_at,
                "collected_at": collected_at,
                "source": source,
                "source_url": source_url,
                "tags": tags,
            }
        )
    return out


def refresh_feed(feed: dict) -> Path:
    path = feed_path(feed)
    collected_at = tehran_jalali_now()
    previous = load_previous_titles(path)
    prompt = build_prompt(feed, collected_at, previous)
    raw, model_used = call_gemini_with_fallback(prompt, feed["system"])
    parsed = extract_json_object(raw)
    items = normalize_items(parsed.get("items"), collected_at, feed["id_prefix"])

    doc = {
        "version": "1.4",
        "topic": feed["topic"],
        "calendar": "jalali",
        "timezone": "Asia/Tehran",
        "interval_hours": 2,
        "count": 10,
        "provider": f"Gemini/{model_used}",
        "updated_at": collected_at,
        "items": items,
    }
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {path} with {len(items)} items via Gemini/{model_used}",
        file=sys.stderr,
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh topic news JSON files via Gemini only."
    )
    parser.add_argument(
        "--only",
        help="Comma-separated feed keys to refresh "
        f"(choices: {', '.join(f['key'] for f in FEEDS)}). Default: all.",
    )
    return parser.parse_args()


def selected_feeds(only: str | None) -> list[dict]:
    if not only:
        return list(FEEDS)
    wanted = {part.strip().lower() for part in only.split(",") if part.strip()}
    known = {f["key"] for f in FEEDS}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"Unknown feed keys: {', '.join(unknown)}")
    return [f for f in FEEDS if f["key"] in wanted]


def main() -> int:
    args = parse_args()
    feeds = selected_feeds(args.only)
    for feed in feeds:
        print(f"=== Refreshing {feed['filename']} ({feed['key']}) ===", file=sys.stderr)
        refresh_feed(feed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
