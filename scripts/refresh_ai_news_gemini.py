#!/usr/bin/env python3
"""Refresh topic news JSON files using ONLY the Gemini API.

Looks for API key in env (in order): Gemini, GEMINI_API_KEY, GOOGLE_API_KEY.
Does not call Cursor models or web search.

Behavior:
  - Hourly cycles APPEND new items to each feed.
  - During Asia/Tehran 00:00-00:59, CLEAR each feed and write a fresh batch.
  - That daily clear+rebuild cycle repeats every 24 hours.

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

DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ("gemini-3.8-flash",)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
REPO_ROOT = Path(__file__).resolve().parents[1]
TEHRAN = ZoneInfo("Asia/Tehran")
ITEMS_PER_CYCLE = 10
INTERVAL_HOURS = 1

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


def is_tehran_midnight_reset() -> bool:
    """True during 00:00-00:59 Asia/Tehran: wipe feeds and start a new day."""
    return datetime.now(TEHRAN).hour == 0


def load_existing_doc(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_existing_items(path: Path) -> list[dict]:
    data = load_existing_doc(path)
    if not data:
        return []
    items = data.get("items") or []
    return [item for item in items if isinstance(item, dict)]


def load_previous_titles(path: Path) -> list[str]:
    titles = []
    for item in load_existing_items(path):
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return titles


def next_item_start_index(existing: list[dict], id_prefix: str, gregorian: str) -> int:
    prefix = f"{id_prefix}-{gregorian}-"
    max_n = 0
    for item in existing:
        item_id = str(item.get("id") or "")
        if not item_id.startswith(prefix):
            continue
        suffix = item_id[len(prefix) :]
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return max(max_n, len(existing)) + 1


def build_prompt(
    feed: dict,
    collected_at: str,
    previous_titles: list[str],
    *,
    reset_day: bool,
) -> str:
    avoid_n = 80 if previous_titles else 0
    avoid = "\n".join(f"- {t}" for t in previous_titles[:avoid_n]) or "- (none)"
    prefix = feed["id_prefix"]
    mode = (
        "This is the daily midnight reset (Asia/Tehran 00:00). Produce a fresh batch; older items are cleared."
        if reset_day
        else "This is an hourly append cycle. Produce NEW stories only; they will be appended to existing items."
    )
    return f"""You are a news desk for {feed["desk"]}.

Return ONLY a valid JSON object (no markdown fences, no commentary) with exactly this shape:
{{
  "items": [
    {{
      "id": "{prefix}-YYYYMMDD-NNN",
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
- {mode}
- Produce exactly {ITEMS_PER_CYCLE} items.
- {feed["scope"]}
- Prefer fresh, real recent stories with real canonical source_url values.
- title and summary MUST be Persian.
- Each summary MUST be substantially longer than one short line: target ~280-350 Persian characters (about 3-4 sentences), roughly double a single-sentence blurb. Include what happened, why it matters, and one concrete detail (product/model/org/number) when available. Do not pad with filler.
- published_at MUST be Jalali calendar + Asia/Tehran as YYYY/MM/DD HH:MM.
- published_at must not be after collected_at={collected_at}.
- id must be unique like {prefix}-20261007-001 using today's Gregorian date in the id prefix.
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
                # 404 / daily quota: skip to next model immediately
                if "HTTP 404" in msg or "HTTP 429" in msg:
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


def normalize_items(
    raw_items: list,
    collected_at: str,
    id_prefix: str,
    *,
    start_index: int = 1,
) -> list[dict]:
    if not isinstance(raw_items, list) or len(raw_items) != ITEMS_PER_CYCLE:
        raise ValueError(
            f"expected exactly {ITEMS_PER_CYCLE} items, got "
            f"{len(raw_items) if isinstance(raw_items, list) else type(raw_items)}"
        )

    gregorian = datetime.now(TEHRAN).strftime("%Y%m%d")
    out: list[dict] = []
    seen_ids: set[str] = set()

    for offset, item in enumerate(raw_items):
        idx = start_index + offset
        if not isinstance(item, dict):
            raise ValueError(f"item {idx} is not an object")

        default_id = f"{id_prefix}-{gregorian}-{idx:03d}"
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
    reset_day = is_tehran_midnight_reset()
    existing = [] if reset_day else load_existing_items(path)
    previous_titles = [str(i.get("title") or "").strip() for i in existing if i.get("title")]
    if reset_day:
        print(f"Midnight reset (Asia/Tehran): clearing {path.name}", file=sys.stderr)
    else:
        print(
            f"Hourly append for {path.name}: keeping {len(existing)} existing items",
            file=sys.stderr,
        )

    prompt = build_prompt(
        feed, collected_at, previous_titles, reset_day=reset_day
    )
    raw, model_used = call_gemini_with_fallback(prompt, feed["system"])
    parsed = extract_json_object(raw)
    gregorian = datetime.now(TEHRAN).strftime("%Y%m%d")
    start_index = 1 if reset_day else next_item_start_index(
        existing, feed["id_prefix"], gregorian
    )
    new_items = normalize_items(
        parsed.get("items"),
        collected_at,
        feed["id_prefix"],
        start_index=start_index,
    )

    existing_title_set = {t.strip().lower() for t in previous_titles if t.strip()}
    unique_new = [
        item
        for item in new_items
        if item["title"].strip().lower() not in existing_title_set
    ]
    items = unique_new if reset_day else existing + unique_new

    doc = {
        "version": "1.5",
        "topic": feed["topic"],
        "calendar": "jalali",
        "timezone": "Asia/Tehran",
        "interval_hours": INTERVAL_HOURS,
        "count": len(items),
        "provider": f"Gemini/{model_used}",
        "updated_at": collected_at,
        "reset_policy": "clear_at_tehran_midnight_append_hourly",
        "items": items,
    }
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {path} with {len(items)} items "
        f"(+{len(unique_new)} new) via Gemini/{model_used}",
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
