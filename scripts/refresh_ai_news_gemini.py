#!/usr/bin/env python3
"""Refresh news.json using ONLY the Gemini API.

Looks for API key in env (in order): Gemini, GEMINI_API_KEY, GOOGLE_API_KEY.
Does not call Cursor models or web search.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jdatetime

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
GEMINI_BASE = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
NEWS_PATH = REPO_ROOT / "news.json"
TEHRAN = ZoneInfo("Asia/Tehran")
CATEGORIES = {"news", "product", "regulation", "research", "topic", "sideline"}


def resolve_api_key() -> str:
    for name in ("Gemini", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def tehran_jalali_now() -> str:
    now = datetime.now(TEHRAN)
    return jdatetime.datetime.fromgregorian(datetime=now).strftime("%Y/%m/%d %H:%M")


def load_previous_titles() -> list[str]:
    if not NEWS_PATH.exists():
        return []
    try:
        data = json.loads(NEWS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    titles = []
    for item in data.get("items") or []:
        title = item.get("title")
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return titles


def build_prompt(collected_at: str, previous_titles: list[str]) -> str:
    avoid = "\n".join(f"- {t}" for t in previous_titles[:10]) or "- (none)"
    return f"""You are a news desk for artificial intelligence.

Return ONLY a valid JSON object (no markdown fences, no commentary) with exactly this shape:
{{
  "items": [
    {{
      "id": "ai-YYYYMMDD-NN",
      "title": "Persian title",
      "summary": "Persian summary, about 280-350 characters (roughly 3-4 sentences)",
      "category": "news|product|regulation|research|topic|sideline",
      "published_at": "YYYY/MM/DD HH:MM",
      "source": "outlet or org name",
      "source_url": "https://...",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

Rules:
- Exactly 10 items about AI (models, labs, agents, research, regulation, industry sidelines).
- Prefer fresh, real recent stories with real canonical source_url values.
- title and summary MUST be Persian.
- Each summary MUST be substantially longer than one short line: target ~280-350 Persian characters (about 3-4 sentences), roughly double a single-sentence blurb. Include what happened, why it matters, and one concrete detail (product/model/org/number) when available. Do not pad with filler.
- published_at MUST be Jalali calendar + Asia/Tehran as YYYY/MM/DD HH:MM.
- published_at must not be after collected_at={collected_at}.
- id must be unique like ai-20261007-01 .. ai-20261007-10 using today's Gregorian date in the id prefix.
- Avoid repeating these previous titles when possible:
{avoid}
"""


def call_gemini(prompt: str) -> str:
    api_key = resolve_api_key()
    if not api_key:
        raise SystemExit(
            "Gemini API key missing. Add a Cloud Agent secret named "
            "`Gemini` or `GEMINI_API_KEY`, then start a NEW agent."
        )

    body = {
        "systemInstruction": {
            "parts": [
                {
                    "text": "You output only strict JSON for an AI news feed. No markdown."
                }
            ]
        },
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }
    req = urllib.request.Request(
        GEMINI_BASE,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "News-Json-GeminiRefresh/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Gemini API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Gemini API network error: {exc}") from exc

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        content = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit(f"Unexpected Gemini response shape: {payload!r}") from exc

    if not content.strip():
        raise SystemExit(f"Empty Gemini content: {payload!r}")
    return content.strip()


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


def normalize_items(raw_items: list, collected_at: str) -> list[dict]:
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

        item_id = str(item.get("id") or f"ai-{gregorian}-{idx:02d}").strip()
        if item_id in seen_ids:
            item_id = f"ai-{gregorian}-{idx:02d}"
        seen_ids.add(item_id)

        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        category = str(item.get("category") or "news").strip().lower()
        if category not in CATEGORIES:
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


def main() -> int:
    collected_at = tehran_jalali_now()
    previous = load_previous_titles()
    prompt = build_prompt(collected_at, previous)
    print(f"Calling Gemini model {GEMINI_MODEL} ...", file=sys.stderr)
    raw = call_gemini(prompt)
    parsed = extract_json_object(raw)
    items = normalize_items(parsed.get("items"), collected_at)

    doc = {
        "version": "1.4",
        "topic": "artificial-intelligence",
        "calendar": "jalali",
        "timezone": "Asia/Tehran",
        "interval_hours": 2,
        "count": 10,
        "provider": f"Gemini/{GEMINI_MODEL}",
        "updated_at": collected_at,
        "items": items,
    }
    NEWS_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {NEWS_PATH} with {len(items)} items via Gemini/{GEMINI_MODEL}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
