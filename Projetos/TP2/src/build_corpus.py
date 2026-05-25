import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS_PATH = PROJECT_ROOT / "seeds.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "TP2-IR-Music/0.1 (academic project for SPLN 2025/26; contact: pg61534@alunos.uminho.pt)"
}

REQUEST_SLEEP = 0.5
MIN_EXTERNAL_TEXT_CHARS = 1800
MIN_WIKI_TEXT_CHARS = 800

GOOD_EXTERNAL_DOMAINS = [
    "britannica.com",
    "grammy.com",
    "rockhall.com",
    "rollingstone.com",
    "billboard.com",
    "pitchfork.com",
    "allmusic.com",
    "nme.com",
    "theguardian.com",
    "officialcharts.com",
    "mtv.com",
    "coachella.com",
    "glastonburyfestivals.co.uk",
    "lollapalooza.com",
    "primaverasound.com",
    "rollingloud.com",
    "eurovision.tv",
]

BAD_URL_PATTERNS = [
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "spotify.com",
    "apple.com",
    "amazon.",
    "discogs.com",
    "musicbrainz.org",
    "wikidata.org",
    "commons.wikimedia.org",
    "web.archive.org",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".mp3",
    ".mp4",
]


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_seeds() -> list[dict]:
    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    records = []

    for section in ["artists", "albums", "genres", "events"]:
        for item in data.get(section, []):
            item = dict(item)
            item["section"] = section
            records.append(item)

    return records


def wiki_query(params: dict) -> dict | None:
    try:
        response = requests.get(
            WIKI_API_URL,
            params=params,
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"[WARN] Wikipedia API error: {exc}")
        return None


def get_wikipedia_extract(title: str) -> tuple[str | None, str | None]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": True,
        "exsectionformat": "plain",
        "redirects": 1,
        "inprop": "url",
        "titles": title,
    }

    data = wiki_query(params)
    if not data:
        return None, None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None, None

    page = next(iter(pages.values()))
    if "missing" in page:
        return None, None

    text = clean_text(page.get("extract", "") or "")
    url = page.get("fullurl")

    # corta secções pouco úteis para QA
    cut_markers = [
        "References",
        "External links",
        "See also",
        "Further reading",
        "Bibliography",
    ]

    for marker in cut_markers:
        pattern = rf"\b{re.escape(marker)}\b"
        match = re.search(pattern, text)
        if match and match.start() > 500:
            text = text[: match.start()].strip()
            break

    return text, url


def get_external_links(title: str) -> list[str]:
    links = []
    params = {
        "action": "query",
        "format": "json",
        "prop": "extlinks",
        "ellimit": "max",
        "redirects": 1,
        "titles": title,
    }

    while True:
        data = wiki_query(params)
        if not data:
            break

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for link_obj in page.get("extlinks", []):
                url = link_obj.get("*") or link_obj.get("url")
                if url:
                    links.append(url)

        if "continue" not in data:
            break

        params.update(data["continue"])
        time.sleep(REQUEST_SLEEP)

    return links


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_bad_url(url: str) -> bool:
    lower = url.lower()
    return any(pattern in lower for pattern in BAD_URL_PATTERNS)


def external_link_score(url: str, entity_type: str) -> int:
    if is_bad_url(url):
        return -100

    domain = domain_of(url)
    score = 0

    for good_domain in GOOD_EXTERNAL_DOMAINS:
        if good_domain in domain:
            score += 20

    # pequenas preferências por tipo de entidade
    if entity_type == "artist":
        if any(d in domain for d in ["grammy.com", "rockhall.com", "britannica.com", "allmusic.com"]):
            score += 8

    if entity_type == "album":
        if any(d in domain for d in ["pitchfork.com", "rollingstone.com", "allmusic.com", "billboard.com", "nme.com"]):
            score += 8

    if entity_type == "genre":
        if any(d in domain for d in ["britannica.com", "allmusic.com"]):
            score += 8

    if entity_type in {"award", "festival", "event", "movement"}:
        if any(
            d in domain
            for d in [
                "grammy.com",
                "rockhall.com",
                "coachella.com",
                "glastonburyfestivals.co.uk",
                "lollapalooza.com",
                "primaverasound.com",
                "rollingloud.com",
                "eurovision.tv",
            ]
        ):
            score += 10

    return score


def select_candidate_external_links(links: list[str], entity_type: str, max_links: int = 8) -> list[str]:
    scored = []

    for url in links:
        score = external_link_score(url, entity_type)
        if score > 0:
            scored.append((score, url))

    scored.sort(reverse=True, key=lambda x: x[0])

    seen_domains = set()
    selected = []

    for _, url in scored:
        domain = domain_of(url)
        if domain in seen_domains:
            continue

        seen_domains.add(domain)
        selected.append(url)

        if len(selected) >= max_links:
            break

    return selected


def extract_external_text(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        if not text:
            return None

        text = clean_text(text)

        if len(text) < MIN_EXTERNAL_TEXT_CHARS:
            return None

        return text

    except Exception:
        return None


def choose_source(seed: dict) -> dict | None:
    name = seed["name"]
    wiki_title = seed["wiki_title"]
    entity_type = seed["type"]

    wiki_text, wiki_url = get_wikipedia_extract(wiki_title)
    time.sleep(REQUEST_SLEEP)

    external_links = get_external_links(wiki_title)
    candidate_links = select_candidate_external_links(external_links, entity_type)

    for url in candidate_links:
        text = extract_external_text(url)
        time.sleep(REQUEST_SLEEP)

        if text:
            return {
                "source_type": "external",
                "source": domain_of(url),
                "url": url,
                "text": text,
                "candidate_external_links": candidate_links,
                "wikipedia_url": wiki_url,
            }

    if wiki_text and len(wiki_text) >= MIN_WIKI_TEXT_CHARS:
        return {
            "source_type": "wikipedia",
            "source": "Wikipedia",
            "url": wiki_url,
            "text": wiki_text,
            "candidate_external_links": candidate_links,
            "wikipedia_url": wiki_url,
        }

    return None


def build_document(seed: dict, source_data: dict) -> dict:
    entity_type = seed["type"]
    name = seed["name"]

    doc_id = f"{entity_type}_{slugify(name)}"

    document = {
        "doc_id": doc_id,
        "title": name,
        "type": entity_type,
        "section": seed.get("section"),
        "source_type": source_data["source_type"],
        "source": source_data["source"],
        "url": source_data["url"],
        "wikipedia_url": source_data.get("wikipedia_url"),
        "text": source_data["text"],
    }

    if "artist" in seed:
        document["artist"] = seed["artist"]

    return document


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds()
    print(f"Loaded {len(seeds)} seeds")

    documents = []
    failures = []

    for seed in tqdm(seeds, desc="Building corpus"):
        source_data = choose_source(seed)

        if not source_data:
            failures.append(seed)
            print(f"[FAIL] {seed['name']} ({seed['wiki_title']})")
            continue

        doc = build_document(seed, source_data)
        documents.append(doc)

        print(
            f"[OK] {doc['title']} | {doc['source_type']} | "
            f"{doc['source']} | {len(doc['text'])} chars"
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Documents saved: {len(documents)}")
    print(f"Output: {OUTPUT_PATH}")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"- {item['name']} | wiki_title: {item['wiki_title']}")


if __name__ == "__main__":
    main()