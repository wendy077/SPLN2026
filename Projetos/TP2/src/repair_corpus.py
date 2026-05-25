import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "corpus.jsonl"
AUDIT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus_audit.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "corpus_repaired.jsonl"

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "TP2-IR-Music/0.1 (academic project for SPLN 2025/26; student: Mariana Miguel Pinto; University of Minho)"
}

REQUEST_SLEEP = 0.2


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


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, docs: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_seeds_by_doc_id() -> dict:
    seeds_path = PROJECT_ROOT / "seeds.yaml"

    with open(seeds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mapping = {}

    for section in ["artists", "albums", "genres", "events"]:
        for item in data.get(section, []):
            item = dict(item)
            item["section"] = section
            doc_id = f"{item['type']}_{slugify(item['name'])}"
            mapping[doc_id] = item

    return mapping


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

    wait_times = [5, 15, 30, 60, 90]

    for attempt, wait_time in enumerate(wait_times, start=1):
        try:
            response = requests.get(
                WIKI_API_URL,
                params=params,
                headers=HEADERS,
                timeout=30,
            )

            if response.status_code == 429:
                print(f"[WARN] Rate limit for {title}. Waiting {wait_time}s... attempt {attempt}/5")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None, None

            page = next(iter(pages.values()))
            if "missing" in page:
                return None, None

            text = clean_text(page.get("extract", "") or "")
            url = page.get("fullurl")

            cut_markers = [
                "References",
                "External links",
                "See also",
                "Further reading",
                "Bibliography",
            ]

            for marker in cut_markers:
                match = re.search(rf"\b{re.escape(marker)}\b", text)
                if match and match.start() > 500:
                    text = text[: match.start()].strip()
                    break

            return text, url

        except Exception as exc:
            print(f"[WARN] Wikipedia API error for {title}: {exc}")
            time.sleep(wait_time)

    return None, None


def main() -> None:
    docs = load_jsonl(CORPUS_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    seeds_by_doc_id = load_seeds_by_doc_id()

    flagged_doc_ids = set(
        audit[audit["warnings"].notna() & (audit["warnings"].astype(str).str.strip() != "")]["doc_id"]
    )

    print(f"Documents in corpus: {len(docs)}")
    print(f"Documents to replace with Wikipedia: {len(flagged_doc_ids)}")

    repaired_docs = []
    replaced = []
    failed = []

    for doc in docs:
        doc_id = doc["doc_id"]

        if doc_id not in flagged_doc_ids:
            repaired_docs.append(doc)
            continue

        seed = seeds_by_doc_id.get(doc_id)

        if not seed:
            print(f"[FAIL] No seed found for {doc_id}")
            failed.append(doc_id)
            repaired_docs.append(doc)
            continue

        wiki_title = seed["wiki_title"]
        wiki_text, wiki_url = get_wikipedia_extract(wiki_title)
        time.sleep(REQUEST_SLEEP)

        if not wiki_text or len(wiki_text) < 800:
            print(f"[FAIL] Wikipedia text too short/missing for {doc['title']} ({wiki_title})")
            failed.append(doc_id)
            repaired_docs.append(doc)
            continue

        new_doc = dict(doc)
        new_doc["source_type"] = "wikipedia"
        new_doc["source"] = "Wikipedia"
        new_doc["url"] = wiki_url
        new_doc["wikipedia_url"] = wiki_url
        new_doc["text"] = wiki_text

        repaired_docs.append(new_doc)
        replaced.append(doc_id)

        print(f"[OK] Replaced {doc['title']} with Wikipedia ({len(wiki_text)} chars)")

    save_jsonl(OUTPUT_PATH, repaired_docs)

    print()
    print(f"Saved repaired corpus to: {OUTPUT_PATH}")
    print(f"Replaced: {len(replaced)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("Failed doc_ids:")
        for doc_id in failed:
            print("-", doc_id)


if __name__ == "__main__":
    main()