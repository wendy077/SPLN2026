import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

ALLOWED_LABELS = {"ORG", "LOC", "GPE", "DATE", "PERSON", "PRODUCT"}


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def is_clean_entity(text: str) -> bool:
    t = text.strip()

    if not t:
        return False

    if "•" in t:
        return False

    if len(t) < 2:
        return False

    # lixo visual comum
    if re.search(r"(?:\b[A-Za-z]\b\s+){4,}", t):
        return False

    # evitar fragmentos muito estranhos
    if re.search(r"[^\w\s\-\+\.'/&()]", t):
        return False

    return True


def normalize_entity_text(text: str) -> str:
    t = " ".join(text.split()).strip()

    replacements = {
        "the Solar System": "Solar System",
        "the Red Planet": "Red Planet",
        "the International Space Station": "International Space Station",
        "the Artemis Accords": "Artemis Accords",
        "the ESA Council": "ESA Council",
        "the European Space Summit": "European Space Summit",
        "the Mars Sample Return": "Mars Sample Return",
    }

    return replacements.get(t, t)


def normalize_entity_label(label: str, text: str) -> str:
    if text in {"Moon", "Mars", "Earth", "LEO", "Solar System", "Red Planet"}:
        return "LOC"

    if text in {"Europe", "Russia", "US", "China", "France", "Toulouse"}:
        return "GPE"

    return label


def should_skip_entity(text: str) -> bool:
    bad_entities = {
        "Agency",
        "Solar",
        "Lunar",
        "Strategy Roadmap",
        "Member States",
        "Participating States",
        "Hermes",
        "Columbus",
        "ESA Agenda 2025",
        "Neptune",  
        "Sun",       
        "Buran",    
        "Argonaut",  
        "ExoMars",   
        "decades",  

    }
    return text in bad_entities


def process_source(source: dict) -> None:
    source_id = source["id"]
    input_path = CLEAN_DIR / f"{source_id}_entities.json"
    output_path = CLEAN_DIR / f"{source_id}_entities_filtered.json"

    data = read_json(input_path)

    grouped = defaultdict(list)
    seen = set()

    for original_label, entities in data.items():
        for item in entities:
            text = normalize_entity_text(item["text"])
            count = item["count"]

            if not is_clean_entity(text):
                continue

            if should_skip_entity(text):
                continue

            new_label = normalize_entity_label(original_label, text)

            if new_label not in ALLOWED_LABELS:
                continue

            key = (new_label, text.lower())
            if key in seen:
                continue
            seen.add(key)

            grouped[new_label].append({
                "text": text,
                "count": count
            })

    filtered = {}
    for label, entities in grouped.items():
        entities.sort(key=lambda x: (-x["count"], x["text"].lower()))
        filtered[label] = entities[:15]

    write_json(output_path, filtered)
    print(f"[OK] {source_id}: entidades filtradas -> {output_path}")


def main():
    sources = load_sources(SOURCES_FILE)
    for source in sources:
        process_source(source)


if __name__ == "__main__":
    main()