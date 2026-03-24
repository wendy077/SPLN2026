import json
from collections import defaultdict
from pathlib import Path

import spacy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = PROJECT_ROOT / "sources.json"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"


def load_sources(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def extract_entities(text: str, nlp) -> dict:
    doc = nlp(text)

    grouped = defaultdict(dict)

    for ent in doc.ents:
        entity_text = " ".join(ent.text.split()).strip()
        label = ent.label_

        if not entity_text:
            continue

        if entity_text in grouped[label]:
            grouped[label][entity_text] += 1
        else:
            grouped[label][entity_text] = 1

    result = {}
    for label, entities in grouped.items():
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: (-x[1], x[0].lower())
        )
        result[label] = [
            {"text": entity, "count": count}
            for entity, count in sorted_entities
        ]

    return result


def process_source(source: dict, nlp) -> None:
    source_id = source["id"]
    input_path = CLEAN_DIR / f"{source_id}.txt"
    output_path = CLEAN_DIR / f"{source_id}_entities.json"

    text = read_text(input_path)
    entities = extract_entities(text, nlp)
    write_json(output_path, entities)

    total = sum(len(v) for v in entities.values())
    print(f"[OK] {source_id}: {total} entidades únicas -> {output_path}")


def main():
    sources = load_sources(SOURCES_FILE)

    # modelo large em vez de small — melhor precisão para textos técnicos
    nlp = spacy.load("en_core_web_lg")

    for source in sources:
        process_source(source, nlp)


if __name__ == "__main__":
    main()